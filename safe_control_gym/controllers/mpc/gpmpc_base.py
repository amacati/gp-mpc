"""Model Predictive Control with a Gaussian Process model.

Based on:
    * L. Hewing, J. Kabzan and M. N. Zeilinger, 'Cautious Model Predictive Control Using Gaussian Process Regression,'
     in IEEE Transactions on Control Systems Technology, vol. 28, no. 6, pp. 2736-2743, Nov. 2020, doi: 10.1109/TCST.2019.2949757.

Implementation details:
    1. The previous time step MPC solution is used to compute the set constraints and GP dynamics rollout.
       Here, the dynamics are rolled out using the Mean Equivelence method, the fastest, but least accurate.
    2. The GP is approximated using the Fully Independent Training Conditional (FITC) outlined in
        * J. Quinonero-Candela, C. E. Rasmussen, and R. Herbrich, “A unifying view of sparse approximate Gaussian process regression,”
          Journal of Machine Learning Research, vol. 6, pp. 1935–1959, 2005.
          https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
        * E. Snelson and Z. Ghahramani, “Sparse gaussian processes using pseudo-inputs,” in Advances in Neural Information Processing
          Systems, Y. Weiss, B. Scholkopf, and J. C. Platt, Eds., 2006, pp. 1257–1264.
       and the inducing points are the previous MPC solution.
    3. Each dimension of the learned error dynamics is an independent Zero Mean SE Kernel GP.
"""
import os
from functools import partial
from termcolor import colored
from abc import abstractmethod
from copy import deepcopy

import casadi as cs
import numpy as np
import torch

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.mpc_utils import (
    compute_discrete_lqr_gain_from_cont_linear_system,
    compute_state_rmse,
    get_cost_weight_matrix,
    reset_constraints,
    rk_discrete,
)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import (
    GENERAL_CONSTRAINTS,
    create_constraint_list,
)
from safe_control_gym.utils.utils import timing
from numpy.linalg import LinAlgError

import gpytorch
import munch
import scipy
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs

from safe_control_gym.controllers.mpc.gp_utils import (
    GaussianProcessCollection,
    ZeroMeanIndependentGPModel,
    covMatern52ard,
    covSEard,
    covSE_single,
    kmeans_centriods,
)
from safe_control_gym.experiments.base_experiment import BaseExperiment


class GPMPC:
    """MPC with Gaussian Process as dynamics residual."""

    def __init__(
        self,
        env_func,
        seed: int = 1337,
        horizon: int = 5,
        q_mpc: list = [1],
        r_mpc: list = [1],
        constraint_tol: float = 1e-8,
        additional_constraints: list = None,
        soft_constraints: dict = None,
        warmstart: bool = True,
        train_iterations: int = None,
        test_data_ratio: float = 0.2,
        overwrite_saved_data: bool = True,
        optimization_iterations: list = None,
        learning_rate: list = None,
        use_gpu: bool = False,
        gp_model_path: str = None,
        kernel: str = "Matern",
        prob: float = 0.955,
        initial_rollout_std: float = 0.005,
        input_mask: list = None,
        target_mask: list = None,
        gp_approx: str = "mean_eq",
        sparse_gp: bool = False,
        n_ind_points: int = 150,
        inducing_point_selection_method: str = "kmeans",
        recalc_inducing_points_at_every_step: bool = False,
        online_learning: bool = False,
        prior_info: dict = None,
        # inertial_prop: list = [1.0],
        prior_param_coeff: float = 1.0,
        terminate_run_on_done: bool = True,
        output_dir: str = "results/temp",
        plot_trained_gp: bool = False,
        **kwargs,
    ):
        """Initialize GP-MPC.

        Args:
            env_func (gym.Env): functionalized initialization of the environment.
            seed (int): random seed.
            horizon (int): MPC planning horizon.
            Q, R (np.array): cost weight matrix.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            train_iterations (int): the number of training examples to use for each dimension of the GP.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            optimization_iterations (list): the number of optimization iterations for each dimension of the GP.
            learning_rate (list): the learning rate for training each dimension of the GP.
            use_gpu (bool): use GPU while training the gp.
            gp_model_path (str): path to a pretrained GP model. If None, will train a new one.
            kernel (str): 'Matern' or 'RBF' kernel.
            output_dir (str): directory to store model and results.
            prob (float): desired probabilistic safety level.
            initial_rollout_std (float): the initial std (across all states) for the mean_eq rollout.
            prior_info (dict): Dictionary specifiy the algorithms prior model parameters.
            prior_param_coeff (float): constant multiplying factor to adjust the prior model intertial properties.
            input_mask (list): list of which input dimensions to use in GP model. If None, all are used.
            target_mask (list): list of which output dimensions to use in the GP model. If None, all are used.
            gp_approx (str): 'mean_eq' used mean equivalence rollout for the GP dynamics. Only one that works currently.
            sparse_gp (bool): True to use sparse GP approximations, otherwise no spare approximation is used.
            n_ind_points (int): Number of inducing points to use got the FTIC gp approximation.
            inducing_point_selection_method (str): kmeans for kmeans clustering, 'random' for random.
            recalc_inducing_points_at_every_step (bool): True to recompute the gp approx at every time step.
            online_learning (bool): if true, GP kernel values will be updated using past trajectory values.
            additional_constraints (list): list of Constraint objects defining additional constraints to be used.
        """

        if prior_info is None or prior_info == {}:
            raise ValueError(
                "GPMPC requires prior_prop to be defined. You may use the real mass properties and then use prior_param_coeff to modify them accordingly."
            )
        prior_info["prior_prop"].update(
            (prop, val * prior_param_coeff)
            for prop, val in prior_info["prior_prop"].items()
        )
        self.prior_env_func = partial(env_func, inertial_prop=prior_info["prior_prop"])
        if soft_constraints is None:
            self.soft_constraints_params = {
                "gp_soft_constraints": False,
                "gp_soft_constraints_coeff": 0,
                "prior_soft_constraints": False,
                "prior_soft_constraints_coeff": 0,
            }
        else:
            self.soft_constraints_params = soft_constraints

        self.sparse_gp = sparse_gp
        self.env_func = env_func
        self.training = True
        self.checkpoint_path = "temp/model_latest.pt"
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cpu" if self.use_gpu is False else "cuda"
        self.seed = seed
        self.prior_info = {}
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.setup_results_dict()

        for k, v in locals().items():
            if k != "self" and k != "kwargs" and "__" not in k:
                self.__dict__.update({k: v})

        # Task.
        self.env = env_func()
        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(
                additional_constraints, GENERAL_CONSTRAINTS, self.env
            )
            self.additional_constraints = additional_ConstraintsList.constraints
            (
                self.constraints,
                self.state_constraints_sym,
                self.input_constraints_sym,
            ) = reset_constraints(
                self.env.constraints.constraints + self.additional_constraints
            )
        else:
            (
                self.constraints,
                self.state_constraints_sym,
                self.input_constraints_sym,
            ) = reset_constraints(self.env.constraints.constraints)
            self.additional_constraints = []
        # Model parameters
        self.model = self.get_prior(self.env)
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        self.constraint_tol = constraint_tol
        self.soft_constraints = soft_constraints
        self.soft_penalty = 10000.0
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        self.X_EQ = self.env.X_GOAL
        self.U_EQ = self.env.U_GOAL
        self.compute_initial_guess_method = "ipopt"
        self.use_lqr_gain_and_terminal_cost = False
        self.init_solver = "ipopt"
        self.solver = "ipopt"

        # Setup environments.
        # TODO: This is a temporary fix
        # the parent class creates a connection to the env
        # but the same handle is overwritten here. Therefore,
        # when call ctrl.close(), the env initialized in the
        # parent class would not be closed properly.
        if hasattr(self, "env"):
            self.env.close()
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None

        # GP and training parameters.
        self.gaussian_process = None
        self.train_iterations = train_iterations
        self.test_data_ratio = test_data_ratio
        self.overwrite_saved_data = overwrite_saved_data
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.gp_model_path = gp_model_path
        self.kernel = kernel
        self.prob = prob
        if input_mask is None:
            self.input_mask = np.arange(self.model.nx + self.model.nu).tolist()
        else:
            self.input_mask = input_mask
        if target_mask is None:
            self.target_mask = np.arange(self.model.nx).tolist()
        else:
            self.target_mask = target_mask
        Bd = np.eye(self.model.nx)
        self.Bd = Bd[:, self.target_mask]
        self.gp_approx = gp_approx
        self.n_ind_points = n_ind_points
        assert inducing_point_selection_method in [
            "kmeans",
            "random",
        ], "[Error]: Inducing method choice is incorrect."
        self.inducing_point_selection_method = inducing_point_selection_method
        self.recalc_inducing_points_at_every_step = recalc_inducing_points_at_every_step
        self.online_learning = online_learning
        self.initial_rollout_std = initial_rollout_std
        self.plot_trained_gp = plot_trained_gp

        # MPC params
        self.gp_soft_constraints = self.soft_constraints_params["gp_soft_constraints"]
        self.gp_soft_constraints_coeff = self.soft_constraints_params[
            "gp_soft_constraints_coeff"
        ]

        self.last_obs = None
        self.last_action = None

    def get_prior(self, env, prior_info={}):
        """Fetch the prior model from the env for the controller.

        Note there's a default env.symbolic when each each env is created.
        To make a different prior model, do the following when initializing a ctrl::

            self.env = env_func()
            self.model = self.get_prior(self.env)

        Besides the env config `base.yaml` and ctrl config `mpc.yaml`,
        you can define an additional prior config `prior.yaml` that looks like::

            algo_config:
                prior_info:
                    prior_prop:
                        M: 0.03
                        Iyy: 0.00003
                    randomize_prior_prop: False
                    prior_prop_rand_info: {}

        and to ensure the resulting config.algo_config contains both the params
        from ctrl config and prior config, chain them to the --overrides like:

            python experiment.py --algo mpc --task quadrotor --overrides base.yaml mpc.yaml prior.yaml ...

        Also note we look for prior_info from the incoming function arg first, then the ctrl itself.
        this allows changing the prior model during learning by calling::

            new_model = self.get_prior(same_env, new_prior_info)

        Alternatively, you can overwrite this method and use your own format for prior_info
        to customize how you get/change the prior model for your controller.

        Args:
            env (BenchmarkEnv): the environment to fetch prior model from.
            prior_info (dict): specifies the prior properties or other things to
                overwrite the default prior model in the env.

        Returns:
            SymbolicModel: CasAdi prior model.
        """
        if not prior_info:
            prior_info = getattr(self, "prior_info", {})
        prior_prop = prior_info.get("prior_prop", {})

        # randomize prior prop, similar to randomizing the inertial_prop in BenchmarkEnv
        # this can simulate the estimation errors in the prior model
        randomize_prior_prop = prior_info.get("randomize_prior_prop", False)
        prior_prop_rand_info = prior_info.get("prior_prop_rand_info", {})
        if randomize_prior_prop and prior_prop_rand_info:
            # check keys, this is due to the current implementation of BenchmarkEnv._randomize_values_by_info()
            for k in prior_prop_rand_info:
                assert (
                    k in prior_prop
                ), "A prior param to randomize does not have a base value in prior_prop."
            prior_prop = env._randomize_values_by_info(prior_prop, prior_prop_rand_info)

        # Note we only reset the symbolic model when prior_prop is nonempty
        if prior_prop:
            env._setup_symbolic(prior_prop=prior_prop)

        # Note this ensures the env can still access the prior model,
        # which is used to get quadratic costs in env.step()
        prior_model = env.symbolic
        return prior_model

    def setup_prior_dynamics(self):
        """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics."""
        # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
        A, B = discretize_linear_system(
            self.prior_ctrl.dfdx, self.prior_ctrl.dfdu, self.dt
        )
        Q_lqr = self.Q
        R_lqr = self.R
        P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
        btp = np.dot(B.T, P)
        self.lqr_gain = -np.dot(np.linalg.inv(self.R + np.dot(btp, B)), np.dot(btp, A))
        self.discrete_dfdx = A
        self.discrete_dfdu = B

        x = cs.MX.sym("x", self.model.nx)
        u = cs.MX.sym("u", self.model.nu)
        z = cs.vertcat(x, u)
        self.fd_func = self.model.fd_func
        residual = (
            self.fd_func(x0=x, p=u)["xf"] - self.prior_dynamics_func(x0=x, p=u)["xf"]
        )
        self.residual_func = cs.Function("residual_func", [z], [residual])
        self.fc_func = self.model.fc_func  # argument x, u

    def create_sparse_GP_machinery(self, n_ind_points):
        """This setups the gaussian process approximations for FITC formulation."""
        (
            lengthscales,
            signal_var,
            noise_var,
            gp_K_plus_noise,
        ) = self.gaussian_process.get_hyperparameters(as_numpy=True)
        self.length_scales = lengthscales.squeeze()
        self.signal_var = signal_var.squeeze()
        self.noise_var = noise_var.squeeze()
        self.gp_K_plus_noise = (
            gp_K_plus_noise  # (target_dim, n_data_points, n_data_points)
        )
        Nx = len(self.input_mask)
        Ny = len(self.target_mask)
        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind, length scales and signal variance.
        # We need the CasADI version of this so that it can by symbolically differentiated in in the MPC optimization.
        z1 = cs.SX.sym("z1", Nx)
        z2 = cs.SX.sym("z2", Nx)
        ell_s = cs.SX.sym("ell", Nx)
        sf2_s = cs.SX.sym("sf2")
        z_ind = cs.SX.sym("z_ind", n_ind_points, Nx)
        ks = cs.SX.zeros(1, n_ind_points)

        if self.kernel == "Matern":
            covMatern = cs.Function(
                "covMatern",
                [z1, z2, ell_s, sf2_s],
                [covMatern52ard(z1, z2, ell_s, sf2_s)],
            )
            for i in range(n_ind_points):
                ks[i] = covMatern(z1, z_ind[i, :], ell_s, sf2_s)
        elif self.kernel == "RBF":
            covSE = cs.Function(
                "covSE", [z1, z2, ell_s, sf2_s], [covSEard(z1, z2, ell_s, sf2_s)]
            )
            for i in range(n_ind_points):
                ks[i] = covSE(z1, z_ind[i, :], ell_s, sf2_s)
        elif self.kernel == "RBF_single":
            covSE = cs.Function(
                "covSE", [z1, z2, ell_s, sf2_s], [covSE_single(z1, z2, ell_s, sf2_s)]
            )
            for i in range(n_ind_points):
                ks[i] = covSE(z1, z_ind[i, :], ell_s, sf2_s)
        else:
            raise NotImplementedError("Kernel type not implemented.")
        ks_func = cs.Function("K_s", [z1, z_ind, ell_s, sf2_s], [ks])
        K_z_zind = cs.SX.zeros(Ny, n_ind_points)
        for i in range(Ny):
            if self.kernel == "RBF_single":
                K_z_zind[i, :] = ks_func(
                    z1, z_ind, self.length_scales[i], self.signal_var[i]
                )
            else:
                K_z_zind[i, :] = ks_func(
                    z1, z_ind, self.length_scales[i, :], self.signal_var[i]
                )

        # This will be mulitplied by the mean_post_factor computed at every time step to compute the approximate mean.
        self.K_z_zind_func = cs.Function(
            "K_z_zind", [z1, z_ind], [K_z_zind], ["z1", "z2"], ["K"]
        )

    def preprocess_training_data(self, x_seq, u_seq, x_next_seq):
        """Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_next_seq (list): next state sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact that
        # it is linearized about an eq using self.X_GOAL and self.U_GOAL.
        x_pred_seq = self.prior_dynamics_func(x0=x_seq.T, p=u_seq.T)["xf"].toarray()
        targets = (x_next_seq.T - x_pred_seq).transpose()
        inputs = np.hstack([x_seq, u_seq])  # (N, nx+nu).
        return inputs, targets

    def precompute_mean_post_factor_all_data(self):
        """If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        """
        dim_gp_outputs = len(self.target_mask)
        n_training_samples = self.train_data["train_targets"].shape[0]
        inputs = self.train_data["train_inputs"]
        targets = self.train_data["train_targets"]

        mean_post_factor = np.zeros((dim_gp_outputs, n_training_samples))
        for i in range(dim_gp_outputs):
            K_z_z = self.gaussian_process.K_plus_noise_inv[i]
            mean_post_factor[i] = (
                K_z_z.detach().numpy() @ targets[:, self.target_mask[i]]
            )

        return mean_post_factor, inputs[:, self.input_mask]

    def precompute_sparse_gp_values(self, n_ind_points):
        """Uses the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points (int): Number of inducing points.
        """
        n_data_points = self.gaussian_process.n_training_samples
        dim_gp_outputs = len(self.target_mask)
        inputs = self.train_data["train_inputs"]
        targets = self.train_data["train_targets"]

        # Get the inducing points.
        if False and self.x_prev is not None and self.u_prev is not None:
            # Use the previous MPC solution as in Hewing 2019.
            z_prev = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))
            z_prev = z_prev[:, self.input_mask]
            inds = self.env.np_random.choice(
                range(n_data_points), size=n_ind_points - self.T, replace=False
            )
            # z_ind = self.data_inputs[inds][:, self.input_mask]
            z_ind = np.vstack((z_prev, inputs[inds][:, self.input_mask]))
        else:
            # If there is no previous solution. Choose T random training set points.
            if self.inducing_point_selection_method == "kmeans":
                centroids = kmeans_centriods(
                    n_ind_points, inputs[:, self.input_mask], rand_state=self.seed
                )
                contiguous_masked_inputs = np.ascontiguousarray(
                    inputs[:, self.input_mask]
                )  # required for version sklearn later than 1.0.2
                inds, _ = pairwise_distances_argmin_min(
                    centroids, contiguous_masked_inputs
                )
                z_ind = inputs[inds][:, self.input_mask]
            elif self.inducing_point_selection_method == "random":
                inds = self.env.np_random.choice(
                    range(n_data_points), size=n_ind_points, replace=False
                )
                z_ind = inputs[inds][:, self.input_mask]
            else:
                raise ValueError(
                    "[Error]: gp_mpc.precompute_sparse_gp_values: Only 'kmeans' or 'random' allowed."
                )
        K_zind_zind = self.gaussian_process.kernel(
            torch.Tensor(z_ind).double()
        )  # (dim_gp_outputs, n_ind_points, n_ind_points)
        K_zind_zind_inv = self.gaussian_process.kernel_inv(
            torch.Tensor(z_ind).double()
        )  # (dim_gp_outputs, n_ind_points, n_ind_points)
        K_x_zind = self.gaussian_process.kernel(
            torch.from_numpy(inputs[:, self.input_mask]).double(),
            torch.tensor(z_ind).double(),
        )  # (dim_gp_outputs, n_data_points, n_ind_points)
        Q_X_X = K_x_zind @ torch.linalg.solve(
            K_zind_zind, K_x_zind.transpose(1, 2)
        )  # (dim_gp_outputs, n_data_points, n_data_points)
        Gamma = torch.diagonal(
            self.gaussian_process.K_plus_noise - Q_X_X, 0, 1, 2
        )  # (dim_gp_outputs, n_data_points)
        Gamma_inv = torch.diag_embed(
            1 / Gamma
        )  # (dim_gp_outputs, n_data_points, n_data_points)
        # TODO: Should inverse be used here instead? pinverse was more stable previsouly.
        Sigma_inv = (
            K_zind_zind + K_x_zind.transpose(1, 2) @ Gamma_inv @ K_x_zind
        )  # (dim_gp_outputs, n_ind_points, n_ind_points)
        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points))
        for i in range(dim_gp_outputs):
            mean_post_factor[i] = torch.linalg.solve(
                Sigma_inv[i],
                K_x_zind[i].T
                @ Gamma_inv[i]
                @ torch.from_numpy(targets[:, self.target_mask[i]]).double(),
            )
        return (
            mean_post_factor.detach().numpy(),
            Sigma_inv.detach().numpy(),
            K_zind_zind_inv.detach().numpy(),
            z_ind,
        )

    @timing
    def train_gp(
        self,
        input_data=None,
        target_data=None,
        gp_model=None,
        overwrite_saved_data: bool = None,
        train_hardware_data: bool = False,
    ):
        """Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            train_hardware_data (bool): True to train on hardware data. If true, will load the data and perform training.
        Returns:
            training_results (dict): Dictionary of the training results.
        """
        if gp_model is None and not train_hardware_data:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
        self.prior_ctrl.remove_constraints(self.prior_ctrl.additional_constraints)
        self.reset()
        if self.online_learning:
            input_data = np.zeros((self.train_iterations, len(self.input_mask)))
            target_data = np.zeros((self.train_iterations, len(self.target_mask)))
        if input_data is None and target_data is None:
            # If no input data is provided, we will generate self.train_iterations
            # + (1+self.test_ratio)* self.train_iterations number of training points. This will ensure the specified
            # number of train iterations are run, and the correct train-test data spilt is achieved.
            train_inputs = []
            train_targets = []
            train_info = []

            ############
            # Use Latin Hypercube Sampling to generate states withing environment bounds.
            lhs_sampler = Lhs(lhs_type="classic", criterion="maximin")
            # limits = [(self.env.INIT_STATE_RAND_INFO[key].low, self.env.INIT_STATE_RAND_INFO[key].high) for key in
            #          self.env.INIT_STATE_RAND_INFO]
            limits = [
                (
                    self.env.INIT_STATE_RAND_INFO["init_" + key]["low"],
                    self.env.INIT_STATE_RAND_INFO["init_" + key]["high"],
                )
                for key in self.env.STATE_LABELS
            ]
            # TODO: parameterize this if we actually want it.
            num_eq_samples = 0
            validation_iterations = int(
                self.train_iterations
                * (self.test_data_ratio / (1 - self.test_data_ratio))
            )
            samples = lhs_sampler.generate(
                limits,
                self.train_iterations + validation_iterations - num_eq_samples,
                random_state=self.seed,
            )
            if self.env.TASK == Task.STABILIZATION and num_eq_samples > 0:
                # TODO: choose if we want eq samples or not.
                delta_plus = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                delta_neg = np.array([0.1, 0.1, 0.1, 0.1, 0.03, 0.3])
                eq_limits = [
                    (
                        self.prior_ctrl.env.X_GOAL[eq] - delta_neg[eq],
                        self.prior_ctrl.env.X_GOAL[eq] + delta_plus[eq],
                    )
                    for eq in range(self.model.nx)
                ]
                eq_samples = lhs_sampler.generate(
                    eq_limits, num_eq_samples, random_state=self.seed
                )
                # samples = samples.append(eq_samples)
                init_state_samples = np.array(samples + eq_samples)
            else:
                init_state_samples = np.array(samples)
            input_limits = np.vstack(
                (
                    self.constraints.input_constraints[0].lower_bounds,
                    self.constraints.input_constraints[0].upper_bounds,
                )
            ).T
            input_samples = lhs_sampler.generate(
                input_limits,
                self.train_iterations + validation_iterations,
                random_state=self.seed,
            )
            input_samples = np.array(input_samples)  # not being used currently
            seeds = self.env.np_random.integers(
                0, 99999, size=self.train_iterations + validation_iterations
            )
            for i in range(self.train_iterations + validation_iterations):
                # For random initial state training.
                # init_state = init_state_samples[i,:]
                init_state = dict(
                    zip(self.env.INIT_STATE_RAND_INFO.keys(), init_state_samples[i, :])
                )
                # Collect data with prior controller.
                run_env = self.env_func(
                    init_state=init_state, randomized_init=False, seed=int(seeds[i])
                )
                episode_results = self.prior_ctrl.run(env=run_env, max_steps=1)
                run_env.close()
                x_obs = episode_results["obs"][-3:, :]
                u_seq = episode_results["action"][-1:, :]
                run_env.close()
                x_seq = x_obs[:-1, :]
                x_next_seq = x_obs[1:, :]
                train_inputs_i, train_targets_i = self.preprocess_training_data(
                    x_seq, u_seq, x_next_seq
                )
                train_inputs.append(train_inputs_i)
                train_targets.append(train_targets_i)
            train_inputs = np.vstack(train_inputs)
            train_targets = np.vstack(train_targets)
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        elif input_data is not None and target_data is not None:
            train_inputs = input_data
            train_targets = target_data
            if (
                self.data_inputs is None and self.data_targets is None
            ) or overwrite_saved_data:
                self.data_inputs = train_inputs
                self.data_targets = train_targets
            else:
                self.data_inputs = np.vstack((self.data_inputs, train_inputs))
                self.data_targets = np.vstack((self.data_targets, train_targets))
        else:
            raise ValueError(
                "[ERROR]: gp_mpc.learn(): Need to provide both targets and inputs."
            )

        total_input_data = self.data_inputs.shape[0]
        # If validation set is desired.
        if self.test_data_ratio > 0 and self.test_data_ratio is not None:
            train_idx, test_idx = train_test_split(
                list(range(total_input_data)),
                test_size=self.test_data_ratio,
                random_state=self.seed,
            )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {"train_inputs": train_inputs, "train_targets": train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {"test_inputs": test_inputs, "test_targets": test_targets}

        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # Define likelihood.
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        self.gaussian_process = GaussianProcessCollection(
            ZeroMeanIndependentGPModel,
            likelihood,
            len(self.target_mask),
            input_mask=self.input_mask,
            target_mask=self.target_mask,
            kernel=self.kernel,
        )
        if gp_model:
            self.gaussian_process.init_with_hyperparam(
                train_inputs_tensor, train_targets_tensor, gp_model
            )
            print(colored(f"Loaded pretrained model from {gp_model}", "green"))
        else:
            # Train the GP.
            self.gaussian_process.train(
                train_inputs_tensor,
                train_targets_tensor,
                test_inputs_tensor,
                test_targets_tensor,
                n_train=self.optimization_iterations,
                learning_rate=self.learning_rate,
                gpu=self.use_gpu,
                output_dir=self.output_dir,
            )

        self.reset()
        self.prior_ctrl.add_constraints(self.prior_ctrl.additional_constraints)
        self.prior_ctrl.reset()
        # Collect training results.
        training_results = {}
        training_results["train_targets"] = train_targets
        training_results["train_inputs"] = train_inputs
        try:
            training_results["info"] = train_info
        except UnboundLocalError:
            training_results["info"] = None
        return training_results

    def add_constraints(self, constraints):
        """Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        """
        (
            self.constraints,
            self.state_constraints_sym,
            self.input_constraints_sym,
        ) = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self, constraints):
        """Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        """
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, ValueError(
                "This constraint is not in the current list of constraints"
            )
            old_constraints_list.remove(constraint)
        (
            self.constraints,
            self.state_constraints_sym,
            self.input_constraints_sym,
        ) = reset_constraints(old_constraints_list)

    def load(self, model_path):
        """Loads a pretrained batch GP model.

        Args:
            model_path (str): Path to the pretrained model.
        """
        data = np.load(f"{model_path}/data.npz")
        gp_model_path = f"{model_path}"
        self.train_gp(
            input_data=data["data_inputs"],
            target_data=data["data_targets"],
            gp_model=gp_model_path,
        )

    def learn(self, env=None):
        """Performs multiple epochs learning."""

        train_runs = {0: {}}
        test_runs = {0: {}}

        # epoch seed factor
        np.random.seed(self.seed)
        epoch_seeds = (
            np.random.randint(1000, size=self.num_epochs, dtype=int) * self.seed
        )
        epoch_seeds = [int(seed) for seed in epoch_seeds]

        if self.same_train_initial_state:
            train_envs = []
            for epoch in range(self.num_epochs):
                train_envs.append(
                    self.env_func(randomized_init=True, seed=epoch_seeds[epoch])
                )
                train_envs[epoch].action_space.seed(epoch_seeds[epoch])
        else:
            train_env = self.env_func(randomized_init=True, seed=epoch_seeds[0])
            train_env.action_space.seed(epoch_seeds[0])
            train_envs = [train_env] * self.num_epochs

        test_envs = []
        if self.same_test_initial_state:
            for epoch in range(self.num_epochs):
                test_envs.append(
                    self.env_func(randomized_init=True, seed=epoch_seeds[epoch])
                )
                test_envs[epoch].action_space.seed(epoch_seeds[epoch])
        else:
            test_env = self.env_func(randomized_init=True, seed=epoch_seeds[0])
            test_env.action_space.seed(epoch_seeds[0])
            test_envs = [test_env] * self.num_epochs

        for env in train_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]
        for env in test_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]

        # creating train and test experiments
        train_experiments = [
            BaseExperiment(env=env, ctrl=self, reset_when_created=False)
            for env in train_envs[1:]
        ]
        test_experiments = [
            BaseExperiment(env=env, ctrl=self, reset_when_created=False)
            for env in test_envs[1:]
        ]
        # first experiments are for the prior
        train_experiments.insert(
            0,
            BaseExperiment(
                env=train_envs[0], ctrl=self.prior_ctrl, reset_when_created=False
            ),
        )
        test_experiments.insert(
            0,
            BaseExperiment(
                env=test_envs[0], ctrl=self.prior_ctrl, reset_when_created=False
            ),
        )

        for episode in range(self.num_train_episodes_per_epoch):
            # run_results = self.prior_ctrl.run(env=train_envs[0],
            #                                   terminate_run_on_done=self.terminate_train_on_done)
            self.env = train_envs[0]
            run_results = train_experiments[0].run_evaluation(n_episodes=1)
            train_runs[0].update({episode: munch.munchify(run_results)})
            # self.reset()
        for test_ep in range(self.num_test_episodes_per_epoch):
            # run_results = self.run(env=test_envs[0],
            #                        terminate_run_on_done=self.terminate_test_on_done)
            self.env = test_envs[0]
            run_results = test_experiments[0].run_evaluation(n_episodes=1)
            test_runs[0].update({test_ep: munch.munchify(run_results)})
        # self.reset()

        training_results = None
        for epoch in range(1, self.num_epochs):
            # only take data from the last episode from the last epoch
            # if self.rand_data_selection:
            episode_length = train_runs[epoch - 1][
                self.num_train_episodes_per_epoch - 1
            ][0]["obs"][0].shape[0]
            if True:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(
                    train_runs,
                    epoch - 1,
                    self.num_samples,
                    train_envs[epoch - 1].np_random,
                )
            else:
                x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(
                    train_runs, epoch - 1, self.num_samples
                )
            train_inputs, train_targets = self.preprocess_training_data(
                x_seq, actions, x_next_seq
            )  # np.ndarray
            training_results = self.train_gp(
                input_data=train_inputs, target_data=train_targets
            )
            # plot training results
            if self.plot_trained_gp:
                self.gaussian_process.plot_trained_gp(
                    train_inputs,
                    train_targets,
                    output_dir=self.output_dir,
                    title=f"epoch_{epoch}_train",
                    residual_func=self.residual_func,
                )
            max_steps = train_runs[epoch - 1][episode][0]["obs"][0].shape[0]
            x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(
                train_runs, epoch - 1, max_steps
            )
            test_inputs, test_outputs = self.preprocess_training_data(
                x_seq, actions, x_next_seq
            )
            if self.plot_trained_gp:
                self.gaussian_process.plot_trained_gp(
                    test_inputs,
                    test_outputs,
                    output_dir=self.output_dir,
                    title=f"epoch_{epoch}_test",
                    residual_func=self.residual_func,
                )

            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(self.num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode][0]["obs"][0][
                    : self.T + 1, :
                ].T
                self.u_prev = test_runs[epoch - 1][episode][0]["action"][0][
                    : self.T, :
                ].T
                # self.reset()
                # run_results = self.run(env=test_envs[epoch],
                #                        terminate_run_on_done=self.terminate_test_on_done)
                self.env = test_envs[epoch]
                run_results = test_experiments[epoch].run_evaluation(n_episodes=1)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})

            x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(
                test_runs, epoch - 1, episode_length
            )
            train_inputs, train_targets = self.preprocess_training_data(
                x_seq, actions, x_next_seq
            )  # np.ndarray

            # gather training data
            train_runs[epoch] = {}
            for episode in range(self.num_train_episodes_per_epoch):
                self.x_prev = train_runs[epoch - 1][episode][0]["obs"][0][
                    : self.T + 1, :
                ].T
                self.u_prev = train_runs[epoch - 1][episode][0]["action"][0][
                    : self.T, :
                ].T
                self.env = train_envs[epoch]
                run_results = train_experiments[epoch].run_evaluation(n_episodes=1)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            # TODO: fix data logging
            np.savez(
                os.path.join(self.output_dir, "epoch_data"),
                data_inputs=training_results["train_inputs"],
                data_targets=training_results["train_targets"],
                train_runs=train_runs,
                test_runs=test_runs,
                num_epochs=self.num_epochs,
                num_train_episodes_per_epoch=self.num_train_episodes_per_epoch,
                num_test_episodes_per_epoch=self.num_test_episodes_per_epoch,
                num_samples=self.num_samples,
                train_data=self.train_data,
                test_data=self.test_data,
            )

        if training_results:
            np.savez(
                os.path.join(self.output_dir, "data"),
                data_inputs=training_results["train_inputs"],
                data_targets=training_results["train_targets"],
            )

        # close environments
        for experiment in train_experiments:
            experiment.env.close()
        for experiment in test_experiments:
            experiment.env.close()
        # delete c_generated_code folder and acados_ocp_solver.json files
        # os.system(f'rm -rf {self.output_dir}/*c_generated_code*')
        # os.system(f'rm -rf {self.output_dir}/*acados_ocp_solver*')

        self.train_runs = train_runs
        self.test_runs = test_runs

        return train_runs, test_runs

    def gather_training_samples(
        self, all_runs, epoch_i, num_samples, rand_generator=None
    ):
        n_episodes = len(all_runs[epoch_i].keys())
        num_samples_per_episode = int(num_samples / n_episodes)
        x_seq_int = []
        x_next_seq_int = []
        actions_int = []
        for episode_i in range(n_episodes):
            run_results_int = all_runs[epoch_i][episode_i][0]
            n = run_results_int["action"][0].shape[0]
            if num_samples_per_episode < n:
                if rand_generator is not None:
                    rand_inds_int = rand_generator.choice(
                        n - 1, num_samples_per_episode, replace=False
                    )
                else:
                    rand_inds_int = np.arange(num_samples_per_episode)
            else:
                rand_inds_int = np.arange(n - 1)
            next_inds_int = rand_inds_int + 1
            x_seq_int.append(run_results_int["obs"][0][rand_inds_int, :])
            actions_int.append(run_results_int["action"][0][rand_inds_int, :])
            x_next_seq_int.append(run_results_int["obs"][0][next_inds_int, :])
        x_seq_int = np.vstack(x_seq_int)
        actions_int = np.vstack(actions_int)
        x_next_seq_int = np.vstack(x_next_seq_int)

        x_dot_seq_int = (x_next_seq_int - x_seq_int) / self.dt

        return x_seq_int, actions_int, x_next_seq_int, x_dot_seq_int

    @abstractmethod
    def select_action(
        self,
        obs,
        info=None,
    ):
        """Select the action based on the given observation.

        Args:
            obs (ndarray): Current observed state.
            info (dict): Current info.

        Returns:
            action (ndarray): Desired policy action.
        """
        raise NotImplementedError

    def close(self):
        """Clean up."""
        self.env_training.close()
        self.env.close()
        self.prior_ctrl.env.close()

    def setup_results_dict(self):
        """Setup the results dictionary to store run information."""
        self.results_dict = {
            "obs": [],
            "reward": [],
            "done": [],
            "info": [],
            "action": [],
            "horizon_inputs": [],
            "horizon_states": [],
            "goal_states": [],
            "frames": [],
            "state_mse": [],
            "common_cost": [],
            "state": [],
            "state_error": [],
            "inference_time": [],
        }
        self.results_dict["input_constraint_set"] = []
        self.results_dict["state_constraint_set"] = []
        self.results_dict["state_horizon_cov"] = []
        self.results_dict["input_horizon_cov"] = []
        self.results_dict["gp_mean_eq_pred"] = []
        self.results_dict["gp_pred"] = []
        self.results_dict["linear_pred"] = []
        self.results_dict["runtime"] = []
        if self.sparse_gp:
            self.results_dict["inducing_points"] = []

    @abstractmethod
    def reset(self):
        """Reset the controller before running."""
        raise NotImplementedError

    def set_dynamics_func(self):
        """Updates symbolic dynamics with actual control frequency."""
        # linear dynamics for LQR ancillary gain and terminal cost
        dfdxdfdu = self.model.df_func(
            x=np.atleast_2d(self.model.X_EQ)[0, :].T,
            u=np.atleast_2d(self.model.U_EQ)[0, :].T,
        )
        dfdx = dfdxdfdu["dfdx"].toarray()
        dfdu = dfdxdfdu["dfdu"].toarray()
        delta_x = cs.MX.sym("delta_x", self.model.nx, 1)
        delta_u = cs.MX.sym("delta_u", self.model.nu, 1)
        Ad, Bd = discretize_linear_system(dfdx, dfdu, self.dt, exact=True)
        x_dot_lin = Ad @ delta_x + Bd @ delta_u
        self.linear_dynamics_func = cs.Function(
            "linear_discrete_dynamics",
            [delta_x, delta_u],
            [x_dot_lin],
            ["x0", "p"],
            ["xf"],
        )
        self.dfdx = dfdx
        self.dfdu = dfdu
        # # check controlled system is stabilizable
        # A = dfdx
        # B = dfdu
        # n = self.model.nx
        # m = self.model.nu
        # import control
        # ctrb = control.ctrb(A, B)
        # if np.linalg.matrix_rank(ctrb) != n:
        #     raise Exception('System is not stabilizable')
        try:
            (
                self.lqr_gain,
                _,
                _,
                self.P,
            ) = compute_discrete_lqr_gain_from_cont_linear_system(
                dfdx, dfdu, self.Q, self.R, self.dt
            )
        except LinAlgError:
            print(colored("LQR gain computation failed", "red"))
            print(
                colored(
                    "Using the LQR gain and terminal cost in the MPC is disabled",
                    "yellow",
                )
            )
            self.use_lqr_gain_and_terminal_cost = False
        # nonlinear dynamics
        self.dynamics_func = rk_discrete(
            self.model.fc_func, self.model.nx, self.model.nu, self.dt
        )

    @timing
    def compute_initial_guess(self, init_state, goal_states=None):
        """Compute an initial guess of the solution to the optimization problem."""
        if goal_states is None:
            goal_states = self.get_references()
        print(
            colored(
                f"computing initial guess using {self.compute_initial_guess_method}",
                "green",
            )
        )
        if self.compute_initial_guess_method == "ipopt":
            self.setup_optimizer(solver=self.init_solver)
            opti_dict = self.opti_dict
            opti = opti_dict["opti"]
            x_var = opti_dict["x_var"]  # optimization variables
            u_var = opti_dict["u_var"]  # optimization variables
            x_init = opti_dict["x_init"]  # initial state
            x_ref = opti_dict["x_ref"]  # reference state/trajectory

            # Assign the initial state.
            opti.set_value(x_init, init_state)  # initial state should have dim (nx,)
            # Assign reference trajectory within horizon.
            opti.set_value(x_ref, goal_states)
            # Solve the optimization problem.
            try:
                sol = opti.solve()
                x_val, u_val = sol.value(x_var), sol.value(u_var)
            except RuntimeError:
                print(colored("Warm-starting fails", "red"))
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
            x_guess = x_val
            u_guess = u_val
        elif self.compute_initial_guess_method == "lqr":
            # initialize the guess solutions
            x_guess = np.zeros((self.model.nx, self.T + 1))
            u_guess = np.zeros((self.model.nu, self.T))
            x_guess[:, 0] = init_state
            # add the lqr gain and states to the guess
            for i in range(self.T):
                u = (
                    self.lqr_gain @ (x_guess[:, i] - goal_states[:, i])
                    + np.atleast_2d(self.model.U_EQ)[0, :].T
                )
                u_guess[:, i] = u
                x_guess[:, i + 1, None] = self.dynamics_func(x0=x_guess[:, i], p=u)[
                    "xf"
                ].toarray()
        else:
            raise Exception("Initial guess method not implemented.")

        self.x_prev = x_guess
        self.u_prev = u_guess

        # set the solver back
        self.setup_optimizer(solver=self.solver)

        return x_guess, u_guess

    def setup_optimizer(self, solver="qrsqp"):
        """Sets up nonlinear optimization problem."""
        print(colored(f"Setting up optimizer with {solver}", "green"))
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(
                x=x_var[:, i],
                u=u_var[:, i],
                Xr=x_ref[:, i],
                Ur=self.U_EQ,
                Q=self.Q,
                R=self.R,
            )["l"]
        # Terminal cost.
        cost += cost_func(
            x=x_var[:, -1],
            u=np.zeros((nu, 1)),
            Xr=x_ref[:, -1],
            Ur=self.U_EQ,
            Q=self.Q if not self.use_lqr_gain_and_terminal_cost else self.P,
            R=self.R,
        )["l"]
        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])["xf"]
            opti.subject_to(x_var[:, i + 1] == next_state)

            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
                    cost += self.soft_penalty * state_slack[sc_i] ** 2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(
                        state_constraint(x_var[:, i]) < -self.constraint_tol
                    )
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                    cost += self.soft_penalty * input_slack[ic_i] ** 2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(
                        input_constraint(u_var[:, i]) < -self.constraint_tol
                    )

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
                cost += self.soft_penalty * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        # Create solver
        opts = {"expand": True, "error_on_fail": False}
        opti.solver(solver, opts)

        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "x_ref": x_ref,
            "cost": cost,
        }

    @timing
    def get_references(self):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        if self.env.TASK == Task.STABILIZATION:
            # Repeat goal state for horizon steps.
            goal_states = np.tile(self.env.X_GOAL.reshape(-1, 1), (1, self.T + 1))
        elif self.env.TASK == Task.TRAJ_TRACKING:
            # if the task is to track a periodic trajectory (circle, square, figure 8)
            # append the T+1 states of the trajectory to the goal_states
            # such that the vel states won't drop at the end of an episode
            self.extended_ref_traj = deepcopy(self.traj)
            if self.env.TASK_INFO["trajectory_type"] in [
                "circle",
                "square",
                "figure8",
            ] and not (
                "ilqr_ref" in self.env.TASK_INFO.keys()
                and self.env.TASK_INFO["ilqr_ref"]
            ):
                self.extended_ref_traj = np.concatenate(
                    [self.extended_ref_traj, self.extended_ref_traj[:, : self.T + 1]],
                    axis=1,
                )
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.extended_ref_traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.extended_ref_traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
            """
            TODO: if using the extended reference trajectory, 
            variable remain will always be 0. Consider removing it.
            """
            # print('start:', start, 'end:', end, 'remain:', remain)
            goal_states = np.concatenate(
                [
                    self.extended_ref_traj[:, start:end],
                    np.tile(self.extended_ref_traj[:, -1:], (1, remain)),
                ],
                -1,
            )
        else:
            raise Exception("Reference for this mode is not implemented.")
        return goal_states  # (nx, T+1).

    def run(
        self,
        env=None,
        render=False,
        logging=False,
        max_steps=None,
        terminate_run_on_done=None,
    ):
        """Runs evaluation with current policy.

        Args:
            render (bool): if to do real-time rendering.
            logging (bool): if to log on terminal.

        Returns:
            dict: evaluation statisitcs, rendered frames.
        """
        if env is None:
            env = self.env
        if terminate_run_on_done is None:
            terminate_run_on_done = self.terminate_run_on_done

        self.x_prev = None
        self.u_prev = None
        if not env.initial_reset:
            env.set_cost_function_param(self.Q, self.R)
        obs, info = env.reset()
        # obs = env.reset()
        print("Init State:")
        print(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        self.setup_results_dict()
        self.results_dict["obs"].append(obs)
        self.results_dict["state"].append(env.state)
        i = 0
        if env.TASK == Task.STABILIZATION:
            if max_steps is None:
                MAX_STEPS = int(env.CTRL_FREQ * env.EPISODE_LEN_SEC)
            else:
                MAX_STEPS = max_steps
        elif env.TASK == Task.TRAJ_TRACKING:
            if max_steps is None:
                MAX_STEPS = self.traj.shape[1]
            else:
                MAX_STEPS = max_steps
        else:
            raise Exception("Undefined Task")
        self.terminate_loop = False
        done = False
        common_metric = 0
        while (
            not (done and terminate_run_on_done)
            and i < MAX_STEPS
            and not (self.terminate_loop)
        ):
            action = self.select_action(obs)
            if self.terminate_loop:
                print("Infeasible MPC Problem")
                break
            # Repeat input for more efficient control.
            obs, reward, done, info = env.step(action)
            self.results_dict["obs"].append(obs)
            self.results_dict["reward"].append(reward)
            self.results_dict["done"].append(done)
            self.results_dict["info"].append(info)
            self.results_dict["action"].append(action)
            self.results_dict["state"].append(env.state)
            self.results_dict["state_mse"].append(info["mse"])
            self.results_dict["state_error"].append(env.state - env.X_GOAL[i, :])
            common_metric += info["mse"]
            print(i, "-th step.")
            print("action:", action)
            print("obs", obs)
            print("reward", reward)
            print("done", done)
            print(info)
            print()
            if render:
                env.render()
                frames.append(env.render("rgb_array"))
            i += 1
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        if logging:
            msg = "****** Evaluation ******\n"
            msg += "eval_ep_length {:.2f} +/- {:.2f} | eval_ep_return {:.3f} +/- {:.3f}\n".format(
                ep_lengths.mean(), ep_lengths.std(), ep_returns.mean(), ep_returns.std()
            )
        if len(frames) != 0:
            self.results_dict["frames"] = frames
        self.results_dict["obs"] = np.vstack(self.results_dict["obs"])
        self.results_dict["state"] = np.vstack(self.results_dict["state"])
        try:
            self.results_dict["reward"] = np.vstack(self.results_dict["reward"])
            self.results_dict["action"] = np.vstack(self.results_dict["action"])
            self.results_dict["full_traj_common_cost"] = common_metric
            self.results_dict["total_rmse_state_error"] = compute_state_rmse(
                self.results_dict["state"]
            )
            self.results_dict["total_rmse_obs_error"] = compute_state_rmse(
                self.results_dict["obs"]
            )
        except ValueError:
            raise Exception(
                "[ERROR] mpc.run().py: MPC could not find a solution for the first step given the initial conditions. "
                "Check to make sure initial conditions are feasible."
            )
        return deepcopy(self.results_dict)

    def reset_before_run(self, obs=None, info=None, env=None):
        """Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        """
        self.setup_results_dict()

    def extract_step(self, info=None):
        """Extracts the current step from the info.

        Args:
            info (dict): The info list returned from the environment.

        Returns:
            step (int): The current step/iteration of the environment.
        """

        if info is not None:
            step = info["current_step"]
        else:
            step = 0

        return step
