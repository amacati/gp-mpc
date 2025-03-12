import os
import time
from copy import deepcopy
from functools import partial

import casadi as cs
import gpytorch
import munch
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split

from safe_control_gym.core.constraints import (
    GENERAL_CONSTRAINTS,
    create_constraint_list,
)
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.mpc.gp_utils import (
    GaussianProcess,
    ZeroMeanIndependentGPModel,
    covSE_single,
    kmeans_centroids,
)
from safe_control_gym.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.mpc.mpc_utils import (
    discretize_linear_system,
    get_cost_weight_matrix,
    reset_constraints,
)


class GPMPC_ACADOS_TRP:
    """Implements a GP-MPC controller with Acados optimization."""

    def __init__(
        self,
        env_func,
        num_samples: int,
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
        n_ind_points: int = 30,
        inducing_point_selection_method="kmeans",
        recalc_inducing_points_at_every_step=False,
        prob: float = 0.955,
        initial_rollout_std: float = 0.005,
        input_mask: list = None,
        target_mask: list = None,
        gp_approx: str = "mean_eq",
        online_learning: bool = False,
        prior_info: dict = None,
        sparse_gp: bool = False,
        prior_param_coeff: float = 1.0,
        terminate_run_on_done: bool = True,
        output_dir: str = "results/temp",
        use_RTI: bool = False,
        train_env_rand_info: dict = None,
    ):
        self.q_mpc = q_mpc
        self.r_mpc = r_mpc
        self.num_samples = num_samples

        if prior_info is None or prior_info == {}:
            raise ValueError(
                "GPMPC requires prior_prop to be defined. You may use the real mass properties and then use prior_param_coeff to modify them accordingly."
            )
        prior_info["prior_prop"].update(
            (prop, val * prior_param_coeff) for prop, val in prior_info["prior_prop"].items()
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
        self.setup_results_dict()

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
            ) = reset_constraints(self.env.constraints.constraints + self.additional_constraints)
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
        self.kernel = "Matern"
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
        self.plot_trained_gp = False

        # MPC params
        self.gp_soft_constraints = self.soft_constraints_params["gp_soft_constraints"]
        self.gp_soft_constraints_coeff = self.soft_constraints_params["gp_soft_constraints_coeff"]

        self.last_obs = None
        self.last_action = None
        ################
        self.uncertain_dim = [1, 3, 5, 7, 9]
        self.Bd = np.eye(self.model.nx)[:, self.uncertain_dim]
        self.input_mask = None
        self.target_mask = None
        self.train_env_rand_info = train_env_rand_info
        self.rand_hist = {"task_rand": [], "domain_rand": []}

        # MPC params
        self.use_RTI = use_RTI

        self.prior_ctrl = MPC_ACADOS(
            env_func=self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            soft_constraints=self.soft_constraints_params["prior_soft_constraints"],
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            use_gpu=use_gpu,
            seed=seed,
            prior_info=prior_info,
        )
        self.prior_ctrl.reset()
        self.prior_dynamics_func = self.prior_ctrl.dynamics_func
        self.prior_dynamics_func_c = self.prior_ctrl.model.fc_func

        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        self.phi_idx = 6
        self.theta_idx = 7
        self.phi_dot_idx = 8
        self.theta_dot_idx = 9
        self.T_cmd_idx = 0
        self.phi_cmd_idx = 1
        self.theta_cmd_idx = 2

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
        g = 9.81
        dt = 1 / 60
        T_cmd = u_seq[:, 0]
        T_prior_data = self.prior_ctrl.env.T_mapping_func(T_cmd).full().flatten()
        # numerical differentiation
        x_dot_seq = [(x_next_seq[i, :] - x_seq[i, :]) / dt for i in range(x_seq.shape[0])]
        x_dot_seq = np.array(x_dot_seq)
        T_true_data = np.sqrt(
            (x_dot_seq[:, 5] + g) ** 2 + (x_dot_seq[:, 1] ** 2) + (x_dot_seq[:, 3] ** 2)
        )
        targets_T = (T_true_data - T_prior_data).reshape(-1, 1)
        input_T = u_seq[:, 0].reshape(-1, 1)

        theta_true = x_dot_seq[:, self.theta_idx]
        theta_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)["f"].toarray()[
            self.theta_idx, :
        ]
        targets_theta = (theta_true - theta_prior).reshape(-1, 1)
        input_theta = np.concatenate(
            [
                x_seq[:, self.theta_idx].reshape(-1, 1),
                x_seq[:, self.theta_dot_idx].reshape(-1, 1),
                u_seq[:, self.theta_cmd_idx].reshape(-1, 1),
            ],
            axis=1,
        )

        phi_true = x_dot_seq[:, self.phi_idx]
        phi_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)["f"].toarray()[self.phi_idx, :]
        targets_phi = (phi_true - phi_prior).reshape(-1, 1)
        input_phi = np.concatenate(
            [
                x_seq[:, self.phi_idx].reshape(-1, 1),
                x_seq[:, self.phi_dot_idx].reshape(-1, 1),
                u_seq[:, self.phi_cmd_idx].reshape(-1, 1),
            ],
            axis=1,
        )

        train_input = np.concatenate([input_T, input_phi, input_theta], axis=1)
        train_output = np.concatenate([targets_T, targets_phi, targets_theta], axis=1)

        return train_input, train_output

    def train_gp(
        self,
        input_data,
        target_data,
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
        self.reset()
        train_inputs = input_data
        train_targets = target_data
        if (self.data_inputs is None and self.data_targets is None) or overwrite_saved_data:
            self.data_inputs = train_inputs
            self.data_targets = train_targets
        else:
            self.data_inputs = np.vstack((self.data_inputs, train_inputs))
            self.data_targets = np.vstack((self.data_targets, train_targets))

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

        # seperate the data for T R P
        train_input_T = train_inputs_tensor[:, 0].reshape(-1)
        train_target_T = train_targets_tensor[:, 0].reshape(-1)
        test_inputs_T = test_inputs_tensor[:, 0].reshape(-1)
        test_targets_T = test_targets_tensor[:, 0].reshape(-1)

        R_data_idx = [1, 2, 3]
        train_input_R = train_inputs_tensor[:, R_data_idx].reshape(-1, 3)
        test_inputs_R = test_inputs_tensor[:, R_data_idx].reshape(-1, 3)
        train_target_R = train_targets_tensor[:, 1].reshape(-1)
        test_targets_R = test_targets_tensor[:, 1].reshape(-1)

        P_data_idx = [4, 5, 6]
        train_input_P = train_inputs_tensor[:, P_data_idx].reshape(-1, 3)
        test_inputs_P = test_inputs_tensor[:, P_data_idx].reshape(-1, 3)
        train_target_P = train_targets_tensor[:, 2].reshape(-1)
        test_targets_P = test_targets_tensor[:, 2].reshape(-1)

        # Define likelihood.
        likelihood_T = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_R = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_P = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()

        GP_T = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_T,
            kernel="RBF_single",
        )
        GP_R = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_R,
            kernel="RBF_single",
        )

        GP_P = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_P,
            kernel="RBF_single",
        )

        if gp_model:
            GP_T.init_with_hyperparam(train_input_T, train_target_T, gp_model[0])
            GP_R.init_with_hyperparam(train_input_R, train_target_R, gp_model[1])
            GP_P.init_with_hyperparam(train_input_P, train_target_P, gp_model[2])
        else:
            GP_T.train(
                train_input_T,
                train_target_T,
                test_inputs_T,
                test_targets_T,
                n_train=self.optimization_iterations[0],
                learning_rate=self.learning_rate[0],
                gpu=self.use_gpu,
                fname=self.output_dir / "best_model_T.pth",
            )
            GP_R.train(
                train_input_R,
                train_target_R,
                test_inputs_R,
                test_targets_R,
                n_train=self.optimization_iterations[1],
                learning_rate=self.learning_rate[1],
                gpu=self.use_gpu,
                fname=self.output_dir / "best_model_R.pth",
            )
            GP_P.train(
                train_input_P,
                train_target_P,
                test_inputs_P,
                test_targets_P,
                n_train=self.optimization_iterations[2],
                learning_rate=self.learning_rate[2],
                gpu=self.use_gpu,
                fname=self.output_dir / "best_model_P.pth",
            )

        self.gaussian_process = [GP_T, GP_R, GP_P]
        self.reset()

        # Collect training results.
        training_results = {}
        training_results["train_targets"] = train_targets
        training_results["train_inputs"] = train_inputs
        return training_results

    def setup_acados_model(self, n_ind_points) -> AcadosModel:
        # setup GP related
        self.inverse_cdf = scipy.stats.norm.ppf(
            1 - (1 / self.model.nx - (self.prob + 1) / (2 * self.model.nx))
        )

        # setup acados model
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = self.env.NAME

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point

        T_pred_point = z[self.T_cmd_idx + self.model.nx]
        R_pred_point = z[[self.phi_idx, self.phi_dot_idx, self.phi_cmd_idx + self.model.nx]]
        P_pred_point = z[[self.theta_idx, self.theta_dot_idx, self.theta_cmd_idx + self.model.nx]]
        if self.sparse_gp:
            self.create_sparse_GP_machinery(n_ind_points)
            # sparse GP inducing points
            """
            z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, len(self.input_mask))
            mean_post_factor should be of shape (len(self.target_mask), n_ind_points)
            Here we create the corresponding parameters since acados supports only 1D parameters
            """
            z_ind = cs.MX.sym("z_ind", n_ind_points, 7)
            mean_post_factor = cs.MX.sym("mean_post_factor", 3, n_ind_points)
            acados_model.p = cs.vertcat(
                cs.reshape(z_ind, -1, 1), cs.reshape(mean_post_factor, -1, 1)
            )
            # define the dynamics
            T_pred = cs.sum2(
                self.K_z_zind_func_T(z1=T_pred_point, z2=z_ind)["K"] * mean_post_factor[0, :]
            )
            R_pred = cs.sum2(
                self.K_z_zind_func_R(z1=R_pred_point, z2=z_ind)["K"] * mean_post_factor[1, :]
            )
            P_pred = cs.sum2(
                self.K_z_zind_func_P(z1=P_pred_point, z2=z_ind)["K"] * mean_post_factor[2, :]
            )
        else:
            GP_T = self.gaussian_process[0]
            GP_R = self.gaussian_process[1]
            GP_P = self.gaussian_process[2]
            T_pred = GP_T.casadi_predict(z=T_pred_point)["mean"]
            R_pred = GP_R.casadi_predict(z=R_pred_point)["mean"]
            P_pred = GP_P.casadi_predict(z=P_pred_point)["mean"]

        f_cont = self.prior_dynamics_func_c(x=acados_model.x, u=acados_model.u)["f"] + cs.vertcat(
            0,
            T_pred
            * (cs.cos(acados_model.x[self.phi_idx]) * cs.sin(acados_model.x[self.theta_idx])),
            0,
            T_pred * (-cs.sin(acados_model.x[self.phi_idx])),
            0,
            T_pred
            * (cs.cos(acados_model.x[self.phi_idx]) * cs.cos(acados_model.x[self.theta_idx])),
            0,
            R_pred,
            0,
            P_pred,
        )
        f_cont_func = cs.Function(
            "f_cont_func", [acados_model.x, acados_model.u, acados_model.p], [f_cont]
        )

        # use rk4 to discretize the continuous dynamics
        k1 = f_cont_func(acados_model.x, acados_model.u, acados_model.p)
        k2 = f_cont_func(acados_model.x + self.dt / 2 * k1, acados_model.u, acados_model.p)
        k3 = f_cont_func(acados_model.x + self.dt / 2 * k2, acados_model.u, acados_model.p)
        k4 = f_cont_func(acados_model.x + self.dt * k3, acados_model.u, acados_model.p)
        f_disc = acados_model.x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = "time"

        self.acados_model = acados_model

        # only for plotting
        T_cmd = cs.MX.sym("T_cmd")
        theta_cmd = cs.MX.sym("theta_cmd")
        theta = cs.MX.sym("theta")
        theta_dot = cs.MX.sym("theta_dot")
        T_true_func = self.env.T_mapping_func
        T_prior_func = self.prior_ctrl.env.T_mapping_func
        T_res = T_true_func(T_cmd) - T_prior_func(T_cmd)
        self.T_res_func = cs.Function("T_res_func", [T_cmd], [T_res])
        R_true_func = self.env.R_mapping_func
        R_prior_func = self.prior_ctrl.env.R_mapping_func
        R_res = R_true_func(theta, theta_dot, theta_cmd) - R_prior_func(theta, theta_dot, theta_cmd)
        self.R_res_func = cs.Function("R_res_func", [theta, theta_dot, theta_cmd], [R_res])
        P_true_func = self.env.P_mapping_func
        P_prior_func = self.prior_ctrl.env.P_mapping_func
        P_res = P_true_func(theta, theta_dot, theta_cmd) - P_prior_func(theta, theta_dot, theta_cmd)
        self.P_res_func = cs.Function("P_res_func", [theta, theta_dot, theta_cmd], [P_res])

    def setup_acados_optimizer(self, n_ind_points):
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        # cost weight matrices
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.P if hasattr(self, "P") else self.Q

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx : (nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # Constraints
        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        state_tighten_list = []
        input_tighten_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
            # chance state constraint tightening
            state_tighten_list.append(
                cs.MX.sym(f"state_tighten_{sc_i}", state_constraint(ocp.model.x).shape[0], 1)
            )
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))
            # chance input constraint tightening
            input_tighten_list.append(
                cs.MX.sym(f"input_tighten_{ic_i}", input_constraint(ocp.model.u).shape[0], 1)
            )

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(
            *state_constraint_expr_list
        )  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(
            ocp, h0_expr, h_expr, he_expr, state_tighten_list, input_tighten_list
        )
        # pass the tightening variables to the ocp object as parameters
        tighten_param = cs.vertcat(*state_tighten_list, *input_tighten_list)
        if self.sparse_gp:
            ocp.model.p = cs.vertcat(ocp.model.p, tighten_param)
            ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))  # dummy values
        else:
            ocp.model.p = tighten_param
            ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))  # dummy values

        # slack costs for nonlinear constraints
        if self.gp_soft_constraints:
            # slack variables for all constraints
            ocp.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            ocp.constraints.Jsh = np.eye(h_expr.shape[0])
            ocp.constraints.Jsh_e = np.eye(he_expr.shape[0])
            # slack penalty
            L2_pen = self.gp_soft_constraints_coeff
            L1_pen = self.gp_soft_constraints_coeff
            ocp.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            ocp.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            ocp.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.zu = L1_pen * np.ones(h_expr.shape[0])
            ocp.cost.Zl_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.Zu_e = L2_pen * np.ones(he_expr.shape[0])
            ocp.cost.zl_e = L1_pen * np.ones(he_expr.shape[0])
            ocp.cost.zu_e = L1_pen * np.ones(he_expr.shape[0])

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"

        ocp.solver_options.nlp_solver_type = "SQP" if not self.use_RTI else "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 25 if not self.use_RTI else 1
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        # c code generation
        ocp.code_export_directory = str(self.output_dir / "gpmpc_c_generated_code")

        self.ocp = ocp
        self.opti_dict = {"n_ind_points": n_ind_points}
        # compute sparse GP values
        # the actual values will be set in select_action_with_gp
        (
            self.mean_post_factor_val_all,
            self.z_ind_val_all,
        ) = self.precompute_mean_post_factor_all_data()
        if self.sparse_gp:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val
        else:
            (
                mean_post_factor_val,
                z_ind_val,
            ) = self.precompute_mean_post_factor_all_data()
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val

    def processing_acados_constraints_expression(
        self,
        ocp: AcadosOcp,
        h0_expr,
        h_expr,
        he_expr,
        state_tighten_list,
        input_tighten_list,
    ) -> AcadosOcp:
        """Preprocess the constraints to be compatible with acados.
            Args:
                h0_expr (casadi expression): initial state constraints
                h_expr (casadi expression): state and input constraints
                he_expr (casadi expression): terminal state constraints
                state_tighten_list (list): list of casadi SX variables for state constraint tightening
                input_tighten_list (list): list of casadi SX variables for input constraint tightening
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set

        Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = idxbu # active constraints dimension
        """
        # NOTE: only the upper bound is tightened due to constraint are defined in the
        # form of g(x, u) <= constraint_tol in safe-control-gym

        # lambda functions to set the upper and lower bounds of the chance constraints
        def constraint_ub_chance(constraint):
            return -self.constraint_tol * np.ones(constraint.shape)

        def constraint_lb_chance(constraint):
            return -1e8 * np.ones(constraint.shape)

        state_tighten_var = cs.vertcat(*state_tighten_list)
        input_tighten_var = cs.vertcat(*input_tighten_list)

        ub = {
            "h": constraint_ub_chance(h_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
            "h0": constraint_ub_chance(h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
            "he": constraint_ub_chance(he_expr - state_tighten_var),
        }
        lb = {
            "h": constraint_lb_chance(h_expr),
            "h0": constraint_lb_chance(h0_expr),
            "he": constraint_lb_chance(he_expr),
        }

        # make sure all the ub and lb are 1D casaadi SX variables
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
        # check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f"ub[{key}] is not 1D numpy array"
            assert lb[key].ndim == 1, f"lb[{key}] is not 1D numpy array"
        assert ub["h"].shape == lb["h"].shape, "h_ub and h_lb have different shapes"

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr = h_expr - cs.vertcat(state_tighten_var, input_tighten_var)
        ocp.model.con_h_expr_e = he_expr - state_tighten_var
        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = (
            h0_expr.shape[0],
            h_expr.shape[0],
            he_expr.shape[0],
        )
        # assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub["h0"]
        ocp.constraints.lh_0 = lb["h0"]
        ocp.constraints.uh = ub["h"]
        ocp.constraints.lh = lb["h"]
        ocp.constraints.uh_e = ub["he"]
        ocp.constraints.lh_e = lb["he"]

        return ocp

    def close(self):
        """Clean up."""
        self.env_training.close()
        self.env.close()
        self.prior_ctrl.env.close()

    def select_action(self, obs, info=None):
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.results_dict["runtime"].append(time_after - time_before)
        self.last_obs = obs
        self.last_action = action

        return action

    def select_action_with_gp(self, obs):
        nx, nu = self.model.nx, self.model.nu

        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)
        # set initial guess for the solution
        if self.warmstart and self.x_prev is None and self.u_prev is None:
            self.acados_ocp_solver.solve_for_x0(obs)
        else:
            for idx in range(self.T + 1):
                self.acados_ocp_solver.set(idx, "x", obs)
            for idx in range(self.T):
                self.acados_ocp_solver.set(idx, "u", np.zeros((nu,)))

        # compute the sparse GP values
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict["inducing_points"].append(z_ind_val)
        else:
            # use the precomputed values
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict["inducing_points"] = [z_ind_val]

        # Set the probabilistic state and input constraint set limits.
        # Tightening at the first step is possible if self.compute_initial_guess is used
        (
            state_constraint_set_prev,
            input_constraint_set_prev,
        ) = self.precompute_probabilistic_limits()
        # set acados parameters
        if self.sparse_gp:
            # sparse GP parameters
            n_ind_points = self.opti_dict["n_ind_points"]
            assert z_ind_val.shape == (n_ind_points, 7)
            assert mean_post_factor_val.shape == (3, n_ind_points)
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order="F")
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order="F")
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)
            # tighten constraints
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                # set the parameter values
                parameter_values = np.concatenate((dyn_value, tighten_value))
                # self.acados_ocp_solver.set(idx, "p", dyn_value)
                # check the shapes
                assert (
                    self.ocp.model.p.shape[0] == parameter_values.shape[0]
                ), f"parameter_values.shape: {parameter_values.shape}; model.p.shape: {self.ocp.model.p.shape}"
                self.acados_ocp_solver.set(idx, "p", parameter_values)
            # tighten terminal state constraints
            tighten_value = np.concatenate(
                (state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,)))
            )
            # set the parameter values
            parameter_values = np.concatenate((dyn_value, tighten_value))
            self.acados_ocp_solver.set(self.T, "p", parameter_values)
        else:
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                self.acados_ocp_solver.set(idx, "p", tighten_value)
            # tighten terminal state constraints
            tighten_value = np.concatenate(
                (state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,)))
            )
            self.acados_ocp_solver.set(self.T, "p", tighten_value)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == "tracking":
            self.traj_step += 1
        y_ref = np.concatenate(
            (goal_states[:, :-1], np.repeat(self.U_EQ.reshape(-1, 1), self.T, axis=1)),
            axis=0,
        )
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, "yref", y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)

        # solve the optimization problem
        if self.use_RTI:
            # preparation phase
            self.acados_ocp_solver.options_set("rti_phase", 1)
            status = self.acados_ocp_solver.solve()

            # feedback phase
            self.acados_ocp_solver.options_set("rti_phase", 2)
            status = self.acados_ocp_solver.solve()
        else:
            status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            self.acados_ocp_solver.print_statistics()
            raise RuntimeError(f"acados returned status {status}. ")

        action = self.acados_ocp_solver.get(0, "u")
        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))

        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, "x")
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, "u")
        if nu == 1:
            self.u_prev = self.u_prev.flatten()
        self.x_guess = self.x_prev
        self.u_guess = self.u_prev

        self.results_dict["inference_time"].append(self.acados_ocp_solver.get_stats("time_tot"))

        if hasattr(self, "K"):
            action += self.K @ (self.x_prev[:, 0] - obs)

        return action

    def precompute_mean_post_factor_all_data(self):
        """If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        """
        dim_gp_outputs = len(self.gaussian_process)
        n_training_samples = self.train_data["train_targets"].shape[0]
        inputs = self.train_data["train_inputs"]
        targets = self.train_data["train_targets"]
        mean_post_factor = np.zeros((dim_gp_outputs, n_training_samples))
        for i in range(dim_gp_outputs):
            K_z_z = self.gaussian_process[i].model.K_plus_noise_inv
            mean_post_factor[i] = K_z_z.detach().numpy() @ targets[:, i]

        return mean_post_factor, inputs

    def precompute_sparse_gp_values(self, n_ind_points):
        """Uses the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points (int): Number of inducing points.
        """
        n_data_points = self.train_data["train_targets"].shape[0]
        dim_gp_outputs = len(self.gaussian_process)
        inputs = self.train_data["train_inputs"]
        targets = self.train_data["train_targets"]
        # If there is no previous solution. Choose T random training set points.
        if self.inducing_point_selection_method == "kmeans":
            centroids = kmeans_centroids(n_ind_points, inputs, rand_state=self.seed)
            contiguous_masked_inputs = np.ascontiguousarray(
                inputs
            )  # required for version sklearn later than 1.0.2
            inds, _ = pairwise_distances_argmin_min(centroids, contiguous_masked_inputs)
            z_ind = inputs[inds]
        elif self.inducing_point_selection_method == "random":
            inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
            z_ind = inputs[inds]
        else:
            raise ValueError(
                "[Error]: gp_mpc.precompute_sparse_gp_values: Only 'kmeans' or 'random' allowed."
            )

        use_pinv = True

        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]
        T_data_idx = [0]
        R_data_idx = [1, 2, 3]
        P_data_idx = [4, 5, 6]
        K_zind_zind_T = GP_T.model.covar_module(torch.from_numpy(z_ind[:, T_data_idx]).double())
        K_zind_zind_R = GP_R.model.covar_module(torch.from_numpy(z_ind[:, R_data_idx]).double())
        K_zind_zind_P = GP_P.model.covar_module(torch.from_numpy(z_ind[:, P_data_idx]).double())
        if use_pinv:
            K_zind_zind_inv_T = torch.pinverse(K_zind_zind_T.evaluate().detach())
            K_zind_zind_inv_R = torch.pinverse(K_zind_zind_R.evaluate().detach())
            K_zind_zind_inv_P = torch.pinverse(K_zind_zind_P.evaluate().detach())
        else:
            K_zind_zind_inv_T = K_zind_zind_T.inv_matmul(torch.eye(n_ind_points).double()).detach()
            K_zind_zind_inv_R = K_zind_zind_R.inv_matmul(torch.eye(n_ind_points).double()).detach()
            K_zind_zind_inv_P = K_zind_zind_P.inv_matmul(torch.eye(n_ind_points).double()).detach()
        K_zind_zind_T = (
            GP_T.model.covar_module(torch.from_numpy(z_ind[:, T_data_idx]).double())
            .evaluate()
            .detach()
        )
        K_zind_zind_R = (
            GP_R.model.covar_module(torch.from_numpy(z_ind[:, R_data_idx]).double())
            .evaluate()
            .detach()
        )
        K_zind_zind_P = (
            GP_P.model.covar_module(torch.from_numpy(z_ind[:, P_data_idx]).double())
            .evaluate()
            .detach()
        )

        K_zind_zind_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        K_zind_zind_inv[0] = K_zind_zind_inv_T
        K_zind_zind_inv[1] = K_zind_zind_inv_R
        K_zind_zind_inv[2] = K_zind_zind_inv_P
        K_x_zind_T = (
            GP_T.model.covar_module(
                torch.from_numpy(inputs[:, T_data_idx]).double(),
                torch.from_numpy(z_ind[:, 0]).double(),
            )
            .evaluate()
            .detach()
        )
        K_x_zind_R = (
            GP_R.model.covar_module(
                torch.from_numpy(inputs[:, R_data_idx]).double(),
                torch.from_numpy(z_ind[:, R_data_idx]).double(),
            )
            .evaluate()
            .detach()
        )
        K_x_zind_P = (
            GP_P.model.covar_module(
                torch.from_numpy(inputs[:, P_data_idx]).double(),
                torch.from_numpy(z_ind[:, P_data_idx]).double(),
            )
            .evaluate()
            .detach()
        )

        K_plus_noise_T = GP_T.model.K_plus_noise.detach()
        K_plus_noise_R = GP_R.model.K_plus_noise.detach()
        K_plus_noise_P = GP_P.model.K_plus_noise.detach()
        K_plus_noise = torch.zeros((dim_gp_outputs, n_data_points, n_data_points))
        K_plus_noise[0] = K_plus_noise_T
        K_plus_noise[1] = K_plus_noise_R
        K_plus_noise[2] = K_plus_noise_P

        if use_pinv:
            Q_X_X_T = K_x_zind_T @ K_zind_zind_inv_T @ K_x_zind_T.T
            Q_X_X_R = K_x_zind_R @ K_zind_zind_inv_R @ K_x_zind_R.T
            Q_X_X_P = K_x_zind_P @ K_zind_zind_inv_P @ K_x_zind_P.T
        else:
            Q_X_X_T = K_x_zind_T @ torch.linalg.solve(K_zind_zind_T, K_x_zind_T.T)
            Q_X_X_R = K_x_zind_R @ torch.linalg.solve(K_zind_zind_R, K_x_zind_R.T)
            Q_X_X_P = K_x_zind_P @ torch.linalg.solve(K_zind_zind_P, K_x_zind_P.T)

        Gamma_T = torch.diagonal(K_plus_noise_T - Q_X_X_T)
        Gamma_inv_T = torch.diag_embed(1 / Gamma_T)
        Gamma_R = torch.diagonal(K_plus_noise_R - Q_X_X_R)
        Gamma_inv_R = torch.diag_embed(1 / Gamma_R)
        Gamma_P = torch.diagonal(K_plus_noise_P - Q_X_X_P)
        Gamma_inv_P = torch.diag_embed(1 / Gamma_P)

        Sigma_inv_T = K_zind_zind_T + K_x_zind_T.T @ Gamma_inv_T @ K_x_zind_T
        Sigma_inv_R = K_zind_zind_R + K_x_zind_R.T @ Gamma_inv_R @ K_x_zind_R
        Sigma_inv_P = K_zind_zind_P + K_x_zind_P.T @ Gamma_inv_P @ K_x_zind_P
        Sigma_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        Sigma_inv[0] = Sigma_inv_T
        Sigma_inv[1] = Sigma_inv_R
        Sigma_inv[2] = Sigma_inv_P

        if use_pinv:
            Sigma_T = torch.pinverse(Sigma_inv_T)
            Sigma_R = torch.pinverse(Sigma_inv_R)
            Sigma_P = torch.pinverse(Sigma_inv_P)
            mean_post_factor_T = (
                Sigma_T @ K_x_zind_T.T @ Gamma_inv_T @ torch.from_numpy(targets[:, 0]).double()
            )
            mean_post_factor_R = (
                Sigma_R @ K_x_zind_R.T @ Gamma_inv_R @ torch.from_numpy(targets[:, 1]).double()
            )
            mean_post_factor_P = (
                Sigma_P @ K_x_zind_P.T @ Gamma_inv_P @ torch.from_numpy(targets[:, 2]).double()
            )
        else:
            mean_post_factor_T = torch.linalg.solve(
                Sigma_inv_T,
                K_x_zind_T.T @ Gamma_inv_T @ torch.from_numpy(targets[:, 0]).double(),
            )
            mean_post_factor_R = torch.linalg.solve(
                Sigma_inv_R,
                K_x_zind_R.T @ Gamma_inv_R @ torch.from_numpy(targets[:, 1]).double(),
            )
            mean_post_factor_P = torch.linalg.solve(
                Sigma_inv_P,
                K_x_zind_P.T @ Gamma_inv_P @ torch.from_numpy(targets[:, 2]).double(),
            )

        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points))
        mean_post_factor[0] = mean_post_factor_T
        mean_post_factor[1] = mean_post_factor_R
        mean_post_factor[2] = mean_post_factor_P

        return (
            mean_post_factor.detach().numpy(),
            Sigma_inv.detach().numpy(),
            K_zind_zind_inv.detach().numpy(),
            z_ind,
        )

    def create_sparse_GP_machinery(self, n_ind_points):
        """This setups the gaussian process approximations for FITC formulation."""

        R_data_idx = [1, 2, 3]
        P_data_idx = [4, 5, 6]
        # lengthscales, signal_var, noise_var, gp_K_plus_noise = self.gaussian_process.get_hyperparameters(as_numpy=True)
        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]

        lengthscales_T = GP_T.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_R = GP_R.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_P = GP_P.model.covar_module.base_kernel.lengthscale.detach().numpy()
        signal_var_T = GP_T.model.covar_module.outputscale.detach().numpy()
        signal_var_R = GP_R.model.covar_module.outputscale.detach().numpy()
        signal_var_P = GP_P.model.covar_module.outputscale.detach().numpy()
        noise_var_T = GP_T.likelihood.noise.detach().numpy()
        noise_var_R = GP_R.likelihood.noise.detach().numpy()
        noise_var_P = GP_P.likelihood.noise.detach().numpy()
        gp_K_plus_noise_T = GP_T.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_R = GP_R.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_P = GP_P.model.K_plus_noise.detach().numpy()

        # stacking
        lengthscales = np.vstack((lengthscales_T, lengthscales_R, lengthscales_P))
        signal_var = np.array([signal_var_T, signal_var_R, signal_var_P])
        noise_var = np.array([noise_var_T, noise_var_R, noise_var_P])
        gp_K_plus_noise = np.zeros((3, gp_K_plus_noise_T.shape[0], gp_K_plus_noise_T.shape[1]))
        gp_K_plus_noise[0] = gp_K_plus_noise_T
        gp_K_plus_noise[1] = gp_K_plus_noise_R
        gp_K_plus_noise[2] = gp_K_plus_noise_P

        self.length_scales = lengthscales.squeeze()
        self.signal_var = signal_var.squeeze()
        self.noise_var = noise_var.squeeze()
        self.gp_K_plus_noise = gp_K_plus_noise
        Nx = self.train_data["train_inputs"].shape[1]
        Ny = self.train_data["train_targets"].shape[1]
        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind, length scales and signal variance.
        # We need the CasADI version of this so that it can by symbolically differentiated in in the MPC optimization.
        z1_T = cs.SX.sym("z1", 1)
        z2_T = cs.SX.sym("z2", 1)
        ell_s_T = cs.SX.sym("ell", 1)
        sf2_s_T = cs.SX.sym("sf2")
        z1_R = cs.SX.sym("z1", 3)
        z2_R = cs.SX.sym("z2", 3)
        ell_s_R = cs.SX.sym("ell", 1)
        sf2_s_R = cs.SX.sym("sf2")
        z1_P = cs.SX.sym("z1", 3)
        z2_P = cs.SX.sym("z2", 3)
        ell_s_P = cs.SX.sym("ell", 1)
        sf2_s_P = cs.SX.sym("sf2")
        z_ind = cs.SX.sym("z_ind", n_ind_points, Nx)
        ks_T = cs.SX.zeros(1, n_ind_points)  # kernel vector
        ks_R = cs.SX.zeros(1, n_ind_points)  # kernel vector
        ks_P = cs.SX.zeros(1, n_ind_points)  # kernel vector

        covSE_T = cs.Function(
            "covSE",
            [z1_T, z2_T, ell_s_T, sf2_s_T],
            [covSE_single(z1_T, z2_T, ell_s_T, sf2_s_T)],
        )
        covSE_R = cs.Function(
            "covSE",
            [z1_R, z2_R, ell_s_R, sf2_s_R],
            [covSE_single(z1_R, z2_R, ell_s_R, sf2_s_R)],
        )
        covSE_P = cs.Function(
            "covSE",
            [z1_P, z2_P, ell_s_P, sf2_s_P],
            [covSE_single(z1_P, z2_P, ell_s_P, sf2_s_P)],
        )
        for i in range(n_ind_points):
            ks_T[i] = covSE_T(z1_T, z_ind[i, 0], ell_s_T, sf2_s_T)
            ks_R[i] = covSE_R(z1_R, z_ind[i, R_data_idx], ell_s_R, sf2_s_R)
            ks_P[i] = covSE_P(z1_P, z_ind[i, P_data_idx], ell_s_P, sf2_s_P)
        ks_func_T = cs.Function("K_s", [z1_T, z_ind, ell_s_T, sf2_s_T], [ks_T])
        ks_func_R = cs.Function("K_s", [z1_R, z_ind, ell_s_R, sf2_s_R], [ks_R])
        ks_func_P = cs.Function("K_s", [z1_P, z_ind, ell_s_P, sf2_s_P], [ks_P])

        K_z_zind = cs.SX.zeros(Ny, n_ind_points)
        K_z_zind_T = ks_func_T(z1_T, z_ind, self.length_scales[0], self.signal_var[0])
        K_z_zind_R = ks_func_R(z1_R, z_ind, self.length_scales[1], self.signal_var[1])
        K_z_zind_P = ks_func_P(z1_P, z_ind, self.length_scales[2], self.signal_var[2])
        self.K_z_zind_func_T = cs.Function(
            "K_z_zind", [z1_T, z_ind], [K_z_zind_T], ["z1", "z2"], ["K"]
        )
        self.K_z_zind_func_R = cs.Function(
            "K_z_zind", [z1_R, z_ind], [K_z_zind_R], ["z1", "z2"], ["K"]
        )
        self.K_z_zind_func_P = cs.Function(
            "K_z_zind", [z1_P, z_ind], [K_z_zind_P], ["z1", "z2"], ["K"]
        )

    def precompute_probabilistic_limits(self, print_sets=False):
        """This updates the constraint value limits to account for the uncertainty in the dynamics rollout.

        Args:
            print_sets (bool): True to print out the sets for debugging purposes.
        """
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        state_covariances = np.zeros((self.T + 1, nx, nx))
        input_covariances = np.zeros((self.T, nu, nu))
        # Initilize lists for the tightening of each constraint.
        state_constraint_set = []
        for state_constraint in self.constraints.state_constraints:
            state_constraint_set.append(np.zeros((state_constraint.num_constraints, T + 1)))
        input_constraint_set = []
        for input_constraint in self.constraints.input_constraints:
            input_constraint_set.append(np.zeros((input_constraint.num_constraints, T)))
        if self.x_prev is not None and self.u_prev is not None:
            cov_x = np.diag([self.initial_rollout_std**2] * nx)
            if nu == 1:
                z_batch = np.hstack(
                    (self.x_prev[:, :-1].T, self.u_prev.reshape(1, -1).T)
                )  # (T, input_dim)
            else:
                z_batch = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))  # (T, input_dim)

            # Compute the covariance of the dynamics at each time step.
            GP_T = self.gaussian_process[0]
            GP_R = self.gaussian_process[1]
            GP_P = self.gaussian_process[2]
            T_pred_point_batch = z_batch[:, self.model.nx + self.T_cmd_idx]
            R_pred_point_batch = z_batch[
                :, [self.phi_idx, self.phi_dot_idx, self.model.nx + self.phi_cmd_idx]
            ]
            P_pred_point_batch = z_batch[
                :,
                [
                    self.theta_idx,
                    self.theta_dot_idx,
                    self.model.nx + self.theta_cmd_idx,
                ],
            ]
            cov_d_batch_T = np.diag(GP_T.predict(T_pred_point_batch, return_pred=False)[1])
            cov_d_batch_R = np.diag(GP_R.predict(R_pred_point_batch, return_pred=False)[1])
            cov_d_batch_P = np.diag(GP_P.predict(P_pred_point_batch, return_pred=False)[1])
            num_batch = z_batch.shape[0]
            cov_d_batch = np.zeros((num_batch, 5, 5))
            cov_d_batch[:, 0, 0] = (
                cov_d_batch_T
                * (np.cos(z_batch[:, self.phi_idx]) * np.sin(z_batch[:, self.theta_idx])) ** 2
            )
            cov_d_batch[:, 1, 1] = cov_d_batch_T * (-np.sin(z_batch[:, self.phi_idx])) ** 2
            cov_d_batch[:, 2, 2] = (
                cov_d_batch_T
                * (np.cos(z_batch[:, self.phi_idx]) * np.cos(z_batch[:, self.theta_idx])) ** 2
            )
            cov_d_batch[:, 3, 3] = cov_d_batch_R
            cov_d_batch[:, 4, 4] = cov_d_batch_P
            cov_noise_T = GP_T.likelihood.noise.detach().numpy()
            cov_noise_R = GP_R.likelihood.noise.detach().numpy()
            cov_noise_P = GP_P.likelihood.noise.detach().numpy()
            cov_noise_batch = np.zeros((num_batch, 5, 5))
            cov_noise_batch[:, 0, 0] = (
                cov_noise_T
                * (np.cos(z_batch[:, self.phi_idx]) * np.sin(z_batch[:, self.theta_idx])) ** 2
            )
            cov_noise_batch[:, 1, 1] = cov_noise_T * (-np.sin(z_batch[:, self.phi_idx])) ** 2
            cov_noise_batch[:, 2, 2] = (
                cov_noise_T
                * (np.cos(z_batch[:, self.phi_idx]) * np.cos(z_batch[:, self.theta_idx])) ** 2
            )
            cov_noise_batch[:, 3, 3] = cov_noise_R
            cov_noise_batch[:, 4, 4] = cov_noise_P
            # discretize
            cov_noise_batch = cov_noise_batch * self.dt**2
            cov_d_batch = cov_d_batch * self.dt**2

            for i in range(T):
                state_covariances[i] = cov_x
                cov_u = self.lqr_gain @ cov_x @ self.lqr_gain.T
                input_covariances[i] = cov_u
                cov_xu = cov_x @ self.lqr_gain.T
                if self.gp_approx == "taylor":
                    raise NotImplementedError("Taylor GP approximation is currently not working.")
                elif self.gp_approx == "mean_eq":
                    cov_d = cov_d_batch[i, :, :]
                    cov_noise = cov_noise_batch[i, :, :]
                    cov_d = cov_d + cov_noise
                else:
                    raise NotImplementedError("gp_approx method is incorrect or not implemented")
                # Loop through input constraints and tighten by the required ammount.
                for ui, input_constraint in enumerate(self.constraints.input_constraints):
                    input_constraint_set[ui][:, i] = (
                        -1
                        * self.inverse_cdf
                        * np.absolute(input_constraint.A)
                        @ np.sqrt(np.diag(cov_u))
                    )
                for si, state_constraint in enumerate(self.constraints.state_constraints):
                    state_constraint_set[si][:, i] = (
                        -1
                        * self.inverse_cdf
                        * np.absolute(state_constraint.A)
                        @ np.sqrt(np.diag(cov_x))
                    )
                if self.gp_approx == "taylor":
                    raise NotImplementedError("Taylor GP rollout not implemented.")
                elif self.gp_approx == "mean_eq":
                    # Compute the next step propogated state covariance using mean equivilence.
                    cov_x = (
                        self.discrete_dfdx @ cov_x @ self.discrete_dfdx.T
                        + self.discrete_dfdx @ cov_xu @ self.discrete_dfdu.T
                        + self.discrete_dfdu @ cov_xu.T @ self.discrete_dfdx.T
                        + self.discrete_dfdu @ cov_u @ self.discrete_dfdu.T
                        + self.Bd @ cov_d @ self.Bd.T
                    )
                else:
                    raise NotImplementedError("gp_approx method is incorrect or not implemented")
            # Update Final covariance.
            for si, state_constraint in enumerate(self.constraints.state_constraints):
                state_constraint_set[si][:, -1] = (
                    -1
                    * self.inverse_cdf
                    * np.absolute(state_constraint.A)
                    @ np.sqrt(np.diag(cov_x))
                )
            state_covariances[-1] = cov_x
        self.results_dict["input_constraint_set"].append(input_constraint_set)
        self.results_dict["state_constraint_set"].append(state_constraint_set)
        self.results_dict["state_horizon_cov"].append(state_covariances)
        self.results_dict["input_horizon_cov"].append(input_covariances)
        return state_constraint_set, input_constraint_set

    def reset(self):
        """Reset the controller before running."""
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = "stabilization"
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = "tracking"
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        self.setup_prior_dynamics()
        if self.gaussian_process is not None:
            # sparse GP
            if self.sparse_gp and self.train_data["train_targets"].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data["train_targets"].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data["train_targets"].shape[0]

            self.acados_model = None
            self.ocp = None
            self.acados_ocp_solver = None

            # reinitialize the acados model and solver
            self.setup_acados_model(n_ind_points)
            self.setup_acados_optimizer(n_ind_points)
            self.acados_ocp_solver = AcadosOcpSolver(
                self.ocp, str(self.output_dir / "gpmpc_acados_ocp_solver.json"), verbose=False
            )

        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None

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
        A, B = discretize_linear_system(self.prior_ctrl.dfdx, self.prior_ctrl.dfdu, self.dt)
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
        residual = self.fd_func(x0=x, p=u)["xf"] - self.prior_dynamics_func(x0=x, p=u)["xf"]
        self.residual_func = cs.Function("residual_func", [z], [residual])
        self.fc_func = self.model.fc_func  # argument x, u

    def learn(
        self,
        env,
        num_epochs: int,
        num_train_episodes_per_epoch: int,
        num_test_episodes_per_epoch: int,
    ):
        """Performs multiple epochs learning."""

        train_runs = {0: {}}
        test_runs = {0: {}}

        # epoch seed factor
        np.random.seed(self.seed)
        epoch_seeds = np.random.randint(1000, size=num_epochs, dtype=int) * self.seed
        epoch_seeds = [int(seed) for seed in epoch_seeds]

        # Keep the reference to the same environment for all epochs -> envs are not independent
        # and will produce different trajectories
        train_env = self.env_func(randomized_init=True, seed=epoch_seeds[0])
        train_env.action_space.seed(epoch_seeds[0])
        train_envs = [train_env] * num_epochs

        test_envs = []
        # Create identical copies of test environments for each epoch
        for epoch in range(num_epochs):
            test_envs.append(self.env_func(randomized_init=True, seed=epoch_seeds[epoch]))
            test_envs[epoch].action_space.seed(epoch_seeds[epoch])

        for env in train_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]
        for env in test_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]

        # creating train and test experiments
        train_experiments = [BaseExperiment(env=env, ctrl=self) for env in train_envs[1:]]
        test_experiments = [BaseExperiment(env=env, ctrl=self) for env in test_envs[1:]]
        # first experiments are for the prior
        train_experiments.insert(
            0,
            BaseExperiment(env=train_envs[0], ctrl=self.prior_ctrl),
        )
        test_experiments.insert(
            0,
            BaseExperiment(env=test_envs[0], ctrl=self.prior_ctrl),
        )

        for episode in range(num_train_episodes_per_epoch):
            self.env = train_envs[0]
            run_results = train_experiments[0].run_evaluation(n_episodes=1)
            train_runs[0].update({episode: munch.munchify(run_results)})
        for test_ep in range(num_test_episodes_per_epoch):
            self.env = test_envs[0]
            run_results = test_experiments[0].run_evaluation(n_episodes=1)
            test_runs[0].update({test_ep: munch.munchify(run_results)})

        training_results = None
        for epoch in range(1, num_epochs):
            # only take data from the last episode from the last epoch
            episode_length = train_runs[epoch - 1][num_train_episodes_per_epoch - 1][0]["obs"][
                0
            ].shape[0]
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
            training_results = self.train_gp(input_data=train_inputs, target_data=train_targets)
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
            test_inputs, test_outputs = self.preprocess_training_data(x_seq, actions, x_next_seq)
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
            for test_ep in range(num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode][0]["obs"][0][: self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode][0]["action"][0][: self.T, :].T
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
            for episode in range(num_train_episodes_per_epoch):
                self.x_prev = train_runs[epoch - 1][episode][0]["obs"][0][: self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode][0]["action"][0][: self.T, :].T
                self.env = train_envs[epoch]
                run_results = train_experiments[epoch].run_evaluation(n_episodes=1)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            # TODO: fix data logging
            np.savez(
                self.output_dir / "epoch_data",
                data_inputs=training_results["train_inputs"],
                data_targets=training_results["train_targets"],
                train_runs=train_runs,
                test_runs=test_runs,
                num_epochs=num_epochs,
                num_train_episodes_per_epoch=num_train_episodes_per_epoch,
                num_test_episodes_per_epoch=num_test_episodes_per_epoch,
                num_samples=self.num_samples,
                train_data=self.train_data,
                test_data=self.test_data,
            )

        if training_results:
            np.savez(
                self.output_dir / "data",
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

    def gather_training_samples(self, all_runs, epoch_i, num_samples, rand_generator=None):
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

    def compute_initial_guess(self, init_state, goal_states=None):
        """Compute an initial guess of the solution to the optimization problem."""
        if goal_states is None:
            goal_states = self.get_references()
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
            sol = opti.solve()
            x_guess, u_guess = sol.value(x_var), sol.value(u_var)
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
                x_guess[:, i + 1, None] = self.dynamics_func(x0=x_guess[:, i], p=u)["xf"].toarray()
        else:
            raise Exception("Initial guess method not implemented.")

        self.x_prev = x_guess
        self.u_prev = u_guess

        # set the solver back
        self.setup_optimizer(solver=self.solver)

        return x_guess, u_guess

    def setup_optimizer(self, solver="qrsqp"):
        """Sets up nonlinear optimization problem."""
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
                    opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                    cost += self.soft_penalty * input_slack[ic_i] ** 2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

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
            ] and not ("ilqr_ref" in self.env.TASK_INFO.keys() and self.env.TASK_INFO["ilqr_ref"]):
                self.extended_ref_traj = np.concatenate(
                    [self.extended_ref_traj, self.extended_ref_traj[:, : self.T + 1]],
                    axis=1,
                )
            # Slice trajectory for horizon steps, if not long enough, repeat last state.
            start = min(self.traj_step, self.extended_ref_traj.shape[-1])
            end = min(self.traj_step + self.T + 1, self.extended_ref_traj.shape[-1])
            remain = max(0, self.T + 1 - (end - start))
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

    def reset_before_run(self, obs=None, info=None, env=None):
        """Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        """
        self.setup_results_dict()
