import time
from functools import partial
from typing import Any

import casadi as cs
import gpytorch
import munch
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.mpc.gp_utils import (
    GaussianProcess,
    ZeroMeanIndependentGPModel,
    covSE_single,
)
from safe_control_gym.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.mpc.mpc_utils import (
    discretize_linear_system,
    get_cost_weight_matrix,
    reset_constraints,
)


class GPMPC_ACADOS_TRP:
    """Implements a GP-MPC controller with Acados optimization."""

    idx = {
        "phi": 6,
        "theta": 7,
        "phi_dot": 8,
        "theta_dot": 9,
        "T_cmd": 0,
        "phi_cmd": 1,
        "theta_cmd": 2,
    }

    def __init__(
        self,
        env_func,
        num_samples: int,
        prior_info: dict,
        horizon: int,
        q_mpc: list,
        r_mpc: list,
        seed: int = 1337,
        constraint_tol: float = 1e-8,
        test_data_ratio: float = 0.2,
        optimization_iterations: list = None,
        learning_rate: list = None,
        device: str = "cpu",
        n_ind_points: int = 30,
        prob: float = 0.955,
        initial_rollout_std: float = 0.005,
        sparse_gp: bool = False,
        output_dir: str = "results/temp",
    ):
        self.q_mpc = q_mpc
        self.r_mpc = r_mpc
        self.num_samples = num_samples

        if prior_info is None or prior_info == {}:
            raise ValueError("GPMPC requires prior_prop to be defined.")
        self.soft_constraints_params = {
            "gp_soft_constraints": False,
            "gp_soft_constraints_coeff": 0,
            "prior_soft_constraints": False,
            "prior_soft_constraints_coeff": 0,
        }
        self.sparse = sparse_gp
        self.env_func = env_func  # TODO: remove after removing the train function
        self.output_dir = output_dir
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        self.device = device
        self.seed = seed

        # Task
        self.env = env_func(randomized_init=False, seed=seed)  # TODO: Remove
        (
            self.constraints,
            self.state_constraints_sym,
            self.input_constraints_sym,
        ) = reset_constraints(self.env.constraints.constraints)
        # Model parameters
        self.model = self.get_prior(self.env, prior_info)
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        self.constraint_tol = constraint_tol

        # Setup environments.
        self.env = env_func(randomized_init=False, seed=seed)
        self.env_training = env_func(randomized_init=True, seed=seed)
        self.traj = self.env.X_GOAL.T
        self.traj_step = 0

        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self.data_inputs = None
        self.data_targets = None

        # GP and training parameters.
        self.gaussian_process = None
        self.test_data_ratio = test_data_ratio  # TODO: Marked for removal
        self.optimization_iterations = optimization_iterations  # TODO: Marked for removal
        self.learning_rate = learning_rate  # TODO: Marked for removal
        self.prob = prob
        self.n_ind_points = n_ind_points  # TODO: Rename to something more descriptive
        self.initial_rollout_std = initial_rollout_std

        uncertain_dim = [1, 3, 5, 7, 9]
        self.Bd = np.eye(self.model.nx)[:, uncertain_dim]

        # MPC params
        self.prior_ctrl = MPC_ACADOS(
            env_func=partial(env_func, inertial_prop=prior_info["prior_prop"]),
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            device=device,
            seed=seed,
            prior_info=prior_info,
        )
        _, _, prior_dfdx, prior_dfdu = self.prior_ctrl.dynamics_fn()
        self.discrete_dfdx, self.discrete_dfdu, self.lqr_gain = self.setup_prior_dynamics(
            prior_dfdx, prior_dfdu, self.Q, self.R, self.dt
        )
        self.prior_dynamics_fn = self.prior_ctrl.model.fc_func

        self.x_prev = None
        self.u_prev = None
        self.results_dict = {}  # Required to remain compatible with base_experiment

    def reset(self):
        """Reset the controller before running."""
        tstart = time.time()
        self.traj = self.env.X_GOAL.T
        self.traj_step = 0
        # Dynamics model.
        if self.gaussian_process is not None:
            # sparse GP
            if self.sparse and self.train_data["train_targets"].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data["train_targets"].shape[0]
            elif self.sparse:
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

        self.prior_ctrl.reset()  # TODO: Check if we need to reset the controller here
        # Previously solved states & inputs
        self.x_prev = None
        self.u_prev = None
        print(f"GPMPC_ACADOS_TRP.reset(): Reset took {time.time() - tstart:.2f} seconds.")

    def preprocess_data(
        self, x_seq: list[NDArray], u_seq: list[NDArray], x_next_seq: list[NDArray]
    ) -> tuple[NDArray, NDArray]:
        """Converts trajectory data for GP trianing.

        Args:
            x_seq: state sequence of Arrays (nx,).
            u_seq: action sequence of Arrays (nu,).
            x_next_seq: next state sequence of Arrays (nx,).

        Returns:
            Inputs and targets for GP training, (N, nx+nu) and (N, nx).
        """
        # Get the predicted dynamics. This is a linear prior, thus we need to account for the fact
        # that it is linearized about an eq using self.X_GOAL and self.U_GOAL.
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

        theta_true = x_dot_seq[:, self.idx["theta"]]
        theta_prior = self.prior_dynamics_fn(x=x_seq.T, u=u_seq.T)["f"].toarray()[
            self.idx["theta"], :
        ]
        targets_theta = (theta_true - theta_prior).reshape(-1, 1)
        input_theta = np.concatenate(
            [
                x_seq[:, self.idx["theta"]].reshape(-1, 1),
                x_seq[:, self.idx["theta_dot"]].reshape(-1, 1),
                u_seq[:, self.idx["theta_cmd"]].reshape(-1, 1),
            ],
            axis=1,
        )

        phi_true = x_dot_seq[:, self.idx["phi"]]
        phi_prior = self.prior_dynamics_fn(x=x_seq.T, u=u_seq.T)["f"].toarray()[self.idx["phi"], :]
        targets_phi = (phi_true - phi_prior).reshape(-1, 1)
        input_phi = np.concatenate(
            [
                x_seq[:, self.idx["phi"]].reshape(-1, 1),
                x_seq[:, self.idx["phi_dot"]].reshape(-1, 1),
                u_seq[:, self.idx["phi_cmd"]].reshape(-1, 1),
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
        overwrite_saved_data: bool = False,
    ):
        """Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.

        Returns:
            training_results (dict): Dictionary of the training results.
        """
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

        GP_T.train(
            train_input_T,
            train_target_T,
            test_inputs_T,
            test_targets_T,
            n_train=self.optimization_iterations[0],
            learning_rate=self.learning_rate[0],
            device=self.device,
            fname=self.output_dir / "best_model_T.pth",
        )
        GP_R.train(
            train_input_R,
            train_target_R,
            test_inputs_R,
            test_targets_R,
            n_train=self.optimization_iterations[1],
            learning_rate=self.learning_rate[1],
            device=self.device,
            fname=self.output_dir / "best_model_R.pth",
        )
        GP_P.train(
            train_input_P,
            train_target_P,
            test_inputs_P,
            test_targets_P,
            n_train=self.optimization_iterations[2],
            learning_rate=self.learning_rate[2],
            device=self.device,
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

        T_pred_point = z[self.idx["T_cmd"] + self.model.nx]
        R_pred_point = z[
            [self.idx["phi"], self.idx["phi_dot"], self.idx["phi_cmd"] + self.model.nx]
        ]
        P_pred_point = z[
            [self.idx["theta"], self.idx["theta_dot"], self.idx["theta_cmd"] + self.model.nx]
        ]
        if self.sparse:
            self.create_sparse_GP_machinery(n_ind_points)
            # sparse GP inducing points
            # z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, nx+nu)
            # mean_post_factor should be of shape (len(nx), n_ind_points)
            # Here we create the corresponding parameters since acados supports only 1D parameters
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

        f_cont = self.prior_dynamics_fn(x=acados_model.x, u=acados_model.u)["f"] + cs.vertcat(
            0,
            T_pred
            * (cs.cos(acados_model.x[self.idx["phi"]]) * cs.sin(acados_model.x[self.idx["theta"]])),
            0,
            T_pred * (-cs.sin(acados_model.x[self.idx["phi"]])),
            0,
            T_pred
            * (cs.cos(acados_model.x[self.idx["phi"]]) * cs.cos(acados_model.x[self.idx["theta"]])),
            0,
            0,
            R_pred,
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
        if self.sparse:
            ocp.model.p = cs.vertcat(ocp.model.p, tighten_param)
            ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))  # dummy values
        else:
            ocp.model.p = tighten_param
            ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))  # dummy values

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"

        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 25
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
        if self.sparse:
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

        Note:
            All constraints in safe-control-gym are defined as g(x, u) <= constraint_tol. However,
            acados requires the constraints to be defined as lb <= g(x, u) <= ub. Thus, a large
            negative number (-1e8) is used as the lower bound.
            See: https://github.com/acados/acados/issues/650


        Args:
            ocp: AcadosOcp solver.
            h0_expr: Initial state constraint expression from casadi.
            h_expr: State and input constraint expression from casadi.
            he_expr: Terminal state constraint expression from casadi.
            state_tighten_list: List of casadi SX variables for state constraint tightening
            input_tighten_list: List of casadi SX variables for input constraint tightening

        Returns:
            The acados ocp object with constraints set
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

    def select_action(self, obs, info: dict | None = None):
        if self.gaussian_process is None:
            return self.prior_ctrl.select_action(obs)
        return self.select_action_with_gp(obs)

    def select_action_with_gp(self, obs):
        nx, nu = self.model.nx, self.model.nu
        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)
        # Omit warmstart for simplicity
        for idx in range(self.T + 1):
            self.acados_ocp_solver.set(idx, "x", obs)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, "u", np.zeros((nu,)))

        # use the precomputed values
        mean_post_factor_val = self.mean_post_factor_val
        z_ind_val = self.z_ind_val

        # Set the probabilistic state and input constraint set limits.
        # Tightening at the first step is possible if self.compute_initial_guess is used
        (
            state_constraint_set_prev,
            input_constraint_set_prev,
        ) = self.precompute_probabilistic_limits()
        # set acados parameters
        if self.sparse:
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
        self.traj_step += 1
        y_ref = np.concatenate(
            (goal_states[:, :-1], np.repeat(self.env.U_GOAL.reshape(-1, 1), self.T, axis=1)),
            axis=0,
        )
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, "yref", y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)

        # solve the optimization problem
        status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            self.acados_ocp_solver.print_statistics()
            raise RuntimeError(f"acados returned status {status}. ")

        action = self.acados_ocp_solver.get(0, "u")
        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))

        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, "x")
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, "u")
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

    def precompute_sparse_gp_values(self, n_ind_points: int):
        """Use the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points: Number of inducing points.
        """
        n_data_points = self.train_data["train_targets"].shape[0]
        dim_gp_outputs = len(self.gaussian_process)
        inputs = self.train_data["train_inputs"]
        targets = self.train_data["train_targets"]
        # Choose T random training set points.
        inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
        z_ind = inputs[inds]

        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]
        T_data_idx = [0]
        R_data_idx = [1, 2, 3]
        P_data_idx = [4, 5, 6]
        K_zind_zind_T = GP_T.model.covar_module(torch.from_numpy(z_ind[:, T_data_idx]).double())
        K_zind_zind_R = GP_R.model.covar_module(torch.from_numpy(z_ind[:, R_data_idx]).double())
        K_zind_zind_P = GP_P.model.covar_module(torch.from_numpy(z_ind[:, P_data_idx]).double())
        K_zind_zind_inv_T = torch.pinverse(K_zind_zind_T.evaluate().detach())
        K_zind_zind_inv_R = torch.pinverse(K_zind_zind_R.evaluate().detach())
        K_zind_zind_inv_P = torch.pinverse(K_zind_zind_P.evaluate().detach())
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

        Q_X_X_T = K_x_zind_T @ K_zind_zind_inv_T @ K_x_zind_T.T
        Q_X_X_R = K_x_zind_R @ K_zind_zind_inv_R @ K_x_zind_R.T
        Q_X_X_P = K_x_zind_P @ K_zind_zind_inv_P @ K_x_zind_P.T

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
        gp_K_plus_noise_T = GP_T.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_R = GP_R.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_P = GP_P.model.K_plus_noise.detach().numpy()

        # stacking
        lengthscales = np.vstack((lengthscales_T, lengthscales_R, lengthscales_P))
        signal_var = np.array([signal_var_T, signal_var_R, signal_var_P])
        gp_K_plus_noise = np.zeros((3, gp_K_plus_noise_T.shape[0], gp_K_plus_noise_T.shape[1]))
        gp_K_plus_noise[0] = gp_K_plus_noise_T
        gp_K_plus_noise[1] = gp_K_plus_noise_R
        gp_K_plus_noise[2] = gp_K_plus_noise_P

        length_scales = lengthscales.squeeze()
        signal_var = signal_var.squeeze()
        Nx = self.train_data["train_inputs"].shape[1]
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

        K_z_zind_T = ks_func_T(z1_T, z_ind, length_scales[0], signal_var[0])
        K_z_zind_R = ks_func_R(z1_R, z_ind, length_scales[1], signal_var[1])
        K_z_zind_P = ks_func_P(z1_P, z_ind, length_scales[2], signal_var[2])
        self.K_z_zind_func_T = cs.Function(
            "K_z_zind", [z1_T, z_ind], [K_z_zind_T], ["z1", "z2"], ["K"]
        )
        self.K_z_zind_func_R = cs.Function(
            "K_z_zind", [z1_R, z_ind], [K_z_zind_R], ["z1", "z2"], ["K"]
        )
        self.K_z_zind_func_P = cs.Function(
            "K_z_zind", [z1_P, z_ind], [K_z_zind_P], ["z1", "z2"], ["K"]
        )

    def precompute_probabilistic_limits(self):
        """Update the constraint value limits to account for the uncertainty in the rollout."""
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
            z_batch = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))  # (T, input_dim)

            # Compute the covariance of the dynamics at each time step.
            GP_T = self.gaussian_process[0]
            GP_R = self.gaussian_process[1]
            GP_P = self.gaussian_process[2]
            T_pred_point_batch = z_batch[:, self.model.nx + self.idx["T_cmd"]]
            R_pred_point_batch = z_batch[
                :, [self.idx["phi"], self.idx["phi_dot"], self.model.nx + self.idx["phi_cmd"]]
            ]
            P_pred_point_batch = z_batch[
                :,
                [
                    self.idx["theta"],
                    self.idx["theta_dot"],
                    self.model.nx + self.idx["theta_cmd"],
                ],
            ]
            cov_d_batch_T = np.diag(GP_T.predict(T_pred_point_batch, return_pred=False)[1])
            cov_d_batch_R = np.diag(GP_R.predict(R_pred_point_batch, return_pred=False)[1])
            cov_d_batch_P = np.diag(GP_P.predict(P_pred_point_batch, return_pred=False)[1])
            num_batch = z_batch.shape[0]
            cov_d_batch = np.zeros((num_batch, 5, 5))
            cov_d_batch[:, 0, 0] = (
                cov_d_batch_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.sin(z_batch[:, self.idx["theta"]])) ** 2
            )
            cov_d_batch[:, 1, 1] = cov_d_batch_T * (-np.sin(z_batch[:, self.idx["phi"]])) ** 2
            cov_d_batch[:, 2, 2] = (
                cov_d_batch_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.cos(z_batch[:, self.idx["theta"]])) ** 2
            )
            cov_d_batch[:, 3, 3] = cov_d_batch_R
            cov_d_batch[:, 4, 4] = cov_d_batch_P
            cov_noise_T = GP_T.likelihood.noise.detach().numpy()
            cov_noise_R = GP_R.likelihood.noise.detach().numpy()
            cov_noise_P = GP_P.likelihood.noise.detach().numpy()
            cov_noise_batch = np.zeros((num_batch, 5, 5))
            cov_noise_batch[:, 0, 0] = (
                cov_noise_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.sin(z_batch[:, self.idx["theta"]])) ** 2
            )
            cov_noise_batch[:, 1, 1] = cov_noise_T * (-np.sin(z_batch[:, self.idx["phi"]])) ** 2
            cov_noise_batch[:, 2, 2] = (
                cov_noise_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.cos(z_batch[:, self.idx["theta"]])) ** 2
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
                # GP mean approximation
                cov_d = cov_d_batch[i, :, :]
                cov_noise = cov_noise_batch[i, :, :]
                cov_d = cov_d + cov_noise
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
                # Compute the next step propogated state covariance using mean equivilence.
                cov_x = (
                    self.discrete_dfdx @ cov_x @ self.discrete_dfdx.T
                    + self.discrete_dfdx @ cov_xu @ self.discrete_dfdu.T
                    + self.discrete_dfdu @ cov_xu.T @ self.discrete_dfdx.T
                    + self.discrete_dfdu @ cov_u @ self.discrete_dfdu.T
                    + self.Bd @ cov_d @ self.Bd.T
                )
            # Update Final covariance.
            for si, state_constraint in enumerate(self.constraints.state_constraints):
                state_constraint_set[si][:, -1] = (
                    -1
                    * self.inverse_cdf
                    * np.absolute(state_constraint.A)
                    @ np.sqrt(np.diag(cov_x))
                )
            state_covariances[-1] = cov_x
        return state_constraint_set, input_constraint_set

    def get_prior(self, env, prior_info: dict):
        """Fetch the prior model from the env for the controller.

        Args:
            env (BenchmarkEnv): the environment to fetch prior model from.
            prior_info (dict): specifies the prior properties or other things to
                overwrite the default prior model in the env.

        Returns:
            SymbolicModel: CasAdi prior model.
        """
        assert prior_info is not None
        prior_prop = prior_info.get("prior_prop", {})
        env._setup_symbolic(prior_prop=prior_prop)
        return env.symbolic

    @staticmethod
    def setup_prior_dynamics(dfdx: NDArray, dfdu: NDArray, Q: NDArray, R: NDArray, dt: float):
        """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics."""
        # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
        A, B = discretize_linear_system(dfdx, dfdu, dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        lqr_gain = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
        return A, B, lqr_gain

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

        for epoch in range(1, num_epochs):
            # only take data from the last episode from the last epoch
            episode_length = train_runs[epoch - 1][num_train_episodes_per_epoch - 1][0]["obs"][
                0
            ].shape[0]
            x_seq, actions, x_next_seq, _ = self.gather_training_samples(
                train_runs,
                epoch - 1,
                self.num_samples,
                train_envs[epoch - 1].np_random,
            )
            train_inputs, train_targets = self.preprocess_data(
                x_seq, actions, x_next_seq
            )  # np.ndarray
            self.train_gp(input_data=train_inputs, target_data=train_targets)
            max_steps = train_runs[epoch - 1][episode][0]["obs"][0].shape[0]
            x_seq, actions, x_next_seq, _ = self.gather_training_samples(
                train_runs, epoch - 1, max_steps
            )

            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode][0]["obs"][0][: self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode][0]["action"][0][: self.T, :].T
                self.env = test_envs[epoch]
                run_results = test_experiments[epoch].run_evaluation(n_episodes=1)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})

            x_seq, actions, x_next_seq, _ = self.gather_training_samples(
                test_runs, epoch - 1, episode_length
            )
            train_inputs, train_targets = self.preprocess_data(x_seq, actions, x_next_seq)
            # gather training data
            train_runs[epoch] = {}
            for episode in range(num_train_episodes_per_epoch):
                self.x_prev = train_runs[epoch - 1][episode][0]["obs"][0][: self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode][0]["action"][0][: self.T, :].T
                self.env = train_envs[epoch]
                run_results = train_experiments[epoch].run_evaluation(n_episodes=1)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

        # close environments
        for experiment in train_experiments:
            experiment.env.close()
        for experiment in test_experiments:
            experiment.env.close()

        return train_runs, test_runs

    def gather_training_samples(self, all_runs, epoch_i, num_samples, rng=None):
        n_episodes = len(all_runs[epoch_i].keys())
        num_samples_per_episode = int(num_samples / n_episodes)
        x_seq_int = []
        x_next_seq_int = []
        actions_int = []
        for episode_i in range(n_episodes):
            run_results_int = all_runs[epoch_i][episode_i][0]
            n = run_results_int["action"][0].shape[0]
            if num_samples_per_episode < n:
                if rng is not None:
                    rand_inds_int = rng.choice(n - 1, num_samples_per_episode, replace=False)
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

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(
                x=x_var[:, i],
                u=u_var[:, i],
                Xr=x_ref[:, i],
                Ur=self.env.U_GOAL,
                Q=self.Q,
                R=self.R,
            )["l"]
        # Terminal cost.
        cost += cost_func(
            x=x_var[:, -1],
            u=np.zeros((nu, 1)),
            Xr=x_ref[:, -1],
            Ur=self.env.U_GOAL,
            Q=self.Q,
            R=self.R,
        )["l"]
        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])["xf"]
            opti.subject_to(x_var[:, i + 1] == next_state)

            for state_constraint in self.state_constraints_sym:
                opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for input_constraint in self.input_constraints_sym:
                opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

        # Final state constraints.
        for state_constraint in self.state_constraints_sym:
            opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        opti.solver(solver, {"expand": True, "error_on_fail": False})

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
        # We append the T+1 states of the trajectory to the goal_states such that the vel states
        # won't drop at the end of an episode
        extended_ref_traj = np.concatenate([self.traj, self.traj[:, : self.T + 1]], axis=1)
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = min(self.traj_step, extended_ref_traj.shape[-1])
        end = min(self.traj_step + self.T + 1, extended_ref_traj.shape[-1])
        remain = max(0, self.T + 1 - (end - start))
        goal_states = np.concatenate(
            [
                extended_ref_traj[:, start:end],
                np.tile(extended_ref_traj[:, -1:], (1, remain)),
            ],
            -1,
        )
        return goal_states  # (nx, T+1).

    def reset_before_run(self, obs: Any = None, info: Any = None, env: Any = None):
        """Reinitialize just the controller before a new run."""
        pass
