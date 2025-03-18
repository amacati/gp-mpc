import shutil
from functools import partial
from pathlib import Path

import casadi as cs
import gpytorch
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from safe_control_gym.mpc.gp_utils import GPRegressionModel, covSE_single, fit_gp
from safe_control_gym.mpc.mpc import MPC
from safe_control_gym.mpc.mpc_utils import (
    discretize_linear_system,
    get_cost_weight_matrix,
    reset_constraints,
)


class GPMPC:
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
    state_labels = ["x", "d_x", "y", "d_y", "z", "d_z", "phi", "theta", "d_phi", "d_theta"]
    action_labels = ["T_c", "R_c", "P_c"]

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
        device: str = "cpu",
        n_ind_points: int = 30,
        prob: float = 0.955,
        initial_rollout_std: float = 0.005,
        sparse_gp: bool = False,
        output_dir: Path = Path("results/temp"),
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
        self.output_dir = output_dir
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        self.device = device
        self.seed = seed

        # Task
        env = env_func(randomized_init=False, seed=seed)
        (
            self.constraints,
            self.state_constraints_sym,
            self.input_constraints_sym,
        ) = reset_constraints(env.constraints.constraints)
        # Model parameters
        assert prior_info is not None
        env._setup_symbolic(prior_prop=prior_info.get("prior_prop", {}))
        self.model = env.symbolic
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        self.inverse_cdf = scipy.stats.norm.ppf(
            1 - (1 / self.model.nx - (prob + 1) / (2 * self.model.nx))
        )
        self.constraint_tol = constraint_tol

        # Setup environments.
        self.traj = env.X_GOAL.T
        self.ref_action = np.repeat(env.U_GOAL.reshape(-1, 1), self.T, axis=1)
        self.traj_step = 0
        self.np_random = np.random.default_rng(seed)

        # No training data accumulated yet so keep the dynamics function as linear prior.
        self.train_data = None
        self._requires_recompile = False

        # GP and training parameters.
        self.gaussian_process = None
        self.max_n_ind_points = n_ind_points  # TODO: Move to max n ind points once debugged
        self.max_n_ind_points = 16
        self.initial_rollout_std = initial_rollout_std

        uncertain_dim = [1, 3, 5, 7, 9]
        self.Bd = np.eye(self.model.nx)[:, uncertain_dim]

        # MPC params
        self.prior_ctrl = MPC(
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

    def reset(self):
        """Reset the controller before running."""
        self.traj_step = 0
        # Dynamics model.
        if self._requires_recompile:
            assert self.gaussian_process is not None, "GP must be trained before reinitializing"
            n_ind_points = self.train_data["train_y"].shape[0]
            if self.sparse:
                n_ind_points = min(n_ind_points, self.max_n_ind_points)

            # reinitialize the acados model and solver
            acados_settings = self.output_dir / "gpmpc_acados_ocp_solver.json"
            # TODO: Remove this once we have found the bug
            if acados_settings.exists():
                acados_settings.unlink()
            code_dir = self.output_dir / "gpmpc_c_generated_code"
            if code_dir.exists():
                shutil.rmtree(code_dir)
            acados_model = self.setup_acados_model(n_ind_points)
            ocp = self.setup_acados_optimizer(acados_model, n_ind_points)
            assert not (self.output_dir / "gpmpc_acados_ocp_solver.json").exists()
            self.acados_ocp_solver = AcadosOcpSolver(
                ocp, str(self.output_dir / "gpmpc_acados_ocp_solver.json"), verbose=False
            )
            self._requires_recompile = False

        self.prior_ctrl.reset()
        # Previously solved states & inputs
        self.x_prev = None
        self.u_prev = None

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
        T_prior_data = self.prior_ctrl.t_symbolic_fn(T_cmd).full().flatten()
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

    def train_gp(self, x, y, iterations: int, learning_rate: float, test_data_ratio: float = 0.2):
        """Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.

        Returns:
            training_results (dict): Dictionary of the training results.
        """
        data_idx = list(range(x.shape[0]))
        train_idx, _ = train_test_split(data_idx, test_size=test_data_ratio, random_state=self.seed)

        train_x = x[train_idx, :]
        train_y = y[train_idx, :]
        self.train_data = {"train_x": train_x, "train_y": train_y}

        train_x = torch.Tensor(train_x).double()
        train_y = torch.Tensor(train_y).double()

        # seperate the data for T R P
        train_x_T = train_x[:, 0].reshape(-1).to(self.device)
        train_y_T = train_y[:, 0].reshape(-1).to(self.device)

        idx_R = [1, 2, 3]
        train_x_R = train_x[:, idx_R].reshape(-1, 3).to(self.device)
        train_y_R = train_y[:, 1].reshape(-1).to(self.device)

        idx_P = [4, 5, 6]
        train_x_P = train_x[:, idx_P].reshape(-1, 3).to(self.device)
        train_y_P = train_y[:, 2].reshape(-1).to(self.device)

        # Define likelihood.
        likelihood_T = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(self.device)
        likelihood_R = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(self.device)
        likelihood_P = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(self.device)

        GP_T = GPRegressionModel(train_x_T, train_y_T, likelihood_T)
        GP_R = GPRegressionModel(train_x_R, train_y_R, likelihood_R)
        GP_P = GPRegressionModel(train_x_P, train_y_P, likelihood_P)
        fit_gp(GP_T, likelihood_T, iterations, learning_rate, self.device)
        fit_gp(GP_R, likelihood_R, iterations, learning_rate, self.device)
        fit_gp(GP_P, likelihood_P, iterations, learning_rate, self.device)

        self.gaussian_process = [GP_T, GP_R, GP_P]
        self._requires_recompile = True  # Force Acados to recompile based on the new GP

    def setup_acados_model(self, n_ind_points) -> AcadosModel:
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym

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

        acados_model.name = "gpmpc"
        acados_model.x_labels = self.state_labels
        acados_model.u_labels = self.action_labels
        acados_model.t_label = "time"

        return acados_model

    def setup_acados_optimizer(self, acados_model: AcadosModel, n_ind_points: int) -> AcadosOcp:
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = acados_model

        # set cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        # cost weight matrices
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q

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
        # chance state constraint tightening
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
            nrow = state_constraint(ocp.model.x).shape[0]
            state_tighten_list.append(cs.MX.sym(f"state_tighten_{sc_i}", nrow, 1))
        # chance input constraint tightening
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(ocp.model.u))
            nrow = input_constraint(ocp.model.u).shape[0]
            input_tighten_list.append(cs.MX.sym(f"input_tighten_{ic_i}", nrow, 1))

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
        assert not (self.output_dir / "gpmpc_c_generated_code").exists()
        ocp.code_export_directory = str(self.output_dir / "gpmpc_c_generated_code")

        if self.sparse:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
        else:
            mean_post_factor_val, z_ind_val = self.precompute_gp_values()
        self.mean_post_factor_val = mean_post_factor_val
        self.z_ind_val = z_ind_val
        return ocp

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

    def select_action(self, obs, info: dict | None = None):
        if self.gaussian_process is None:
            return self.prior_ctrl.select_action(obs)
        return self.select_action_with_gp(obs)

    def select_action_with_gp(self, obs):
        assert not self._requires_recompile, "Acados model must be recompiled"
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
            # n_points = self.gaussian_process[0].n_training_samples
            # assert z_ind_val.shape == (n_points, 7), f"{z_ind_val.shape} != {(n_points, 7)}"
            mean_shape = mean_post_factor_val.shape
            # assert mean_shape == (3, n_points), f"{mean_shape} != {(3, n_points)}"
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order="F")
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order="F")
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)
            # tighten constraints
            state_constraint = state_constraint_set_prev[0]
            # Add zeros to the terminal input constraint
            input_constraint = np.concatenate(
                (input_constraint_set_prev[0], np.zeros((2 * nu, 1))), axis=-1
            )
            tighten_value = np.concatenate((state_constraint, input_constraint), axis=0)
            batch_dyn_value = np.tile(dyn_value[..., None], (1, self.T + 1))  # Shape (..., T + 1)
            p_value = np.concatenate((batch_dyn_value, tighten_value), axis=0)
            assert (
                (p_shape := self.acados_ocp_solver.get_flat("p").shape) == p_value.flatten().shape
            ), f"Acados p shape {p_shape} does not match p_value shape {p_value.flatten().shape}"
            self.acados_ocp_solver.set_flat("p", p_value.T.flatten())  # Shape (T + 1, ...)
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
        y_ref = np.concatenate((goal_states[:, :-1], self.ref_action), axis=0)
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

    def precompute_gp_values(self):
        """If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        """
        dim_gp_outputs = len(self.gaussian_process)
        n_training_samples = self.train_data["train_y"].shape[0]
        inputs = self.train_data["train_x"]
        targets = self.train_data["train_y"]
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
        assert self.gaussian_process is not None, "Gaussian process is not initialized"
        n_data_points = self.train_data["train_y"].shape[0]
        dim_gp_outputs = len(self.gaussian_process)
        train_x = torch.from_numpy(self.train_data["train_x"]).to(self.device)
        train_y = torch.from_numpy(self.train_data["train_y"]).to(self.device)
        # Choose T random training set points.
        inds = self.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
        train_x_sparse = train_x[inds]
        train_y_sparse = train_y[inds]

        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]
        idx_T = [0]
        idx_R = [1, 2, 3]
        idx_P = [4, 5, 6]
        K_zz_T_sparse = GP_T.covar_module(train_x_sparse[:, idx_T]).to_dense()
        K_zz_R_sparse = GP_R.covar_module(train_x_sparse[:, idx_R]).to_dense()
        K_zz_P_sparse = GP_P.covar_module(train_x_sparse[:, idx_P]).to_dense()
        K_zz_T_inv_sparse = torch.pinverse(K_zz_T_sparse)
        K_zz_R_inv_sparse = torch.pinverse(K_zz_R_sparse)
        K_zz_P_inv_sparse = torch.pinverse(K_zz_P_sparse)

        K_inv_sparse = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        K_inv_sparse[0] = K_zz_T_inv_sparse
        K_inv_sparse[1] = K_zz_R_inv_sparse
        K_inv_sparse[2] = K_zz_P_inv_sparse
        K_xz_T = GP_T.covar_module(train_x[:, idx_T], train_x_sparse[:, idx_T]).to_dense()
        K_xz_R = GP_R.covar_module(train_x[:, idx_R], train_x_sparse[:, idx_R]).to_dense()
        K_xz_P = GP_P.covar_module(train_x[:, idx_P], train_x_sparse[:, idx_P]).to_dense()

        Q_xx_T = K_xz_T @ K_zz_T_inv_sparse @ K_xz_T.T
        Q_xx_R = K_xz_R @ K_zz_R_inv_sparse @ K_xz_R.T
        Q_X_X_P = K_xz_P @ K_zz_P_inv_sparse @ K_xz_P.T

        Gamma_T = torch.diagonal(K_zz_T_sparse - Q_xx_T)
        Gamma_inv_T = torch.diag_embed(1 / Gamma_T)
        Gamma_R = torch.diagonal(K_zz_R_sparse - Q_xx_R)
        Gamma_inv_R = torch.diag_embed(1 / Gamma_R)
        Gamma_P = torch.diagonal(K_zz_P_sparse - Q_X_X_P)
        Gamma_inv_P = torch.diag_embed(1 / Gamma_P)

        Sigma_inv_T = K_zz_T_sparse + K_xz_T.T @ Gamma_inv_T @ K_xz_T
        Sigma_inv_R = K_zz_R_sparse + K_xz_R.T @ Gamma_inv_R @ K_xz_R
        Sigma_inv_P = K_zz_P_sparse + K_xz_P.T @ Gamma_inv_P @ K_xz_P
        Sigma_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        Sigma_inv[0] = Sigma_inv_T
        Sigma_inv[1] = Sigma_inv_R
        Sigma_inv[2] = Sigma_inv_P

        Sigma_T = torch.pinverse(Sigma_inv_T)
        Sigma_R = torch.pinverse(Sigma_inv_R)
        Sigma_P = torch.pinverse(Sigma_inv_P)
        mean_post_factor_T = Sigma_T @ K_xz_T.T @ Gamma_inv_T @ train_y_sparse[:, 0]
        mean_post_factor_R = Sigma_R @ K_xz_R.T @ Gamma_inv_R @ train_y_sparse[:, 1]
        mean_post_factor_P = Sigma_P @ K_xz_P.T @ Gamma_inv_P @ train_y_sparse[:, 2]

        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points))
        mean_post_factor[0] = mean_post_factor_T
        mean_post_factor[1] = mean_post_factor_R
        mean_post_factor[2] = mean_post_factor_P

        return (
            mean_post_factor.detach().numpy(),
            Sigma_inv.detach().numpy(),
            K_inv_sparse.detach().numpy(),
            train_x_sparse.numpy(force=True),
        )

    def create_sparse_GP_machinery(self, n_ind_points):
        """This setups the gaussian process approximations for FITC formulation."""
        idx_R = [1, 2, 3]
        idx_P = [4, 5, 6]
        GP_T = self.gaussian_process[0]
        GP_R = self.gaussian_process[1]
        GP_P = self.gaussian_process[2]

        lengthscales_T = GP_T.covar_module.base_kernel.lengthscale.numpy(force=True)
        lengthscales_R = GP_R.covar_module.base_kernel.lengthscale.numpy(force=True)
        lengthscales_P = GP_P.covar_module.base_kernel.lengthscale.numpy(force=True)
        signal_var_T = GP_T.covar_module.outputscale.numpy(force=True)
        signal_var_R = GP_R.covar_module.outputscale.numpy(force=True)
        signal_var_P = GP_P.covar_module.outputscale.numpy(force=True)
        gp_K_T = GP_T.K.numpy(force=True)
        gp_K_R = GP_R.K.numpy(force=True)
        gp_K_P = GP_P.K.numpy(force=True)

        # stacking
        lengthscales = np.vstack((lengthscales_T, lengthscales_R, lengthscales_P))
        signal_var = np.array([signal_var_T, signal_var_R, signal_var_P])
        gp_K = np.zeros((3, gp_K_T.shape[0], gp_K_T.shape[1]))
        gp_K[0] = gp_K_T
        gp_K[1] = gp_K_R
        gp_K[2] = gp_K_P

        length_scales = lengthscales.squeeze()
        signal_var = signal_var.squeeze()
        Nx = self.train_data["train_x"].shape[1]
        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind,
        # length scales and signal variance. We need the CasADI version of this so that it can be
        # symbolically differentiated in the MPC optimization.
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
            ks_R[i] = covSE_R(z1_R, z_ind[i, idx_R], ell_s_R, sf2_s_R)
            ks_P[i] = covSE_P(z1_P, z_ind[i, idx_P], ell_s_P, sf2_s_P)
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
            z_batch_tensor = torch.from_numpy(z_batch).to(self.device)

            # Compute the covariance of the dynamics at each time step.
            GP_T = self.gaussian_process[0].eval()
            GP_R = self.gaussian_process[1].eval()
            GP_P = self.gaussian_process[2].eval()
            T_pred_point_batch = z_batch_tensor[:, self.model.nx + self.idx["T_cmd"]]
            R_pred_point_batch = z_batch_tensor[
                :, [self.idx["phi"], self.idx["phi_dot"], self.model.nx + self.idx["phi_cmd"]]
            ]
            P_pred_point_batch = z_batch_tensor[
                :,
                [
                    self.idx["theta"],
                    self.idx["theta_dot"],
                    self.model.nx + self.idx["theta_cmd"],
                ],
            ]
            with torch.no_grad(), gpytorch.settings.fast_pred_var(
                state=True
            ), gpytorch.settings.fast_pred_samples(state=True):
                cov_d_T = torch.diag(GP_T.likelihood(GP_T(T_pred_point_batch)).covariance_matrix)
                cov_d_T = cov_d_T.numpy(force=True)
                cov_d_R = torch.diag(GP_R.likelihood(GP_R(R_pred_point_batch)).covariance_matrix)
                cov_d_P = torch.diag(GP_P.likelihood(GP_P(P_pred_point_batch)).covariance_matrix)

            num_batch = z_batch.shape[0]
            cov_d_batch = np.zeros((num_batch, 5, 5))
            cov_d_batch[:, 0, 0] = (
                cov_d_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.sin(z_batch[:, self.idx["theta"]])) ** 2
            )
            cov_d_batch[:, 1, 1] = cov_d_T * (-np.sin(z_batch[:, self.idx["phi"]])) ** 2
            cov_d_batch[:, 2, 2] = (
                cov_d_T
                * (np.cos(z_batch[:, self.idx["phi"]]) * np.cos(z_batch[:, self.idx["theta"]])) ** 2
            )
            cov_d_batch[:, 3, 3] = cov_d_R.numpy(force=True)
            cov_d_batch[:, 4, 4] = cov_d_P.numpy(force=True)
            cov_noise_T = GP_T.likelihood.noise.numpy(force=True)
            cov_noise_R = GP_R.likelihood.noise.numpy(force=True)
            cov_noise_P = GP_P.likelihood.noise.numpy(force=True)
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

    @staticmethod
    def setup_prior_dynamics(dfdx: NDArray, dfdu: NDArray, Q: NDArray, R: NDArray, dt: float):
        """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics."""
        # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
        A, B = discretize_linear_system(dfdx, dfdu, dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        lqr_gain = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
        return A, B, lqr_gain

    def get_references(self):
        """Constructs reference states along mpc horizon.(nx, T+1)."""
        # We append the T+1 states of the trajectory to the goal_states such that the vel states
        # won't drop at the end of an episode
        extended_traj = np.concatenate([self.traj, self.traj[:, : self.T + 1]], axis=1)
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = min(self.traj_step, extended_traj.shape[-1])
        end = min(self.traj_step + self.T + 1, extended_traj.shape[-1])
        remain = max(0, self.T + 1 - (end - start))
        tail = np.tile(extended_traj[:, end][:, None], (1, remain))
        goal_states = np.concatenate([extended_traj[:, start:end], tail], -1)
        return goal_states  # (nx, T+1).
