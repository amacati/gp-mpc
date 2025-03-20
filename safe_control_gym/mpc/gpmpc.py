import time
from functools import partial, wraps
from pathlib import Path

import casadi as cs
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from gpytorch.settings import fast_pred_samples, fast_pred_var
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from safe_control_gym.mpc.gp import (
    GaussianProcess,
    covSE_single,
    fit_gp,
    gpytorch_predict2casadi,
)
from safe_control_gym.mpc.mpc import MPC
from safe_control_gym.mpc.mpc_utils import discretize_linear_system


def timed(func):
    """Decorator that prints the execution time of a function.

    Args:
        func: The function to time.

    Returns:
        The wrapped function that prints timing information.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds to execute")
        return result

    return wrapper


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
        env_fn,
        num_samples: int,
        prior_info: dict,
        horizon: int,
        q_mpc: list,
        r_mpc: list,
        seed: int = 1337,
        device: str = "cpu",
        n_ind_points: int = 30,
        prob: float = 0.955,
        initial_rollout_std: float = 0.005,
        sparse_gp: bool = False,
        output_dir: Path = Path("results/temp"),
    ):
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
        env = env_fn(randomized_init=False, seed=seed)
        self.state_cnstr, self.input_cnstr = env.constraints.constraints
        # Model parameters
        assert prior_info is not None
        env._setup_symbolic(prior_prop=prior_info.get("prior_prop", {}))
        self.model = env.symbolic
        self.dt = self.model.dt
        self.T = horizon
        assert len(q_mpc) == self.model.nx and len(r_mpc) == self.model.nu
        self.Q = np.diag(q_mpc)
        self.R = np.diag(r_mpc)

        # Setup environments.
        self.traj = env.X_GOAL.T
        self.ref_action = np.repeat(env.U_GOAL.reshape(-1, 1), self.T, axis=1)
        self.traj_step = 0
        self.np_random = np.random.default_rng(seed)

        # GP and training parameters.
        self.gp_idx = [[0], [1, 2, 3], [4, 5, 6]]  # State input indices for each GP
        self.gaussian_process = None
        self._requires_recompile = False
        self.inverse_cdf = scipy.stats.norm.ppf(
            1 - (1 / self.model.nx - (prob + 1) / (2 * self.model.nx))
        )
        self.n_ind_points = n_ind_points
        self.max_n_ind_points = n_ind_points  # TODO: Move to max n ind points once debugged
        self.initial_rollout_std = initial_rollout_std

        uncertain_dim = [1, 3, 5, 7, 9]
        self.Bd = np.eye(self.model.nx)[:, uncertain_dim]

        # MPC params
        env_fn = partial(env_fn, inertial_prop=prior_info["prior_prop"])
        self.prior_ctrl = MPC(
            env_fn=env_fn, horizon=horizon, q_mpc=q_mpc, r_mpc=r_mpc, output_dir=output_dir
        )
        x_eq, u_eq = self.prior_ctrl.model.X_EQ, self.prior_ctrl.model.U_EQ
        dfdx_dfdu = self.prior_ctrl.model.df_func(x=x_eq, u=u_eq)
        prior_dfdx, prior_dfdu = dfdx_dfdu["dfdx"].toarray(), dfdx_dfdu["dfdu"].toarray()
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
            n_ind_points = self.gaussian_process[0].train_targets.shape[0]
            if self.sparse:
                # TODO: Fix by setting to max_n_ind_points
                n_ind_points = min(n_ind_points, self.n_ind_points)
            # reinitialize the acados model and solver
            acados_model = self.setup_acados_model(n_ind_points)
            self.ocp = self.setup_acados_optimizer(acados_model, n_ind_points)
            self.acados_solver = AcadosOcpSolver(
                self.ocp, str(self.output_dir / "gpmpc_acados_ocp_solver.json"), verbose=False
            )
            self._requires_recompile = False

        self.prior_ctrl.reset()
        # Previously solved states & inputs
        self.x_prev = None
        self.u_prev = None

    def preprocess_data(self, x: NDArray, u: NDArray, x_next: NDArray) -> tuple[NDArray, NDArray]:
        """Convert trajectory data for GP trianing.

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
        thrust_cmd = u[:, 0]
        x_dot = (x_next - x) / dt  # Approximate via numerical differentiation

        # Faster than broadcasted version of np.linalg.norm
        acc = np.sqrt(x_dot[:, 1] ** 2 + x_dot[:, 3] ** 2 + (x_dot[:, 5] + g) ** 2)
        acc_prior = self.prior_ctrl.t_symbolic_fn(thrust_cmd).full().flatten()
        acc_target = acc - acc_prior
        acc_input = thrust_cmd.reshape(-1, 1)

        theta = x_dot[:, self.idx["theta"]]
        theta_prior = self.prior_dynamics_fn(x=x.T, u=u.T)["f"].toarray()[self.idx["theta"], :]
        theta_target = theta - theta_prior
        theta_input = np.vstack(
            (x[:, self.idx["theta"]], x[:, self.idx["theta_dot"]], u[:, self.idx["theta_cmd"]])
        ).T

        phi = x_dot[:, self.idx["phi"]]
        phi_prior = self.prior_dynamics_fn(x=x.T, u=u.T)["f"].toarray()[self.idx["phi"], :]
        phi_target = phi - phi_prior
        phi_input = np.vstack(
            (x[:, self.idx["phi"]], x[:, self.idx["phi_dot"]], u[:, self.idx["phi_cmd"]])
        ).T

        train_input = np.concatenate([acc_input, phi_input, theta_input], axis=-1)
        train_output = np.vstack((acc_target, phi_target, theta_target)).T
        return train_input, train_output

    def train_gp(self, x: NDArray, y: NDArray, lr: float, iterations: int, test_size: float = 0.2):
        """Fit the GPs to the training data."""
        seed = self.np_random.integers(0, 2**32 - 1)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, random_state=seed)
        x_train = torch.tensor(x_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)

        # Initialize GPs
        self.gaussian_process = []
        for i, gp_idx in enumerate(self.gp_idx):
            gp = GaussianProcess(x_train[:, gp_idx], y_train[:, i])
            fit_gp(gp, n_train=iterations, lr=lr, device=self.device)
            self.gaussian_process.append(gp)

        self._requires_recompile = True

    # TODO: Refactor
    def setup_acados_model(self, n_ind_points) -> AcadosModel:
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point

        idx_cs_T = [self.idx["T_cmd"] + self.model.nx]
        idx_cs_R = [self.idx["phi"], self.idx["phi_dot"], self.idx["phi_cmd"] + self.model.nx]
        idx_cs_P = [self.idx["theta"], self.idx["theta_dot"], self.idx["theta_cmd"] + self.model.nx]

        if self.sparse:
            self.create_sparse_GP_machinery(n_ind_points)
            # Here we create the corresponding parameters since acados supports only 1D parameters
            sparse_idx = cs.MX.sym("sparse_idx", n_ind_points, 7)
            posterior_mean = cs.MX.sym("posterior_mean", 3, n_ind_points)
            acados_model.p = cs.vertcat(
                cs.reshape(sparse_idx, -1, 1), cs.reshape(posterior_mean, -1, 1)
            )

            T_pred = cs.sum2(
                self.K_zs_T_cs(z1=z[idx_cs_T], z2=sparse_idx)["K"] * posterior_mean[0, :]
            )
            R_pred = cs.sum2(
                self.K_zs_R_cs(z1=z[idx_cs_R], z2=sparse_idx)["K"] * posterior_mean[1, :]
            )
            P_pred = cs.sum2(
                self.K_zs_P_cs(z1=z[idx_cs_P], z2=sparse_idx)["K"] * posterior_mean[2, :]
            )
        else:
            T_pred = gpytorch_predict2casadi(self.gaussian_process[0])(z=z[idx_cs_T])["mean"]
            R_pred = gpytorch_predict2casadi(self.gaussian_process[1])(z=z[idx_cs_R])["mean"]
            P_pred = gpytorch_predict2casadi(self.gaussian_process[2])(z=z[idx_cs_P])["mean"]

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
        ocp = AcadosOcp()
        ocp.model = acados_model
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # Configure costs
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx : (nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Constraints
        state_cnstr = self.state_cnstr.sym_func(ocp.model.x)
        input_cnstr = self.input_cnstr.sym_func(ocp.model.u)
        ocp = self.setup_acados_constraints(ocp, state_cnstr, input_cnstr)

        # Initialize with placeholder zero values
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
        ocp.constraints.x0 = np.zeros((nx))

        # Set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 25
        ocp.solver_options.tf = self.T * self.dt  # prediction duration
        ocp.code_export_directory = str(self.output_dir / "gpmpc_c_generated_code")

        # TODO: Remove assignments to class variables
        self.n_ind_points = n_ind_points  # TODO: Maybe remove

        if self.sparse:
            posterior_mean, sparse_idx_val = self.precompute_sparse_posterior_mean(n_ind_points)
        else:
            posterior_mean, sparse_idx_val = self.precompute_posterior_mean()
        self.posterior_mean = posterior_mean
        self.sparse_idx_val = sparse_idx_val
        return ocp

    def setup_acados_constraints(
        self, ocp: AcadosOcp, state_cnstr: cs.MX, input_cnstr: cs.MX
    ) -> AcadosOcp:
        """Preprocess the constraints to be compatible with Acados.

        Args:
            ocp: AcadosOcp solver.
            state_cnstr: State constraints
            input_cnstr: Input constraints

        Returns:
            The acados ocp object with constraints set
        """
        initial_cnstr = cs.vertcat(state_cnstr, input_cnstr)
        cnstr = cs.vertcat(state_cnstr, input_cnstr)
        terminal_cnstr = cs.vertcat(state_cnstr)  # Terminal constraints are only state constraints
        state_tighten_param = cs.MX.sym("state_tighten_param", *state_cnstr.shape)
        input_tighten_param = cs.MX.sym("input_tighten_param", *input_cnstr.shape)
        state_input_tighten_param = cs.vertcat(state_tighten_param, input_tighten_param)

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = initial_cnstr - state_input_tighten_param
        ocp.model.con_h_expr = cnstr - state_input_tighten_param
        ocp.model.con_h_expr_e = terminal_cnstr - state_tighten_param
        ocp.dims.nh_0 = initial_cnstr.shape[0]
        ocp.dims.nh = cnstr.shape[0]
        ocp.dims.nh_e = terminal_cnstr.shape[0]

        # All constraints are defined as g(x, u) <= tol. Acados requires the constraints to be
        # defined as lb <= g(x, u) <= ub. Thus, a large negative number (-1e8) is used as the lower
        # bound to ensure that the constraints are not active. np.prod makes sure all ub and lb are
        # 1D numpy arrays
        # See: https://github.com/acados/acados/issues/650
        # See: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche
        ocp.constraints.uh_0 = -1e-8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.lh_0 = -1e8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.uh = -1e-8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.lh = -1e8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.uh_e = -1e-8 * np.ones(np.prod(terminal_cnstr.shape))
        ocp.constraints.lh_e = -1e8 * np.ones(np.prod(terminal_cnstr.shape))

        # Pass the tightening variables to the ocp object as parameters
        tighten_param = cs.vertcat(state_tighten_param, input_tighten_param)
        ocp.model.p = cs.vertcat(ocp.model.p, tighten_param) if self.sparse else tighten_param

        return ocp

    # TODO: Refactor
    def select_action(self, obs: NDArray) -> NDArray:
        """Solve the nonlinear MPC problem to get the next action."""
        assert not self._requires_recompile, "Acados model must be recompiled"
        assert self.gaussian_process is not None, "Gaussian processes are not initialized"
        # Set initial condition (0-th state)
        self.acados_solver.set(0, "lbx", obs)
        self.acados_solver.set(0, "ubx", obs)
        nu = self.model.nu

        # Set the probabilistic state and input constraint set limits
        (state_constraint_set_prev, input_constraint_set_prev) = (
            self.precompute_probabilistic_limits()
        )
        # set acados parameters
        if self.sparse:
            posterior_mean = self.posterior_mean
            sparse_idx_val = self.sparse_idx_val

            # sparse GP parameters
            assert sparse_idx_val.shape == (self.n_ind_points, 7)
            assert posterior_mean.shape == (3, self.n_ind_points)
            # casadi use column major order, while np uses row major order by default
            sparse_idx_val = sparse_idx_val.reshape(-1, 1, order="F")
            posterior_mean = posterior_mean.reshape(-1, 1, order="F")
            dyn_value = np.concatenate((sparse_idx_val, posterior_mean)).reshape(-1)
            # tighten constraints
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                # set the parameter values
                parameter_values = np.concatenate((dyn_value, tighten_value))
                # self.acados_solver.set(idx, "p", dyn_value)
                # check the shapes
                assert self.ocp.model.p.shape[0] == parameter_values.shape[0], (
                    f"parameter_values.shape: {parameter_values.shape}; model.p.shape: {self.ocp.model.p.shape}"
                )
                self.acados_solver.set(idx, "p", parameter_values)
            # tighten terminal state constraints
            tighten_value = np.concatenate(
                (state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,)))
            )
            # set the parameter values
            parameter_values = np.concatenate((dyn_value, tighten_value))
            self.acados_solver.set(self.T, "p", parameter_values)
        else:
            for idx in range(self.T):
                # tighten initial and path constraints
                state_constraint_set = state_constraint_set_prev[0][:, idx]
                input_constraint_set = input_constraint_set_prev[0][:, idx]
                tighten_value = np.concatenate((state_constraint_set, input_constraint_set))
                self.acados_solver.set(idx, "p", tighten_value)
            # tighten terminal state constraints
            tighten_value = np.concatenate(
                (state_constraint_set_prev[0][:, self.T], np.zeros((2 * nu,)))
            )
            self.acados_solver.set(self.T, "p", tighten_value)

        # Set reference for the control horizon
        goal_states = self.reference_trajectory()
        self.traj_step += 1
        y_ref = np.concatenate((goal_states[:, :-1], self.ref_action), axis=0)
        for idx in range(self.T):
            self.acados_solver.set(idx, "yref", y_ref[:, idx])
        self.acados_solver.set(self.T, "yref", goal_states[:, -1])

        # solve the optimization problem
        status = self.acados_solver.solve()
        assert status in [0, 2], f"acados returned unexpected status {status}."
        self.x_prev = self.acados_solver.get_flat("x").reshape(self.T + 1, -1).T
        self.u_prev = self.acados_solver.get_flat("u").reshape(self.T, -1).T
        return self.acados_solver.get(0, "u")

    def precompute_posterior_mean(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the posterior mean and inducing points for each GP."""
        gps = self.gaussian_process
        posterior_mean = torch.stack([torch.linalg.solve(gp.K, gp.train_targets) for gp in gps])
        train_data = torch.cat([gp.train_inputs[0] for gp in gps], dim=-1)
        return posterior_mean.numpy(force=True), train_data.numpy(force=True)

    def precompute_sparse_posterior_mean(self, n_ind_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Use the last MPC solution to precomupte values with the FITC GP approximation.

        Args:
            n_ind_points: Number of inducing points.
        """
        inputs = torch.cat([gp.train_inputs[0] for gp in self.gaussian_process], dim=-1)
        targets = torch.stack([gp.train_targets for gp in self.gaussian_process], dim=1)

        # Choose T random training set points.
        rand_idx = self.np_random.choice(range(inputs.shape[0]), size=n_ind_points, replace=False)
        sparse_inputs = inputs[rand_idx]

        posterior_means = []
        for i, (gp, idx) in enumerate(zip(self.gaussian_process, self.gp_idx)):
            K_ss = gp.covar_module(sparse_inputs[:, idx]).to_dense()
            K_xs = gp.covar_module(inputs[:, idx], sparse_inputs[:, idx]).to_dense()
            Gamma = torch.diagonal(gp.K - K_xs @ torch.linalg.solve(K_ss, K_xs.T))
            Gamma_inv = torch.diag_embed(1 / Gamma)
            Sigma_inv = K_ss + K_xs.T @ Gamma_inv @ K_xs
            posterior_mean = torch.linalg.solve(Sigma_inv, K_xs.T) @ Gamma_inv @ targets[:, i]
            posterior_means.append(posterior_mean)
        posterior_mean = torch.stack(posterior_means)
        return posterior_mean.numpy(force=True), sparse_inputs.numpy(force=True)

    # TODO: Refactor
    def create_sparse_GP_machinery(self, n_ind_points):
        """This setups the gaussian process approximations for FITC formulation."""
        idx_T, idx_R, idx_P = [0], [1, 2, 3], [4, 5, 6]
        GP_T, GP_R, GP_P = self.gaussian_process

        lengthscales_T = GP_T.covar_module.base_kernel.lengthscale.numpy(force=True)
        lengthscales_R = GP_R.covar_module.base_kernel.lengthscale.numpy(force=True)
        lengthscales_P = GP_P.covar_module.base_kernel.lengthscale.numpy(force=True)
        scale_T = GP_T.covar_module.outputscale.numpy(force=True)
        scale_R = GP_R.covar_module.outputscale.numpy(force=True)
        scale_P = GP_P.covar_module.outputscale.numpy(force=True)

        Nx = sum([gp.train_inputs[0].shape[1] for gp in self.gaussian_process])
        # Create CasADI function for computing the kernel K_zs with parameters for z, s, length
        # scales and output scaling. We need the CasADI version of this for symbolic differentiation
        # in the MPC optimization.
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
        sparse_idx = cs.SX.sym("sparse_idx", n_ind_points, Nx)
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
            ks_T[i] = covSE_T(z1_T, sparse_idx[i, idx_T], ell_s_T, sf2_s_T)
            ks_R[i] = covSE_R(z1_R, sparse_idx[i, idx_R], ell_s_R, sf2_s_R)
            ks_P[i] = covSE_P(z1_P, sparse_idx[i, idx_P], ell_s_P, sf2_s_P)
        ks_func_T = cs.Function("K_s", [z1_T, sparse_idx, ell_s_T, sf2_s_T], [ks_T])
        ks_func_R = cs.Function("K_s", [z1_R, sparse_idx, ell_s_R, sf2_s_R], [ks_R])
        ks_func_P = cs.Function("K_s", [z1_P, sparse_idx, ell_s_P, sf2_s_P], [ks_P])

        K_z_zind_T = ks_func_T(z1_T, sparse_idx, lengthscales_T, scale_T)
        K_z_zind_R = ks_func_R(z1_R, sparse_idx, lengthscales_R, scale_R)
        K_z_zind_P = ks_func_P(z1_P, sparse_idx, lengthscales_P, scale_P)
        self.K_zs_T_cs = cs.Function(
            "K_z_zind", [z1_T, sparse_idx], [K_z_zind_T], ["z1", "z2"], ["K"]
        )
        self.K_zs_R_cs = cs.Function(
            "K_z_zind", [z1_R, sparse_idx], [K_z_zind_R], ["z1", "z2"], ["K"]
        )
        self.K_zs_P_cs = cs.Function(
            "K_z_zind", [z1_P, sparse_idx], [K_z_zind_P], ["z1", "z2"], ["K"]
        )

    # TODO: Refactor
    def precompute_probabilistic_limits(self):
        """Update the constraint value limits to account for the uncertainty in the rollout."""
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        state_covariances = np.zeros((self.T + 1, nx, nx))
        input_covariances = np.zeros((self.T, nu, nu))
        # Initilize lists for the tightening of each constraint.
        state_constraint_set = [np.zeros((self.state_cnstr.num_constraints, T + 1))]
        input_constraint_set = [np.zeros((self.input_cnstr.num_constraints, T))]
        if self.x_prev is not None and self.u_prev is not None:
            cov_x = np.diag([self.initial_rollout_std**2] * nx)
            z_batch = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))  # (T, input_dim)

            # Compute the covariance of the dynamics at each time step.
            GP_T = self.gaussian_process[0]
            GP_R = self.gaussian_process[1]
            GP_P = self.gaussian_process[2]
            T_pred_point_batch = z_batch[:, [self.model.nx + self.idx["T_cmd"]]]
            T_pred_point_batch = torch.from_numpy(T_pred_point_batch).to(self.device)
            R_pred_point_batch = z_batch[
                :, [self.idx["phi"], self.idx["phi_dot"], self.model.nx + self.idx["phi_cmd"]]
            ]
            R_pred_point_batch = torch.from_numpy(R_pred_point_batch).to(self.device)
            P_pred_point_batch = z_batch[
                :,
                [
                    self.idx["theta"],
                    self.idx["theta_dot"],
                    self.model.nx + self.idx["theta_cmd"],
                ],
            ]
            P_pred_point_batch = torch.from_numpy(P_pred_point_batch).to(self.device)
            # Predict the covariance of the dynamics at each time step.
            for gp in self.gaussian_process:
                gp.eval()
            with torch.no_grad(), fast_pred_var(state=True), fast_pred_samples(state=True):
                cov = GP_T.likelihood(GP_T(T_pred_point_batch)).covariance_matrix
                cov_d_batch_T = torch.diag(cov).numpy(force=True)
                cov = GP_R.likelihood(GP_R(R_pred_point_batch)).covariance_matrix
                cov_d_batch_R = torch.diag(cov).numpy(force=True)
                cov = GP_P.likelihood(GP_P(P_pred_point_batch)).covariance_matrix
                cov_d_batch_P = torch.diag(cov).numpy(force=True)

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
                input_constraint_set[0][:, i] = (
                    -1
                    * self.inverse_cdf
                    * np.absolute(self.input_cnstr.A)
                    @ np.sqrt(np.diag(cov_u))
                )
                state_constraint_set[0][:, i] = (
                    -1
                    * self.inverse_cdf
                    * np.absolute(self.state_cnstr.A)
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
            state_constraint_set[0][:, -1] = (
                -1 * self.inverse_cdf * np.absolute(self.state_cnstr.A) @ np.sqrt(np.diag(cov_x))
            )
            state_covariances[-1] = cov_x
        return state_constraint_set, input_constraint_set

    @staticmethod
    def setup_prior_dynamics(dfdx: NDArray, dfdu: NDArray, Q: NDArray, R: NDArray, dt: float):
        """Compute the LQR gain used for propograting GP uncertainty from the prior dynamics."""
        A, B = discretize_linear_system(dfdx, dfdu, dt)
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        btp = np.dot(B.T, P)
        lqr_gain = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
        return A, B, lqr_gain

    def reference_trajectory(self) -> NDArray:
        """Construct reference states along mpc horizon.(nx, T+1)."""
        # We append the T+1 states of the trajectory to the goal_states to avoid discontinuities
        extended_traj = np.concatenate([self.traj, self.traj[:, : self.T + 1]], axis=1)
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = min(self.traj_step, extended_traj.shape[-1])
        end = min(self.traj_step + self.T + 1, extended_traj.shape[-1])
        remain = max(0, self.T + 1 - (end - start))
        tail = np.tile(extended_traj[:, end][:, None], (1, remain))
        return np.concatenate([extended_traj[:, start:end], tail], -1)
