"""Model Predictive Control using Acados."""

from pathlib import Path

import casadi as cs
import numpy as np
import scipy
import torch
from numpy.typing import NDArray

from safe_control_gym.mpc.mpc_utils import reset_constraints

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    raise ImportError("Acados is not installed") from e


class MPC:
    """MPC with full nonlinear model."""

    state_labels = ["x", "d_x", "y", "d_y", "z", "d_z", "phi", "theta", "d_phi", "d_theta"]
    action_labels = ["T_c", "R_c", "P_c"]

    def __init__(
        self,
        env_func,
        q_mpc: list,
        r_mpc: list,
        output_dir: Path,
        horizon: int = 5,
        device: str = "cpu",
        seed: int = 0,
        prior_info: dict | None = None,
    ):
        """Creates task and controller.

        Args:
            env_func: function to instantiate task/environment.
            q_mpc: diagonals of state cost weight.
            r_mpc: diagonals of input/action cost weight.
            output_dir: output directory to write logs and results.
            horizon: mpc planning horizon.
            device: torch device.
            seed: random seed.
            prior_info: prior model information.
        """
        self.output_dir = output_dir
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        self.device = device
        self.seed = seed

        # Task.
        env = env_func()
        _, state_cnstrs, input_cnstrs = reset_constraints(env.constraints.constraints)
        assert len(state_cnstrs) == 1 and len(input_cnstrs) == 1

        # Model parameters
        env._setup_symbolic(prior_prop=prior_info.get("prior_prop", {}))
        self.model = env.symbolic
        self.t_symbolic_fn = env.T_mapping_func  # Required for GP_MPC. TODO: Remove
        self.dt = self.model.dt
        self.T = horizon
        self.traj = env.X_GOAL.T
        self.traj_step = 0
        self.u_ref = np.repeat(env.U_GOAL[:, None], self.T, axis=-1)
        assert len(q_mpc) == self.model.nx
        assert len(r_mpc) == self.model.nu
        Q = np.diag(q_mpc)
        R = np.diag(r_mpc)

        acados_model = self.setup_acados_model()
        ocp = self.setup_acados_optimizer(acados_model, Q, R)
        state_cnstr, input_cnstr = (state_cnstrs[0](ocp.model.x), input_cnstrs[0](ocp.model.u))
        ocp = self.setup_acados_constraints(ocp, state_cnstr, input_cnstr)
        json_file = self.output_dir / "acados_ocp.json"
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=str(json_file), verbose=False)
        env.close()

    def reset(self):
        """Prepares for training or evaluation."""
        self.acados_ocp_solver.reset()
        self.traj_step = 0

    def setup_acados_model(self) -> AcadosModel:
        """Set up the symbolic model for acados.

        Returns:
            acados_model: Acados model object.
        """
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym

        # set up rk4 (acados need symbolic expression of dynamics, not function)
        fc_func = self.model.fc_func  # continuous-time dynamics
        k1 = fc_func(acados_model.x, acados_model.u)
        k2 = fc_func(acados_model.x + self.dt / 2 * k1, acados_model.u)
        k3 = fc_func(acados_model.x + self.dt / 2 * k2, acados_model.u)
        k4 = fc_func(acados_model.x + self.dt * k3, acados_model.u)
        f_disc = acados_model.x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acados_model.disc_dyn_expr = f_disc

        # store meta information
        acados_model.name = "mpc"
        acados_model.x_labels = self.state_labels
        acados_model.u_labels = self.action_labels
        acados_model.t_label = "time"

        return acados_model

    def setup_acados_optimizer(self, model: AcadosModel, Q: NDArray, R: NDArray) -> AcadosOcp:
        """Sets up nonlinear optimization problem."""
        # Create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = model
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # Configure costs
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx : (nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # Placeholder y_ref and y_ref_e. We update these in select_action.
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # Placeholder initial state constraint
        ocp.constraints.x0 = np.zeros(nx)

        # set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 25
        ocp.solver_options.tf = self.T * self.dt  # prediction duration
        ocp.code_export_directory = str(self.output_dir / "mpc_c_generated_code")

        return ocp

    @staticmethod
    def setup_acados_constraints(
        ocp: AcadosOcp, state_cnstr: cs.MX, input_cnstr: cs.MX, tol: float = 1e-8
    ) -> AcadosOcp:
        """Preprocess the constraints to be compatible with acados.

        Args:
            ocp: Acados ocp object
            state_cnstr: State constraints
            input_cnstr: Input constraints
            tol: Tolerance for constraints

        Returns:
            The acados ocp object with constraints set.
        """
        # Combine state and input constraints for non-terminal steps
        initial_cnstr = cs.vertcat(state_cnstr, input_cnstr)
        cnstr = cs.vertcat(state_cnstr, input_cnstr)
        terminal_cnstr = cs.vertcat(state_cnstr)  # Terminal constraints are only state constraints

        ocp.model.con_h_expr_0 = initial_cnstr
        ocp.model.con_h_expr = cnstr
        ocp.model.con_h_expr_e = terminal_cnstr
        ocp.dims.nh_0 = initial_cnstr.shape[0]
        ocp.dims.nh = cnstr.shape[0]
        ocp.dims.nh_e = terminal_cnstr.shape[0]
        # assign constraints upper and lower bounds. lower bounds are set to -1e8 to ensure that the
        # constraints are not active. np.prod makes sure all ub and lb are 1D numpy arrays
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        ocp.constraints.uh_0 = tol * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.lh_0 = -1e8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.uh = tol * np.ones(np.prod(cnstr.shape))
        ocp.constraints.lh = -1e8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.uh_e = tol * np.ones(np.prod(terminal_cnstr.shape))
        ocp.constraints.lh_e = -1e8 * np.ones(np.prod(terminal_cnstr.shape))
        return ocp

    def select_action(self, obs: NDArray) -> NDArray:
        """Solves the nonlinear mpc problem to get next action."""
        # Set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)
        # Skip warmstart for simplicity. Acados has built-in warmstart (?)
        goal_states = self.reference_trajectory()
        self.traj_step += 1

        # TODO: Can this be replaced with a single set call?
        y_ref = np.concatenate((goal_states[:, :-1], self.u_ref), axis=0)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, "yref", y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)
        # Solve the optimization problem
        self.acados_ocp_solver.solve()
        return self.acados_ocp_solver.get(0, "u")

    def reference_trajectory(self):
        """Construct reference states along mpc horizon.(nx, T+1)."""
        # We append the T+1 states of the trajectory to the goal_states such that the vel states
        # won't drop at the end of an episode
        extended_traj = np.concatenate([self.traj, self.traj[:, : self.T + 1]], axis=1)
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = min(self.traj_step, extended_traj.shape[-1])
        end = min(self.traj_step + self.T + 1, extended_traj.shape[-1])
        remain = max(0, self.T + 1 - (end - start))
        tail = np.tile(extended_traj[:, end][:, None], (1, remain))
        goal_states = np.concatenate([extended_traj[:, start:end], tail], -1)
        return goal_states
