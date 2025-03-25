"""Model Predictive Control using Acados."""

from pathlib import Path

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from numpy.typing import NDArray


class MPC:
    """MPC with full nonlinear model."""

    U_EQ: NDArray = np.array([0.3234, 0, 0, 0])

    def __init__(
        self,
        symbolic_model,
        traj: NDArray,
        q_mpc: list,
        r_mpc: list,
        output_dir: Path,
        horizon: int = 5,
    ):
        """Creates task and controller.

        Args:
            symbolic_model: Symbolic casadi model for the environment dynamics.
            traj: Reference trajectory.
            q_mpc: diagonals of state cost weight.
            r_mpc: diagonals of input/action cost weight.
            output_dir: output directory to write logs and results.
            horizon: mpc planning horizon.
        """
        # Model parameters
        self.model = symbolic_model
        self.T = horizon
        self.traj = traj
        self.traj_step = 0
        self.u_ref = np.repeat(self.U_EQ[..., None], self.T, axis=-1)
        assert len(q_mpc) == self.model.nx
        assert len(r_mpc) == self.model.nu
        Q = np.diag(q_mpc)
        R = np.diag(r_mpc)

        self.output_dir = output_dir
        acados_model = self.setup_acados_model()
        ocp = self.setup_acados_optimizer(acados_model, Q, R)
        s_low = np.array([-2, -15, -2, -15, -0.05, -15, -1.5, -1.5, -10, -8.5, -8.5, -10])
        s_high = np.array([2, 15, 2, 15, 2, 15, 1.5, 1.5, 10, 8.5, 8.5, 10])
        state_cnstr = self.setup_constraints(ocp.model.x, s_low, s_high)
        u_low = np.array([0.12, -0.43, -0.43, -0.43])
        u_high = np.array([0.59, 0.43, 0.43, 0.43])
        input_cnstr = self.setup_constraints(ocp.model.u, u_low, u_high)
        ocp = self.setup_acados_constraints(ocp, state_cnstr, input_cnstr)
        json_file = output_dir / "acados_ocp.json"
        self.acados_solver = AcadosOcpSolver(ocp, json_file=str(json_file), verbose=False)

    def reset(self):
        """Prepares for training or evaluation."""
        self.acados_solver.reset()
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
        k2 = fc_func(acados_model.x + self.model.dt / 2 * k1, acados_model.u)
        k3 = fc_func(acados_model.x + self.model.dt / 2 * k2, acados_model.u)
        k4 = fc_func(acados_model.x + self.model.dt * k3, acados_model.u)
        f_disc = acados_model.x + self.model.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acados_model.disc_dyn_expr = f_disc

        # store meta information
        acados_model.name = "mpc"
        acados_model.t_label = "time"

        return acados_model

    def setup_acados_optimizer(self, model: AcadosModel, Q: NDArray, R: NDArray) -> AcadosOcp:
        """Set up the nonlinear optimization problem in Acados."""
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
        # Placeholder y_ref, y_ref_e and initial state constraint. We update yref in select_action.
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.constraints.x0 = np.zeros(nx)

        # Set up solver options
        ocp.solver_options.N_horizon = self.T  # prediction horizon
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 25
        ocp.solver_options.tf = self.T * self.model.dt  # prediction duration
        ocp.code_export_directory = str(self.output_dir / "mpc_c_generated_code")

        return ocp

    @staticmethod
    def setup_acados_constraints(
        ocp: AcadosOcp, state_cnstr: cs.MX, input_cnstr: cs.MX, tol: float = 1e-8
    ) -> AcadosOcp:
        """Preprocess the constraints to be compatible with Acados.

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
        # All constraints are defined as g(x, u) <= tol. Acados requires the constraints to be
        # defined as lb <= g(x, u) <= ub. Thus, a large negative number (-1e8) is used as the lower
        # bound to ensure that the constraints are not active. np.prod makes sure all ub and lb are
        # 1D numpy arrays
        # See: https://github.com/acados/acados/issues/650
        # See: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche
        ocp.constraints.uh_0 = tol * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.lh_0 = -1e8 * np.ones(np.prod(initial_cnstr.shape))
        ocp.constraints.uh = tol * np.ones(np.prod(cnstr.shape))
        ocp.constraints.lh = -1e8 * np.ones(np.prod(cnstr.shape))
        ocp.constraints.uh_e = tol * np.ones(np.prod(terminal_cnstr.shape))
        ocp.constraints.lh_e = -1e8 * np.ones(np.prod(terminal_cnstr.shape))
        return ocp

    @staticmethod
    def setup_constraints(sym: cs.MX, low: NDArray, high: NDArray) -> cs.MX:
        dim = low.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-low, high))
        return A @ sym - b

    def select_action(self, obs: NDArray) -> NDArray:
        # Set initial condition (0-th state)
        self.acados_solver.set(0, "lbx", obs)
        self.acados_solver.set(0, "ubx", obs)
        goal_states = self.reference_trajectory()
        self.traj_step += 1

        y_ref = np.concatenate((goal_states[:, :-1], self.u_ref), axis=0)
        for idx in range(self.T):
            self.acados_solver.set(idx, "yref", y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_solver.set(self.T, "yref", y_ref_e)
        status = self.acados_solver.solve()
        assert status in [0, 2], f"acados returned unexpected status {status}."
        return self.acados_solver.get(0, "u")

    def reference_trajectory(self) -> NDArray:
        """Construct reference states along mpc horizon."""
        # We wrap around the trajectory if it is not long enough. Note that this assumes that the
        # trajectory is periodic!
        indices = np.arange(self.traj_step, self.traj_step + self.T + 1) % self.traj.shape[-1]
        return self.traj[:, indices]  # (nx, T+1)
