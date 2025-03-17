"""Model Predictive Control using Acados."""

import shutil
import time
from pathlib import Path
from typing import Any

import casadi as cs
import numpy as np
import scipy
import torch

from safe_control_gym.mpc.mpc_utils import (
    discretize_linear_system,
    get_cost_weight_matrix,
    reset_constraints,
    rk_discrete,
    set_acados_constraint_bound,
)

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as e:
    raise ImportError("Acados is not installed") from e


class MPC_ACADOS:
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
        constraint_tol: float = 1e-6,
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
            constraint_tol: Tolerance to add the the constraint as sometimes solvers are not exact.
            device: torch device.
            seed: random seed.
            prior_info: prior model information.
        """
        self.output_dir = output_dir
        if "cuda" in device and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        self.device = device
        self.seed = seed
        self.prior_info = prior_info
        self.q_mpc = q_mpc
        self.r_mpc = r_mpc

        # Task.
        env = env_func()
        (
            self.constraints,
            self.state_constraints_sym,
            self.input_constraints_sym,
        ) = reset_constraints(env.constraints.constraints)
        # Model parameters
        self.model = self.get_prior(env)
        self.t_symbolic_fn = env.T_mapping_func  # Required for GP_MPC. TODO: Remove
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)
        self.traj = env.X_GOAL.T
        self.traj_step = 0
        self.u_goal = env.U_GOAL.reshape(-1, 1)

        self.constraint_tol = constraint_tol
        self.results_dict = {}  # Required to remain compatible with base_experiment

        # Compile the acados model
        # delete the generated c code directory
        generated_code_path = self.output_dir / "mpc_c_generated_code"
        if generated_code_path.exists():
            shutil.rmtree(generated_code_path)
            assert not generated_code_path.exists(), "Failed to delete the c code directory"

        self.acados_model = self.setup_acados_model()
        self.ocp = self.setup_acados_optimizer()
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=str(self.output_dir / "acados_ocp.json"), verbose=False
        )
        env.close()

    def reset(self):
        """Prepares for training or evaluation."""
        self.acados_ocp_solver.reset()
        self.traj_step = 0
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

    def setup_acados_model(self) -> AcadosModel:
        """Set up the symbolic model for acados.

        Returns:
            acados_model: Acados model object.
        """
        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym

        # continuous-time dynamics
        fc_func = self.model.fc_func
        # set up rk4 (acados need symbolic expression of dynamics, not function)
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

    def setup_acados_optimizer(self) -> AcadosOcp:
        """Sets up nonlinear optimization problem."""
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set cost (NOTE: safe-control-gym uses quadratic cost)
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
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
        for state_constraint in self.state_constraints_sym:
            state_constraint_expr_list.append(state_constraint(ocp.model.x))
        for input_constraint in self.input_constraints_sym:
            input_constraint_expr_list.append(input_constraint(ocp.model.u))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(
            *state_constraint_expr_list
        )  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr)

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

        ocp.solver_options.tf = self.T * self.dt  # prediction horizon

        # c code generation
        # NOTE: when using GP-MPC, a separated directory is needed;
        # otherwise, Acados solver can read the wrong c code
        ocp.code_export_directory = str(self.output_dir / "mpc_c_generated_code")
        return ocp

    def processing_acados_constraints_expression(
        self,
        ocp: AcadosOcp,
        h0_expr: cs.MX,
        h_expr: cs.MX,
        he_expr: cs.MX,
    ) -> AcadosOcp:
        """Preprocess the constraints to be compatible with acados.

        Args:
            ocp (AcadosOcp): acados ocp object
            h0_expr (cs.MX expression): initial state constraints
            h_expr (cs.MX expression): state and input constraints
            he_expr (cs.MX expression): terminal state constraints

        Returns:
            ocp (AcadosOcp): acados ocp object with constraints set.
        """

        ub = {
            "h": set_acados_constraint_bound(h_expr, "ub", self.constraint_tol),
            "h0": set_acados_constraint_bound(h0_expr, "ub", self.constraint_tol),
            "he": set_acados_constraint_bound(he_expr, "ub", self.constraint_tol),
        }

        lb = {
            "h": set_acados_constraint_bound(h_expr, "lb"),
            "h0": set_acados_constraint_bound(h0_expr, "lb"),
            "he": set_acados_constraint_bound(he_expr, "lb"),
        }

        # make sure all the ub and lb are 1D numpy arrays
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
        ocp.model.con_h_expr_0, ocp.model.con_h_expr, ocp.model.con_h_expr_e = (
            h0_expr,
            h_expr,
            he_expr,
        )
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

    def select_action(self, obs, info=None):
        """Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        """
        nx, nu = self.model.nx, self.model.nu
        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, "lbx", obs)
        self.acados_ocp_solver.set(0, "ubx", obs)
        # Skip warmstart for simplicity. Acados has built-in warmstart (?)
        goal_states = self.get_references()
        self.traj_step += 1

        y_ref = np.concatenate(
            (goal_states[:, :-1], np.repeat(self.u_goal, self.T, axis=1)),
            axis=0,
        )
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, "yref", y_ref[:, idx])
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, "yref", y_ref_e)

        # solve the optimization problem

        try:
            status = self.acados_ocp_solver.solve()

            # get the open-loop solution
            if self.x_prev is None and self.u_prev is None:
                self.x_prev = np.zeros((nx, self.T + 1))
                self.u_prev = np.zeros((nu, self.T))
            for i in range(self.T + 1):
                self.x_prev[:, i] = self.acados_ocp_solver.get(i, "x")
            for i in range(self.T):
                self.u_prev[:, i] = self.acados_ocp_solver.get(i, "u")
        except Exception as e:
            self.acados_ocp_solver.print_statistics()
            status = self.acados_ocp_solver.get_stats("status")
            raise RuntimeError(f"acados returned status {status}. ") from e
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

    def dynamics_fn(self) -> tuple[cs.Function, cs.Function, np.ndarray, np.ndarray]:
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
        linear_dynamics_fn = cs.Function(
            "linear_discrete_dynamics",
            [delta_x, delta_u],
            [x_dot_lin],
            ["x0", "p"],
            ["xf"],
        )
        # nonlinear dynamics
        dynamics_fn = rk_discrete(self.model.fc_func, self.model.nx, self.model.nu, self.dt)
        return dynamics_fn, linear_dynamics_fn, dfdx, dfdu

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

    def get_prior(self, env):
        """Fetch the prior model from the env for the controller."""
        prior_prop = self.prior_info.get("prior_prop", {})
        env._setup_symbolic(prior_prop=prior_prop)
        return env.symbolic
