"""General MPC utility functions."""

import casadi as cs
import numpy as np
import scipy
import scipy.linalg

from safe_control_gym.core.constraints import ConstraintList


def discretize_linear_system(A, B, dt, exact=False):
    """Discretization of a linear system

    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A (ndarray): System transition matrix.
        B (ndarray): Input matrix.
        dt (scalar): Step time interval.
        exact (bool): If to use exact discretization.

    Returns:
        Ad (ndarray): The discrete linear state matrix A.
        Bd (ndarray): The discrete linear input matrix B.
    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        return Md[:state_dim, :state_dim], Md[:state_dim, state_dim:]

    return np.eye(state_dim) + A * dt, B * dt


def get_cost_weight_matrix(weights, dim):
    """Gets weight matrix from input args.

    Args:
        weights (list): A 1D list of weights.
        dim (int): The dimension of the desired cost weight matrix.

    Returns:
        W (ndarray): The cost weight matrix.
    """
    assert len(weights) == dim or len(weights) == 1, "Wrong dimension for cost weights."
    if len(weights) == 1:
        return np.diag(weights * dim)
    return np.diag(weights)


def compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, Q_lqr, R_lqr, dt):
    """Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics.

    Args:
        dfdx (np.array): CT A matrix
        dfdu (np.array): CT B matrix
        Q, R (np.array): Gain matrices
        dt (float): Time discretization

    Retrun:
        lqr_gain (np.array): LQR optimal gain, such that (A+BK) is hurwitz
    """
    # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
    A, B = discretize_linear_system(dfdx, dfdu, dt)
    P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    lqr_gain = -np.dot(np.linalg.inv(R_lqr + np.dot(btp, B)), np.dot(btp, A))

    return lqr_gain, A, B, P


def rk_discrete(f, n, m, dt):
    """Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym("X", n)
    U = cs.SX.sym("U", m)
    # Runge-Kutta 4 integration
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    rk_dyn = cs.Function("rk_f", [X, U], [x_next], ["x0", "p"], ["xf"])

    return rk_dyn


def compute_state_rmse(state_error):
    """Compute root-mean-square error."""
    mse = np.mean(state_error**2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))
    return state_rmse, state_rmse_scalar


def reset_constraints(constraints):
    """Setup the constraints list.

    Args:
        constraints (list): List of constraints controller is subject too.
    """
    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError("[Error] Cannot handle combined state input constraints yet.")
    return constraints_list, state_constraints_sym, input_constraints_sym


def set_acados_constraint_bound(
    constraint,
    bound_type,
    bound_value=None,
):
    """Set the acados constraint bound.
    Args:
        constraint (casadi expression): Constraint expression.
        bound_type (str): Type of bound (lb, ub).
        bound_value (float): Value of the bound.

    Returns:
        bound (np.array): Constraint bound value.

    Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650
    """
    if bound_value is None:
        assert bound_type in ("lb", "ub"), f"bound_type must be 'lb' or 'ub', got {bound_type}"
        bound_value = -1e8 if bound_type == "lb" else 1e-6
    return bound_value * np.ones(constraint.shape)
