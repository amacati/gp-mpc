"""General MPC utility functions."""

import numpy as np
import scipy
import scipy.linalg


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
