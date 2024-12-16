'''General MPC utility functions.'''

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
from termcolor import colored

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.envs.constraints import ConstraintList


def get_cost_weight_matrix(weights,
                           dim
                           ):
    '''Gets weight matrix from input args.'''
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception('Wrong dimension for cost weights.')
    return W


def compute_discrete_lqr_gain_from_cont_linear_system(dfdx, dfdu, Q_lqr, R_lqr, dt):
    '''Computes the LQR gain used for propograting GP uncertainty from the prior model dynamics.

    Args:
        dfdx (np.array): CT A matrix
        dfdu (np.array): CT B matrix
        Q, R (np.array): Gain matrices
        dt (float): Time discretization

    Retrun:
        lqr_gain (np.array): LQR optimal gain, such that (A+BK) is hurwitz
    '''
    # Determine the LQR gain K to propogate the input uncertainty (doing this at each timestep will increase complexity).
    A, B = discretize_linear_system(dfdx, dfdu, dt)
    P = scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    lqr_gain = -np.dot(np.linalg.inv(R_lqr + np.dot(btp, B)), np.dot(btp, A))

    return lqr_gain, A, B, P


def rk_discrete(f, n, m, dt):
    '''Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    '''
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    # Runge-Kutta 4 integration
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    rk_dyn = cs.Function('rk_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return rk_dyn


def compute_state_rmse(state_error):
    '''Compute root-mean-square error.'''
    mse = np.mean(state_error ** 2, axis=0)
    state_rmse = np.sqrt(mse)
    state_rmse_scalar = np.sqrt(np.sum(mse))

    return state_rmse, state_rmse_scalar


def reset_constraints(constraints):
    '''Setup the constraints list.

    Args:
        constraints (list): List of constraints controller is subject too.
    '''

    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError('[Error] Cannot handle combined state input constraints yet.')
    return constraints_list, state_constraints_sym, input_constraints_sym


def set_acados_constraint_bound(constraint,
                                bound_type,
                                bound_value=None,
                                ):
    '''Set the acados constraint bound.

    Note:
        all constraints in safe-control-gym are defined as g(x, u) <= constraint_tol
        However, acados requires the constraints to be defined as lb <= g(x, u) <= ub
        Thus, a large negative number (-1e8) is used as the lower bound.
        See: https://github.com/acados/acados/issues/650
    '''
    if bound_value is None:
        if bound_type == 'lb':
            bound_value = -1e8
        elif bound_type == 'ub':
            bound_value = 1e-6

    return bound_value * np.ones(constraint.shape)


def plot_open_loop_sol(ctrl_list, 
                       save_path=None, 
                       label=None, 
                       title=None,
                       state_label=None,
                       action_label=None):
    ''' Plot the open loop predction of the MPC controller.

    Args:
        ctrl (MPC): MPC controller object.
    '''
    # Plot the open loop prediction
    nx = ctrl_list[0].x_prev.shape[0]  # state dim
    nu = ctrl_list[0].u_prev.shape[0]  # input dim
    fig, axs = plt.subplots(nx + nu, 1, figsize=(5, 8))
    for ctrl_idx, ctrl in enumerate(ctrl_list):  
        if ctrl.x_prev is not None and ctrl.u_prev is not None:
            steps = ctrl.T  # prediction horizon
            dt = ctrl.dt  # ctrl frequency
            x = ctrl.x_prev  # open loop state (nx, steps + 1)
            u = ctrl.u_prev  # open loop input (nu, steps)
            # get the reference trajectory
            goal_states = ctrl.get_references()
            for i in range(nx):
                sol_label = f'{ctrl_idx} pred' if label is None else f'{label[ctrl_idx]} pred'
                ref_label = f'{ctrl_idx} ref' if label is None else f'{label[ctrl_idx]} ref'
                axs[i].plot(np.arange(steps + 1) * dt, x[i, :], label=sol_label)
                axs[i].plot(np.arange(steps + 1) * dt, goal_states[i, :], 'k--', label=ref_label)
                y_label = state_label[i] if state_label is not None else f'$x_{i}$'
                axs[i].set_ylabel(y_label)
            # put the legend to the right of the plot
            # axs[i].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -4.0))
            for i in range(nu):
                sol_label = f'{ctrl_idx} pred' if label is None else f'{label[ctrl_idx]} pred'
                axs[nx + i].plot(np.arange(steps) * dt, u[i, :], label=sol_label)
                y_label = action_label[i] if action_label is not None else f'$u_{i}$'
                axs[nx + i].set_ylabel(y_label)
            axs[nx+i].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -1.6))
        else:
            print(colored('[Warning] No open loop solution to plot.', 'yellow'))
        plt.xlabel('Time [s]')
        fig.suptitle(f'Open Loop Prediction {title}')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
