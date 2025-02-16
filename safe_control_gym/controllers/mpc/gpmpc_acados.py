

import csv
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial

import casadi as cs
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from skopt.sampler import Lhs
from termcolor import colored

from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.controllers.mpc.gp_utils import (GaussianProcessCollection, ZeroMeanIndependentGPModel,
                                                       covSEard, kmeans_centriods)
from safe_control_gym.controllers.mpc.linear_mpc import MPC, LinearMPC
from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC
from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.utils import timing

class GPMPC_ACADOS(GPMPC):
    '''Implements a GP-MPC controller with Acados optimization.'''

    def __init__(
            self,
            env_func,
            seed: int = 1337,
            horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            constraint_tol: float = 1e-8,
            additional_constraints: list = None,
            warmstart: bool = True,
            train_iterations: int = None,
            test_data_ratio: float = 0.2,
            overwrite_saved_data: bool = True,
            optimization_iterations: list = None,
            learning_rate: list = None,
            normalize_training_data: bool = False,
            use_gpu: bool = False,
            gp_model_path: str = None,
            n_ind_points: int = 30,
            inducing_point_selection_method='kmeans',
            recalc_inducing_points_at_every_step=False,
            prob: float = 0.955,
            initial_rollout_std: float = 0.005,
            input_mask: list = None,
            target_mask: list = None,
            gp_approx: str = 'mean_eq',
            online_learning: bool = False,
            prior_info: dict = None,
            sparse_gp: bool = False,
            # inertial_prop: list = [1.0],
            prior_param_coeff: float = 1.0,
            terminate_run_on_done: bool = True,
            output_dir: str = 'results/temp',
            **kwargs
    ):
        super().__init__(
            env_func=env_func,
            seed=seed,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            constraint_tol=constraint_tol,
            additional_constraints=additional_constraints,
            warmstart=warmstart,
            train_iterations=train_iterations,
            test_data_ratio=test_data_ratio,
            overwrite_saved_data=overwrite_saved_data,
            optimization_iterations=optimization_iterations,
            learning_rate=learning_rate,
            normalize_training_data=normalize_training_data,
            use_gpu=use_gpu,
            gp_model_path=gp_model_path,
            prob=prob,
            initial_rollout_std=initial_rollout_std,
            input_mask=input_mask,
            target_mask=target_mask,
            gp_approx=gp_approx,
            sparse_gp=sparse_gp,
            n_ind_points=n_ind_points,
            inducing_point_selection_method=inducing_point_selection_method,
            recalc_inducing_points_at_every_step=recalc_inducing_points_at_every_step,
            online_learning=online_learning,
            prior_info=prior_info,
            prior_param_coeff=prior_param_coeff,
            terminate_run_on_done=terminate_run_on_done,
            output_dir=output_dir,
            **kwargs)

        if hasattr(self, 'prior_ctrl'):
            self.prior_ctrl.close()
        
        self.prior_ctrl = MPC_ACADOS(
            env_func=self.prior_env_func,
            horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=warmstart,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            use_gpu=use_gpu,
            seed=seed,
            prior_info=prior_info,
        )
        self.prior_ctrl.reset()
        print('prior_ctrl:', type(self.prior_ctrl))
        self.prior_dynamics_func = self.prior_ctrl.dynamics_func
        self.prior_dynamcis_func_c = self.prior_ctrl.model.fc_func

        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        print('prior_info[prior_prop]', prior_info['prior_prop'])

    def setup_acados_model(self, n_ind_points) -> AcadosModel:

        # setup GP related
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1 / self.model.nx - (self.prob + 1) / (2 * self.model.nx)))
        if self.sparse_gp:
            self.create_sparse_GP_machinery(n_ind_points)

        # setup acados model
        model_name = self.env.NAME

        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = model_name


        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point
        z = z[self.input_mask]

        if self.sparse_gp:
            # sparse GP inducing points
            '''
            z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, len(self.input_mask))
            mean_post_factor should be of shape (len(self.target_mask), n_ind_points)
            Here we create the corresponding parameters since acados supports only 1D parameters
            '''
            z_ind = cs.MX.sym('z_ind', n_ind_points, len(self.input_mask))
            mean_post_factor = cs.MX.sym('mean_post_factor', len(self.target_mask), n_ind_points)
            acados_model.p = cs.vertcat(cs.reshape(z_ind, -1, 1), cs.reshape(mean_post_factor, -1, 1))
            # define the dynamics
            f_disc = self.prior_dynamics_func(x0=acados_model.x, p=acados_model.u)['xf'] \
                + self.Bd @ cs.sum2(self.K_z_zind_func(z1=z, z2=z_ind)['K'] * mean_post_factor)

            self.sparse_gp_func = cs.Function('sparse_func',
                                              [acados_model.x, acados_model.u, z_ind, mean_post_factor], [f_disc])
        else:
            f_disc = self.prior_dynamics_func(x0=acados_model.x, p=acados_model.u)['xf'] \
                + self.Bd @ self.gaussian_process.casadi_predict(z=z)['mean']

        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    def setup_acados_optimizer(self, n_ind_points):
        print('=================Setting up acados optimizer=================')
        # before_optimizer_setup = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        ocp.model = self.acados_model

        # set dimensions
        ocp.dims.N = self.T  # prediction horizon

        # set cost
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        # cost weight matrices
        ocp.cost.W = scipy.linalg.block_diag(self.Q, self.R)
        ocp.cost.W_e = self.P if hasattr(self, 'P') else self.Q
        # ocp.cost.W_e = self.Q

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        # placeholder y_ref and y_ref_e (will be set in select_action)
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Constraints
        # general constraint expressions
        state_constraint_expr_list = []
        input_constraint_expr_list = []

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        ocp = self.processing_acados_constraints_expression(ocp, h0_expr, h_expr, he_expr)
        ocp.parameter_values = np.zeros((ocp.model.p.shape[0], ))  # dummy values


        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP' 
        ocp.solver_options.nlp_solver_max_iter = 10 
        ocp.solver_options.qp_solver_iter_max = 10
        ocp.solver_options.qp_tol = 1e-4
        ocp.solver_options.tol = 1e-4

        # ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH' if not self.use_RTI else 'MERIT_BACKTRACKING'
        # ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        # c code generation
        # NOTE: when using GP-MPC, a separated directory is needed;
        # otherwise, Acados solver can read the wrong c code
        ocp.code_export_directory = self.output_dir + '/gpmpc_c_generated_code'

        self.ocp = ocp
        self.opti_dict = {'n_ind_points': n_ind_points}
        # compute sparse GP values
        # the actual values will be set in select_action_with_gp
        if self.sparse_gp:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val
        else:
            mean_post_factor_val, z_ind_val = self.precompute_mean_post_factor_all_data()
            self.mean_post_factor_val = mean_post_factor_val
            self.z_ind_val = z_ind_val

    def processing_acados_constraints_expression(self, ocp: AcadosOcp, h0_expr, h_expr, he_expr,
                                                 state_tighten_list=None, input_tighten_list=None) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
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
        '''
        # NOTE: only the upper bound is tightened due to constraint are defined in the
        # form of g(x, u) <= constraint_tol in safe-control-gym

        # lambda functions to set the upper and lower bounds of the chance constraints
        def constraint_ub_chance(constraint): return -self.constraint_tol * np.ones(constraint.shape)
        def constraint_lb_chance(constraint): return -1e8 * np.ones(constraint.shape)
        if state_tighten_list is not None and input_tighten_list is not None:
            state_tighten_var = cs.vertcat(*state_tighten_list)
            input_tighten_var = cs.vertcat(*input_tighten_list)
            ub = {'h': constraint_ub_chance(h_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
                'h0': constraint_ub_chance(h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)),
                'he': constraint_ub_chance(he_expr - state_tighten_var)}
        else:
            ub = {'h': constraint_ub_chance(h_expr),
              'h0': constraint_ub_chance(h0_expr),
              'he': constraint_ub_chance(he_expr)}
        lb = {'h': constraint_lb_chance(h_expr),
              'h0': constraint_lb_chance(h0_expr),
              'he': constraint_lb_chance(he_expr)}

        # make sure all the ub and lb are 1D casaadi SX variables
        # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
        for key in ub.keys():
            ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
            lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
        # check ub and lb dimensions
        for key in ub.keys():
            assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
            assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
        assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'

        # pass the constraints to the ocp object
        ocp.model.con_h_expr_0 = h0_expr
        ocp.model.con_h_expr = h_expr
        ocp.model.con_h_expr_e = he_expr
        if state_tighten_list is not None and input_tighten_list is not None:
            ocp.model.con_h_expr_0 = h0_expr - cs.vertcat(state_tighten_var, input_tighten_var)
            ocp.model.con_h_expr = h_expr - cs.vertcat(state_tighten_var, input_tighten_var)
            ocp.model.con_h_expr_e = he_expr - state_tighten_var

        ocp.dims.nh_0, ocp.dims.nh, ocp.dims.nh_e = \
            h0_expr.shape[0], h_expr.shape[0], he_expr.shape[0]
        # assign constraints upper and lower bounds
        ocp.constraints.uh_0 = ub['h0']
        ocp.constraints.lh_0 = lb['h0']
        ocp.constraints.uh = ub['h']
        ocp.constraints.lh = lb['h']
        ocp.constraints.uh_e = ub['he']
        ocp.constraints.lh_e = lb['he']

        return ocp

    def select_action(self, obs, info=None):
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.results_dict['runtime'].append(time_after - time_before)
        print('current obs:', obs)
        print('current action:', action)
        self.last_obs = obs
        self.last_action = action

        return action

    # @timing
    def select_action_with_gp(self, obs):
        time_before = time.time()
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx
        # TODO: replace this with something safer
        n_ind_points = self.opti_dict['n_ind_points']

        # set initial condition (0-th state)
        self.acados_ocp_solver.set(0, 'lbx', obs)
        self.acados_ocp_solver.set(0, 'ubx', obs)

        # compute the sparse GP values
        if self.recalc_inducing_points_at_every_step:
            mean_post_factor_val, _, _, z_ind_val = self.precompute_sparse_gp_values(n_ind_points)
            self.results_dict['inducing_points'].append(z_ind_val)
        else:
            # use the precomputed values
            mean_post_factor_val = self.mean_post_factor_val
            z_ind_val = self.z_ind_val
            self.results_dict['inducing_points'] = [z_ind_val]
        # set acados parameters
        if self.sparse_gp:
            # sparse GP parameters
            assert z_ind_val.shape == (n_ind_points, len(self.input_mask))
            assert mean_post_factor_val.shape == (len(self.target_mask), n_ind_points)
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order='F')
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order='F')
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)

            for idx in range(self.T):
                # set the parameter values
                self.acados_ocp_solver.set(idx, "p", dyn_value)
            # set the parameter values
            self.acados_ocp_solver.set(self.T, "p", dyn_value)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        for idx in range(self.T):
            y_ref = np.concatenate((goal_states[:, idx], np.zeros((nu,))))
            self.acados_ocp_solver.set(idx, 'yref', y_ref)
        y_ref_e = goal_states[:, -1]
        self.acados_ocp_solver.set(self.T, 'yref', y_ref_e)

        # solve the optimization problem
        status = self.acados_ocp_solver.solve()
        if status not in [0, 2]:
            self.acados_ocp_solver.print_statistics()
            print(colored(f'acados returned status {status}. ', 'red'))

        action = self.acados_ocp_solver.get(0, 'u')
        # get the open-loop solution
        if self.x_prev is None and self.u_prev is None:
            self.x_prev = np.zeros((nx, self.T + 1))
            self.u_prev = np.zeros((nu, self.T))
        if self.u_prev is not None and nu == 1:
            self.u_prev = self.u_prev.reshape((1, -1))

        for i in range(self.T + 1):
            self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
        for i in range(self.T):
            self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
        if nu == 1:
            self.u_prev = self.u_prev.flatten()
        self.x_guess = self.x_prev
        self.u_guess = self.u_prev

        time_after = time.time()
        print(f'gpmpc acados sol time: {time_after - time_before:.3f}; sol status {status}; nlp iter {self.acados_ocp_solver.get_stats("sqp_iter")}; qp iter {self.acados_ocp_solver.get_stats("qp_iter")}')
        self.results_dict['inference_time'].append(self.acados_ocp_solver.get_stats("time_tot"))
        
        return action

    @timing
    def reset(self):
        '''Reset the controller before running.'''
        # Setup reference input.
        if self.env.TASK == Task.STABILIZATION:
            self.mode = 'stabilization'
            self.x_goal = self.env.X_GOAL
        elif self.env.TASK == Task.TRAJ_TRACKING:
            self.mode = 'tracking'
            self.traj = self.env.X_GOAL.T
            self.traj_step = 0
        # Dynamics model.
        self.setup_prior_dynamics()
        if self.gaussian_process is not None:
            # self.compute_terminal_cost_and_ancillary_gain()
            # sparse GP
            if self.sparse_gp and self.train_data['train_targets'].shape[0] <= self.n_ind_points:
                n_ind_points = self.train_data['train_targets'].shape[0]
            elif self.sparse_gp:
                n_ind_points = self.n_ind_points
            else:
                n_ind_points = self.train_data['train_targets'].shape[0]

            # explicitly clear the previously generated c code, ocp and solver
            # otherwise the number of parameters will be incorrect
            # TODO: find a better way to handle this
            self.acados_model = None
            self.ocp = None
            self.acados_ocp_solver = None

            # reinitialize the acados model and solver
            self.setup_acados_model(n_ind_points)
            self.setup_acados_optimizer(n_ind_points)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, self.output_dir + '/gpmpc_acados_ocp_solver.json')

        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None
