

import os
import shutil
import time
from datetime import datetime

import casadi as cs
import gpytorch
import munch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from termcolor import colored

from safe_control_gym.controllers.mpc.gp_utils import (ZeroMeanIndependentGPModel,
                                                       covSE_single, kmeans_centriods, GaussianProcess)
from safe_control_gym.controllers.mpc.gpmpc_base import GPMPC
from safe_control_gym.controllers.mpc.mpc_acados import MPC_ACADOS
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.utils.utils import timing
from safe_control_gym.experiments.base_experiment import BaseExperiment

class GPMPC_ACADOS_TP(GPMPC):
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
        self.uncertain_dim = [1, 3, 5]
        self.Bd = np.eye(self.model.nx)[:, self.uncertain_dim]
        self.input_mask = None
        self.target_mask = None
        self.rand_hist = {'task_rand': [], 'domain_rand': []}


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
        self.prior_dynamics_func_c = self.prior_ctrl.model.fc_func

        self.x_guess = None
        self.u_guess = None
        self.x_prev = None
        self.u_prev = None
        print('prior_info[prior_prop]', prior_info['prior_prop'])
        

    def preprocess_training_data(self,
                                 x_seq,
                                 u_seq,
                                 x_next_seq
                                 ):
        '''Converts trajectory data for GP trianing.

        Args:
            x_seq (list): state sequence of np.array (nx,).
            u_seq (list): action sequence of np.array (nu,).
            x_next_seq (list): next state sequence of np.array (nx,).

        Returns:
            np.array: inputs for GP training, (N, nx+nu).
            np.array: targets for GP training, (N, nx).
        '''
        g = 9.81
        dt = 1/60
        T_cmd = u_seq[:, 0]
        T_prior_data = self.prior_ctrl.env.T_mapping_func(T_cmd).full().flatten()
        # numerical differentiation
        x_dot_seq = [(x_next_seq[i, :] - x_seq[i, :])/dt for i in range(x_seq.shape[0])]
        x_dot_seq = np.array(x_dot_seq)

        T_true_data = np.sqrt((x_dot_seq[:, 3] +  g) ** 2 + (x_dot_seq[:, 1] ** 2))
        targets_T = (T_true_data - T_prior_data).reshape(-1, 1)
        input_T = u_seq[:, 0].reshape(-1, 1)

        theta_true = x_dot_seq[:, 5]
        theta_prior = self.prior_dynamics_func_c(x=x_seq.T, u=u_seq.T)['f'].toarray()[5, :]
        targets_theta = (theta_true - theta_prior).reshape(-1, 1)
        input_theta = np.concatenate([x_seq[:, 4].reshape(-1, 1), 
                                      x_seq[:, 5].reshape(-1, 1), 
                                      u_seq[:, 1].reshape(-1, 1)], axis=1) 
        
        train_input = np.concatenate([input_T, input_theta], axis=1)
        train_output = np.concatenate([targets_T, targets_theta], axis=1)


        return train_input, train_output

    def learn(self, env=None):
        '''Performs multiple epochs learning.
        '''

        train_runs = {0: {}}
        test_runs = {0: {}}
        
        # epoch seed factor 
        np.random.seed(self.seed)
        epoch_seeds = np.random.randint(1000, size=self.num_epochs, dtype=int) * self.seed
        epoch_seeds = [int(seed) for seed in epoch_seeds]

        if self.same_train_initial_state:
            train_envs = []
            for epoch in range(self.num_epochs):
                train_envs.append(self.env_func(randomized_init=True, seed=epoch_seeds[epoch]))
                train_envs[epoch].action_space.seed(epoch_seeds[epoch])
        else:
            train_env = self.env_func(randomized_init=True, seed=epoch_seeds[0])
            train_env.action_space.seed(epoch_seeds[0])
            train_envs = [train_env] * self.num_epochs
        
        test_envs = []
        if self.same_test_initial_state:
            for epoch in range(self.num_epochs):
                test_envs.append(self.env_func(randomized_init=True, seed=epoch_seeds[epoch]))
                test_envs[epoch].action_space.seed(epoch_seeds[epoch])
        else:
            test_env = self.env_func(randomized_init=True, seed=epoch_seeds[0])
            test_env.action_space.seed(epoch_seeds[0])
            test_envs = [test_env] * self.num_epochs

        for env in train_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]
        for env in test_envs:
            if isinstance(env.EPISODE_LEN_SEC, list):
                idx = np.random.choice(len(env.EPISODE_LEN_SEC))
                env.EPISODE_LEN_SEC = env.EPISODE_LEN_SEC[idx]

        # creating train and test experiments
        train_experiments = [BaseExperiment(env=env, ctrl=self, reset_when_created=False) for env in train_envs[1:]]
        test_experiments = [BaseExperiment(env=env, ctrl=self, reset_when_created=False) for env in test_envs[1:]]
        # first experiments are for the prior
        train_experiments.insert(0, BaseExperiment(env=train_envs[0], ctrl=self.prior_ctrl, reset_when_created=False))
        test_experiments.insert(0, BaseExperiment(env=test_envs[0], ctrl=self.prior_ctrl, reset_when_created=False))
        
        for episode in range(self.num_train_episodes_per_epoch):
            self.env = train_envs[0]
            run_results = train_experiments[0].run_evaluation(n_episodes=1)
            train_runs[0].update({episode: munch.munchify(run_results)})
        for test_ep in range(self.num_test_episodes_per_epoch):
            self.env = test_envs[0]
            run_results = test_experiments[0].run_evaluation(n_episodes=1)
            test_runs[0].update({test_ep: munch.munchify(run_results)})
        
        training_results = None
        for epoch in range(1, self.num_epochs):
            # only take data from the last episode from the last epoch
            episode_length = train_runs[epoch - 1][self.num_train_episodes_per_epoch - 1][0]['obs'][0].shape[0]
            x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(train_runs, epoch - 1, self.num_samples, train_envs[epoch - 1].np_random)
            train_inputs, train_targets = self.preprocess_training_data(x_seq, actions, x_next_seq) # np.ndarray
            training_results = self.train_gp(input_data=train_inputs, target_data=train_targets)
            
            if self.plot_trained_gp:
                self.plot_gp_TP(train_inputs, train_targets, title=f'epoch_{epoch}_train', output_dir=self.output_dir)
            
            # Test new policy.
            test_runs[epoch] = {}
            for test_ep in range(self.num_test_episodes_per_epoch):
                self.x_prev = test_runs[epoch - 1][episode][0]['obs'][0][:self.T + 1, :].T
                self.u_prev = test_runs[epoch - 1][episode][0]['action'][0][:self.T, :].T
                self.env = test_envs[epoch]
                run_results = test_experiments[epoch].run_evaluation(n_episodes=1)
                test_runs[epoch].update({test_ep: munch.munchify(run_results)})

            x_seq, actions, x_next_seq, x_dot_seq = self.gather_training_samples(test_runs, epoch - 1, episode_length)
            train_inputs, train_targets = self.preprocess_training_data(x_seq, actions, x_next_seq) # np.ndarray
            if self.plot_trained_gp:
                self.plot_gp_TP(train_inputs, train_targets, title=f'epoch_{epoch}_test', output_dir=self.output_dir)

            # gather training data
            train_runs[epoch] = {}
            for episode in range(self.num_train_episodes_per_epoch):
                self.x_prev = train_runs[epoch - 1][episode][0]['obs'][0][:self.T + 1, :].T
                self.u_prev = train_runs[epoch - 1][episode][0]['action'][0][:self.T, :].T
                self.env = train_envs[epoch]
                run_results = train_experiments[epoch].run_evaluation(n_episodes=1)
                train_runs[epoch].update({episode: munch.munchify(run_results)})

            # compute the condition number of the kernel matrix
            self.rand_hist['task_rand'].append(test_experiments[epoch].env.episode_len)
            domain_rand_info = {}
            for keys, values in test_experiments[epoch].env.disturbances.items():
                if keys == 'downwash':
                    domain_rand_info[keys] = test_experiments[epoch].env.dw_model.pos
                else:
                    domain_rand_info[keys] = values.disturbances[0].std
            self.rand_hist['domain_rand'].append(domain_rand_info)
            # TODO: fix data logging
            np.savez(os.path.join(self.output_dir, 'epoch_data'),
                    data_inputs=training_results['train_inputs'],
                    data_targets=training_results['train_targets'],
                    train_runs=train_runs,
                    test_runs=test_runs,
                    num_epochs=self.num_epochs,
                    num_train_episodes_per_epoch=self.num_train_episodes_per_epoch,
                    num_test_episodes_per_epoch=self.num_test_episodes_per_epoch,
                    num_samples=self.num_samples,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    )

        if training_results:
            np.savez(os.path.join(self.output_dir, 'data'),
                    data_inputs=training_results['train_inputs'],
                    data_targets=training_results['train_targets'])

        # close environments
        for experiment in train_experiments:
            experiment.env.close()
        for experiment in test_experiments:
            experiment.env.close()
        # delete c_generated_code folder and acados_ocp_solver.json files
        # os.system(f'rm -rf {self.output_dir}/*c_generated_code*')
        # os.system(f'rm -rf {self.output_dir}/*acados_ocp_solver*')

        self.train_runs = train_runs
        self.test_runs = test_runs

        return train_runs, test_runs
    
    def load(self, model_path):
        '''Load the model from a file.

        Args:
            model_path (str): Path to the model to load.
        '''
        data = np.load(f'{model_path}/data.npz')
        gp_model_path_T = f'{model_path}/best_model_T.pth'
        gp_model_path_P = f'{model_path}/best_model_P.pth'
        gp_model_path = [gp_model_path_T, gp_model_path_P]
        self.train_gp(input_data=data['data_inputs'], 
                        target_data=data['data_targets'],
                        gp_model=gp_model_path,)
    
    @timing
    def train_gp(self,
                 input_data, target_data,
                 gp_model=None,
                 overwrite_saved_data: bool = None,
                 train_hardware_data: bool = False,
                 ):
        '''Performs GP training.

        Args:
            input_data, target_data (optiona, np.array): data to use for training
            gp_model (str): if not None, this is the path to pretrained models to use instead of training new ones.
            overwrite_saved_data (bool): Overwrite the input and target data to the already saved data if it exists.
            train_hardware_data (bool): True to train on hardware data. If true, will load the data and perform training.
        Returns:
            training_results (dict): Dictionary of the training results.
        '''
        if gp_model is None and not train_hardware_data:
            gp_model = self.gp_model_path
        if overwrite_saved_data is None:
            overwrite_saved_data = self.overwrite_saved_data
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
                random_state=self.seed
            )

        else:
            # Otherwise, just copy the training data into the test data.
            train_idx = list(range(total_input_data))
            test_idx = list(range(total_input_data))

        train_inputs = self.data_inputs[train_idx, :]
        train_targets = self.data_targets[train_idx, :]
        self.train_data = {'train_inputs': train_inputs, 'train_targets': train_targets}
        test_inputs = self.data_inputs[test_idx, :]
        test_targets = self.data_targets[test_idx, :]
        self.test_data = {'test_inputs': test_inputs, 'test_targets': test_targets}


        train_inputs_tensor = torch.Tensor(train_inputs).double()
        train_targets_tensor = torch.Tensor(train_targets).double()
        test_inputs_tensor = torch.Tensor(test_inputs).double()
        test_targets_tensor = torch.Tensor(test_targets).double()

        # seperate the data for T and P
        train_input_T = train_inputs_tensor[:, 0].reshape(-1)
        train_target_T = train_targets_tensor[:, 0].reshape(-1)
        test_inputs_T = test_inputs_tensor[:, 0].reshape(-1)
        test_targets_T = test_targets_tensor[:, 0].reshape(-1)
        train_input_P = train_inputs_tensor[:, 1:].reshape(-1, 3)
        test_inputs_P = test_inputs_tensor[:, 1:].reshape(-1, 3)
        train_target_P = train_targets_tensor[:, 1:].reshape(-1)
        test_targets_P = test_targets_tensor[:, 1:].reshape(-1)

        # Define likelihood.
        likelihood_T = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()
        likelihood_P = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6),
        ).double()

        GP_T = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_T,
            kernel='RBF_single', 
        )

        GP_P = GaussianProcess(
            model_type=ZeroMeanIndependentGPModel,
            likelihood=likelihood_P,
            kernel='RBF_single',
        )

        if gp_model:
            print(colored(f'Loaded pretrained model from {self.gp_model_path}', 'green'))
            GP_T.init_with_hyperparam(train_input_T, train_target_T, gp_model[0])
            GP_P.init_with_hyperparam(train_input_P, train_target_P, gp_model[1])
        else:
            GP_T.train(train_input_T, train_target_T, test_inputs_T, test_targets_T,
                    n_train=self.optimization_iterations[0], learning_rate=self.learning_rate[0], 
                    gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_T.pth'))
            GP_P.train(train_input_P, train_target_P, test_inputs_P, test_targets_P,
                    n_train=self.optimization_iterations[1], learning_rate=self.learning_rate[1],
                    gpu=self.use_gpu, fname=os.path.join(self.output_dir, 'best_model_P.pth'))

        self.gaussian_process = [GP_T, GP_P]

        self.reset()
        training_results = {}
        training_results['train_targets'] = train_targets
        training_results['train_inputs'] = train_inputs
        return training_results

    def setup_acados_model(self, n_ind_points) -> AcadosModel:

        # setup GP related
        self.inverse_cdf = scipy.stats.norm.ppf(1 - (1 / self.model.nx - (self.prob + 1) / (2 * self.model.nx)))
        self.create_sparse_GP_machinery(n_ind_points)

        # setup acados model
        model_name = self.env.NAME

        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        acados_model.name = self.env.NAME + '_' + current_time

        z = cs.vertcat(acados_model.x, acados_model.u)  # GP prediction point
        T_pred_point = z[6]
        P_pred_point = z[[4, 5, 7]]

        if self.sparse_gp:
            # sparse GP inducing points
            '''
            z_ind should be of shape (n_ind_points, z.shape[0]) or (n_ind_points, len(self.input_mask))
            mean_post_factor should be of shape (len(self.target_mask), n_ind_points)
            Here we create the corresponding parameters since acados supports only 1D parameters
            '''
            z_ind = cs.MX.sym('z_ind', n_ind_points, 4)
            mean_post_factor = cs.MX.sym('mean_post_factor', 2, n_ind_points)
            acados_model.p = cs.vertcat(cs.reshape(z_ind, -1, 1), cs.reshape(mean_post_factor, -1, 1))
            # define the dynamics
            T_pred = cs.sum2(self.K_z_zind_func_T(z1=T_pred_point, z2=z_ind)['K'] * mean_post_factor[0, :])
            P_pred = cs.sum2(self.K_z_zind_func_P(z1=P_pred_point, z2=z_ind)['K'] * mean_post_factor[1, :])
            f_cont = self.prior_dynamics_func_c(x=acados_model.x, u=acados_model.u)['f']\
                    + cs.vertcat(0, cs.sin(acados_model.x[4])*T_pred,
                                 0, cs.cos(acados_model.x[4])*T_pred,
                                 0, P_pred)
            f_cont_func = cs.Function('f_cont_func', [acados_model.x, acados_model.u, acados_model.p], [f_cont])
            # use rk4 to discretize the continuous dynamics
            k1 = f_cont_func(acados_model.x, acados_model.u, acados_model.p)
            k2 = f_cont_func(acados_model.x + self.dt/2 * k1, acados_model.u, acados_model.p)
            k3 = f_cont_func(acados_model.x + self.dt/2 * k2, acados_model.u, acados_model.p)
            k4 = f_cont_func(acados_model.x + self.dt * k3, acados_model.u, acados_model.p)
            f_disc = acados_model.x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        else:
            GP_T = self.gaussian_process[0]
            GP_P = self.gaussian_process[1]
            T_pred = GP_T.casadi_predict(T_pred_point)
            P_pred = GP_P.casadi_predict(P_pred_point)
            # T_pred = 

            f_cont = self.prior_dynamics_func_c(x=acados_model.x, u=acados_model.u)['f']\
                    + cs.vertcat(0, cs.sin(acados_model.x[4])*T_pred,
                                 0, cs.cos(acados_model.x[4])*T_pred,
                                 0, P_pred)
            f_cont_func = cs.Function('f_cont_func', [acados_model.x, acados_model.u, acados_model.p], [f_cont])
            # use rk4 to discretize the continuous dynamics
            k1 = f_cont_func(acados_model.x, acados_model.u, acados_model.p)
            k2 = f_cont_func(acados_model.x + self.dt/2 * k1, acados_model.u, acados_model.p)
            k3 = f_cont_func(acados_model.x + self.dt/2 * k2, acados_model.u, acados_model.p)
            k4 = f_cont_func(acados_model.x + self.dt * k3, acados_model.u, acados_model.p)
            f_disc = acados_model.x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        acados_model.disc_dyn_expr = f_disc

        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model
        T_cmd = cs.MX.sym('T_cmd')
        theta_cmd = cs.MX.sym('theta_cmd')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        T_true_func = self.env.T_mapping_func
        T_prior_func = self.prior_ctrl.env.T_mapping_func
        T_res = T_true_func(T_cmd) - T_prior_func(T_cmd)
        self.T_res_func = cs.Function('T_res_func', [T_cmd], [T_res])
        P_true_func = self.env.P_mapping_func
        P_prior_func = self.prior_ctrl.env.P_mapping_func
        P_res = P_true_func(theta, theta_dot, theta_cmd) - P_prior_func(theta, theta_dot, theta_cmd)
        self.P_res_func = cs.Function('P_res_func', [theta, theta_dot, theta_cmd], [P_res])

    def setup_acados_optimizer(self, n_ind_points):
        print('=================Setting up GPMPC acados optimizer=================')
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
        if self.sparse_gp:
            ocp.parameter_values = np.zeros((ocp.model.p.shape[0], ))  # dummy values

        # placeholder initial state constraint
        x_init = np.zeros((nx))
        ocp.constraints.x0 = x_init

        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'DISCRETE'
        # ocp.solver_options.integrator_type = 'ERK'

        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.nlp_solver_max_iter = 10
        ocp.solver_options.qp_solver_iter_max = 10
        ocp.solver_options.qp_tol = 1e-4
        ocp.solver_options.tol = 1e-4

        # prediction horizon
        ocp.solver_options.tf = self.T * self.dt

        # c code generation
        ocp.code_export_directory = self.output_dir + '/gpmpc_c_generated_code'

        self.ocp = ocp
        self.opti_dict = {'n_ind_points': n_ind_points}
        # compute sparse GP values
        # the actual values will be set in select_action_with_gp
        self.mean_post_factor_val_all, self.z_ind_val_all = self.precompute_mean_post_factor_all_data()
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


    # @timing
    def select_action(self, obs, info=None):
        time_before = time.time()
        if self.gaussian_process is None:
            action = self.prior_ctrl.select_action(obs)
        else:
            action = self.select_action_with_gp(obs)
        time_after = time.time()
        self.results_dict['runtime'].append(time_after - time_before)
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
            assert z_ind_val.shape == (n_ind_points, 4)
            assert mean_post_factor_val.shape == (2, n_ind_points)
            # casadi use column major order, while np uses row major order by default
            # Thus, Fortran order (column major) is used to reshape the arrays
            z_ind_val = z_ind_val.reshape(-1, 1, order='F')
            mean_post_factor_val = mean_post_factor_val.reshape(-1, 1, order='F')
            dyn_value = np.concatenate((z_ind_val, mean_post_factor_val)).reshape(-1)
            # tighten constraints
            for idx in range(self.T):
                # set the parameter values
                self.acados_ocp_solver.set(idx, "p", dyn_value)
            # set the parameter values
            self.acados_ocp_solver.set(self.T, "p", dyn_value)

        # set reference for the control horizon
        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1
        y_ref = np.concatenate((goal_states[:, :-1], np.repeat(self.U_EQ.reshape(-1, 1), self.T, axis=1)), axis=0)
        for idx in range(self.T):
            self.acados_ocp_solver.set(idx, 'yref', y_ref[:, idx])
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
    
    def precompute_mean_post_factor_all_data(self):
        '''If the number of data points is less than the number of inducing points, use all the data
        as kernel points.
        '''
        dim_gp_outputs = len(self.gaussian_process)
        n_training_samples = self.train_data['train_targets'].shape[0]
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']
        mean_post_factor = np.zeros((dim_gp_outputs, n_training_samples))
        for i in range(dim_gp_outputs):
            K_z_z = self.gaussian_process[i].model.K_plus_noise_inv
            mean_post_factor[i] = K_z_z.detach().numpy() @ targets[:, i]

        return mean_post_factor, inputs

    def precompute_sparse_gp_values(self, n_ind_points):
        '''Uses the last MPC solution to precomupte values associated with the FITC GP approximation.

        Args:
            n_ind_points (int): Number of inducing points.
        '''
        n_data_points = self.train_data['train_targets'].shape[0]
        dim_gp_outputs = len(self.gaussian_process)
        inputs = self.train_data['train_inputs']
        targets = self.train_data['train_targets']
        # Get the inducing points.
        if False and self.x_prev is not None and self.u_prev is not None:
            # Use the previous MPC solution as in Hewing 2019.
            z_prev = np.hstack((self.x_prev[:, :-1].T, self.u_prev.T))
            z_prev = z_prev[:, self.input_mask]
            inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points - self.T, replace=False)
            # z_ind = self.data_inputs[inds][:, self.input_mask]
            z_ind = np.vstack((z_prev, inputs[inds][:, self.input_mask]))
        else:
            # If there is no previous solution. Choose T random training set points.
            if self.inducing_point_selection_method == 'kmeans':
                centroids = kmeans_centriods(n_ind_points, inputs, rand_state=self.seed)
                contiguous_masked_inputs = np.ascontiguousarray(inputs)  # required for version sklearn later than 1.0.2
                inds, _ = pairwise_distances_argmin_min(centroids, contiguous_masked_inputs)
                z_ind = inputs[inds]
            elif self.inducing_point_selection_method == 'random':
                inds = self.env.np_random.choice(range(n_data_points), size=n_ind_points, replace=False)
                z_ind = inputs[inds]
            else:
                raise ValueError('[Error]: gp_mpc.precompute_sparse_gp_values: Only \'kmeans\' or \'random\' allowed.')
        
        use_pinv = True
        # use_pinv = False
        
        GP_T = self.gaussian_process[0]
        GP_P = self.gaussian_process[1]
        K_zind_zind_T = GP_T.model.covar_module(torch.from_numpy(z_ind[:,0]).double())
        K_zind_zind_P = GP_P.model.covar_module(torch.from_numpy(z_ind[:,1:]).double())
        if use_pinv:
            K_zind_zind_inv_T = torch.pinverse(K_zind_zind_T.evaluate().detach())
            K_zind_zind_inv_P = torch.pinverse(K_zind_zind_P.evaluate().detach())
        else:
            K_zind_zind_inv_T = K_zind_zind_T.inv_matmul(torch.eye(n_ind_points).double()).detach()
            K_zind_zind_inv_P = K_zind_zind_P.inv_matmul(torch.eye(n_ind_points).double()).detach()
        K_zind_zind_T = GP_T.model.covar_module(torch.from_numpy(z_ind[:,0]).double()).evaluate().detach()
        K_zind_zind_P = GP_P.model.covar_module(torch.from_numpy(z_ind[:,1:]).double()).evaluate().detach()
        K_zind_zind = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        K_zind_zind[0] = K_zind_zind_T
        K_zind_zind[1] = K_zind_zind_P

        K_zind_zind_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        K_zind_zind_inv[0] = K_zind_zind_inv_T
        K_zind_zind_inv[1] = K_zind_zind_inv_P
        K_x_zind_T = GP_T.model.covar_module(torch.from_numpy(inputs[:,0]).double(), torch.from_numpy(z_ind[:,0]).double()).evaluate().detach()
        K_x_zind_P = GP_P.model.covar_module(torch.from_numpy(inputs[:,1:]).double(), torch.from_numpy(z_ind[:,1:]).double()).evaluate().detach()
        K_x_zind = torch.zeros((dim_gp_outputs, n_data_points, n_ind_points))
        K_x_zind[0] = K_x_zind_T
        K_x_zind[1] = K_x_zind_P
        
        K_plus_noise_T = GP_T.model.K_plus_noise.detach()
        K_plus_noise_P = GP_P.model.K_plus_noise.detach()
        K_plus_noise = torch.zeros((dim_gp_outputs, n_data_points, n_data_points))
        K_plus_noise[0] = K_plus_noise_T
        K_plus_noise[1] = K_plus_noise_P

        if use_pinv:
            Q_X_X_T = K_x_zind_T @ K_zind_zind_inv_T @ K_x_zind_T.T
            Q_X_X_P = K_x_zind_P @ K_zind_zind_inv_P @ K_x_zind_P.T
        else:
            Q_X_X_T = K_x_zind_T @ torch.linalg.solve(K_zind_zind_T, K_x_zind_T.T)
            Q_X_X_P = K_x_zind_P @ torch.linalg.solve(K_zind_zind_P, K_x_zind_P.T)

        Gamma_T = torch.diagonal(K_plus_noise_T - Q_X_X_T)
        Gamma_inv_T = torch.diag_embed(1 / Gamma_T)
        Gamma_P = torch.diagonal(K_plus_noise_P - Q_X_X_P)
        Gamma_inv_P = torch.diag_embed(1 / Gamma_P)

        Sigma_inv_T = K_zind_zind_T + K_x_zind_T.T @ Gamma_inv_T @ K_x_zind_T
        Sigma_inv_P = K_zind_zind_P + K_x_zind_P.T @ Gamma_inv_P @ K_x_zind_P
        Sigma_inv = torch.zeros((dim_gp_outputs, n_ind_points, n_ind_points))
        Sigma_inv[0] = Sigma_inv_T
        Sigma_inv[1] = Sigma_inv_P

        if use_pinv:
            Sigma_T = torch.pinverse(Sigma_inv_T)
            Sigma_P = torch.pinverse(Sigma_inv_P)
            mean_post_factor_T = Sigma_T @ K_x_zind_T.T @ Gamma_inv_T @ torch.from_numpy(targets[:,0]).double()
            mean_post_factor_P = Sigma_P @ K_x_zind_P.T @ Gamma_inv_P @ torch.from_numpy(targets[:,1]).double()
        else:
            mean_post_factor_T = torch.linalg.solve(Sigma_inv_T, K_x_zind_T.T @ Gamma_inv_T @ torch.from_numpy(targets[:,0]).double())
            mean_post_factor_P = torch.linalg.solve(Sigma_inv_P, K_x_zind_P.T @ Gamma_inv_P @ torch.from_numpy(targets[:,1]).double())

        mean_post_factor = torch.zeros((dim_gp_outputs, n_ind_points))
        mean_post_factor[0] = mean_post_factor_T
        mean_post_factor[1] = mean_post_factor_P

        return mean_post_factor.detach().numpy(), Sigma_inv.detach().numpy(), K_zind_zind_inv.detach().numpy(), z_ind

    def create_sparse_GP_machinery(self, n_ind_points):
        '''This setups the gaussian process approximations for FITC formulation.'''
        GP_T = self.gaussian_process[0]
        GP_P = self.gaussian_process[1]
        lengthscales_T = GP_T.model.covar_module.base_kernel.lengthscale.detach().numpy()
        lengthscales_P = GP_P.model.covar_module.base_kernel.lengthscale.detach().numpy()
        signal_var_T = GP_T.model.covar_module.outputscale.detach().numpy()
        signal_var_P = GP_P.model.covar_module.outputscale.detach().numpy()
        noise_var_T = GP_T.likelihood.noise.detach().numpy()
        noise_var_P = GP_P.likelihood.noise.detach().numpy()
        gp_K_plus_noise_T = GP_T.model.K_plus_noise.detach().numpy()
        gp_K_plus_noise_P = GP_P.model.K_plus_noise.detach().numpy()

        # stacking
        lengthscales = np.vstack((lengthscales_T, lengthscales_P))
        signal_var = np.array([signal_var_T, signal_var_P])
        noise_var = np.array([noise_var_T, noise_var_P])
        gp_K_plus_noise = np.zeros((2, gp_K_plus_noise_T.shape[0], gp_K_plus_noise_T.shape[1]))
        gp_K_plus_noise[0] = gp_K_plus_noise_T
        gp_K_plus_noise[1] = gp_K_plus_noise_P
        
        self.length_scales = lengthscales.squeeze()
        self.signal_var = signal_var.squeeze()
        self.noise_var = noise_var.squeeze()
        self.gp_K_plus_noise = gp_K_plus_noise
        Nx =self.train_data['train_inputs'].shape[1]
        Ny =self.train_data['train_targets'].shape[1]
        # Create CasADI function for computing the kernel K_z_zind with parameters for z, z_ind, length scales and signal variance.
        # We need the CasADI version of this so that it can by symbolically differentiated in in the MPC optimization.
        z1_T = cs.SX.sym('z1', 1)
        z2_T = cs.SX.sym('z2', 1)
        ell_s_T = cs.SX.sym('ell', 1)
        sf2_s_T = cs.SX.sym('sf2')
        z1_P = cs.SX.sym('z1', 3)
        z2_P = cs.SX.sym('z2', 3)
        ell_s_P = cs.SX.sym('ell', 1)
        sf2_s_P = cs.SX.sym('sf2')
        z_ind = cs.SX.sym('z_ind', n_ind_points, Nx)
        ks_T = cs.SX.zeros(1, n_ind_points) # kernel vector
        ks_P = cs.SX.zeros(1, n_ind_points) # kernel vector

        covSE_T = cs.Function('covSE', [z1_T, z2_T, ell_s_T, sf2_s_T], 
                                       [covSE_single(z1_T, z2_T, ell_s_T, sf2_s_T)])
        covSE_P = cs.Function('covSE', [z1_P, z2_P, ell_s_P, sf2_s_P],
                                       [covSE_single(z1_P, z2_P, ell_s_P, sf2_s_P)])
        for i in range(n_ind_points):
            ks_T[i] = covSE_T(z1_T, z_ind[i, 0], ell_s_T, sf2_s_T)
            ks_P[i] = covSE_P(z1_P, z_ind[i, 1:], ell_s_P, sf2_s_P)
        ks_func_T = cs.Function('K_s', [z1_T, z_ind, ell_s_T, sf2_s_T], [ks_T])
        ks_func_P = cs.Function('K_s', [z1_P, z_ind, ell_s_P, sf2_s_P], [ks_P])

        K_z_zind_T = ks_func_T(z1_T, z_ind, self.length_scales[0], self.signal_var[0])
        K_z_zind_P = ks_func_P(z1_P, z_ind, self.length_scales[1], self.signal_var[1])
        self.K_z_zind_func_T = cs.Function('K_z_zind', [z1_T, z_ind], [K_z_zind_T], ['z1', 'z2'], ['K'])
        self.K_z_zind_func_P = cs.Function('K_z_zind', [z1_P, z_ind], [K_z_zind_P], ['z1', 'z2'], ['K'])

    @timing
    def reset(self):
        print(colored('Resetting the GPMPC controller.', 'green'))
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
            # delete the generated c code directory
            if os.path.exists(self.output_dir + '/gpmpc_c_generated_code'):
                print('deleting the generated c code directory')
                shutil.rmtree(self.output_dir + '/gpmpc_c_generated_code', ignore_errors=False)
                assert not os.path.exists(self.output_dir + '/gpmpc_c_generated_code')

            # reinitialize the acados model and solver
            self.setup_acados_model(n_ind_points)
            self.setup_acados_optimizer(n_ind_points)
            # get time in $ymd_HMS format
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp)

        self.prior_ctrl.reset()
        self.setup_results_dict()
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.x_guess = None
        self.u_guess = None

    def plot_gp_TP(self, train_inputs, train_targets, title=None, output_dir=None):
        num_data = train_inputs.shape[0]
        mean_T, _, preds_T = self.gaussian_process[0].predict(train_inputs[:, 0].reshape(-1, 1))
        mean_P, _, preds_P = self.gaussian_process[1].predict(train_inputs[:, 1:].reshape(-1, 3))
        t = np.arange(num_data)
        lower_T, upper_T = preds_T.confidence_region()
        lower_P, upper_P = preds_P.confidence_region()
        mean_T = mean_T.numpy()
        mean_P = mean_P.numpy()
        lower_T = lower_T.numpy()
        upper_T = upper_T.numpy()
        lower_P = lower_P.numpy()
        upper_P = upper_P.numpy()
        residual_T = self.T_res_func(train_inputs[:, 0].reshape(-1, 1)).full().flatten()
        residual_P = self.P_res_func(train_inputs[:, 1].reshape(-1, 1),
                                     train_inputs[:, 2].reshape(-1, 1),
                                     train_inputs[:, 3].reshape(-1, 1)).full().flatten()
        num_within_2std_T = np.sum((train_targets[:, 0] > lower_T) & (train_targets[:, 0] < upper_T))
        percentage_within_2std_T = num_within_2std_T / num_data * 100
        num_within_2std_P = np.sum((train_targets[:, 1] > lower_P) & (train_targets[:, 1] < upper_P))
        percentage_within_2std_P = num_within_2std_P / num_data * 100

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].scatter(t, train_targets[:, 0], label='Target', color='gray')
        ax[0].plot(t, mean_T, label='GP mean', color='blue')
        ax[0].fill_between(t, lower_T, upper_T, alpha=0.5, color='skyblue', label='2-$\sigma$')
        ax[0].plot(t, residual_T, label='Residual (analytical)', color='green')
        ax[0].set_ylabel('T residual [$m/s^2$]')
        ax[0].set_xlabel('data points')
        ax[0].set_title(f'T residual, {percentage_within_2std_T:.2f}% within 2-$\sigma$')
        ax[0].legend()

        ax[1].scatter(train_inputs[:, 0], train_targets[:, 0], label='Target', color='gray')
        ax[1].plot(train_inputs[:, 0], mean_T, label='GP mean', color='blue')
        ax[1].plot(train_inputs[:, 0], residual_T, label='Residual (analytical)', color='green')
        ax[1].legend()
        ax[1].set_ylabel('T residual [$m/s^2$]')
        ax[1].set_xlabel('$T_c$ [$N$]')
        ax[1].set_title(f'Acc residual vs $T_c$')

        ax[2].scatter(t, train_targets[:, 1], label='Target', color='gray')
        ax[2].plot(t, mean_P, label='GP mean', color='blue')
        ax[2].fill_between(t, lower_P, upper_P, alpha=0.5, color='skyblue', label='2-$\sigma$')
        ax[2].plot(t, residual_P, label='Residual (analytical)', color='green')
        ax[2].set_title('P')
        ax[2].legend()
        ax[2].set_ylabel('P residual [$rad/s^2$]')
        ax[2].set_xlabel('data points')
        ax[2].set_title(f'P residual, {percentage_within_2std_P:.2f}% within 2-$\sigma$')
        plt_title = f'GP_validation_{title}'
        plt.suptitle(plt_title)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{plt_title}.png'))
        print(f'Plot saved at {os.path.join(output_dir, f"{plt_title}.png")}')
        plt.close()