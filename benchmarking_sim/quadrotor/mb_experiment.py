
import os
import sys
import pickle
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.experiments.epoch_experiments import EpochExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, timing
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.gpmpc_plotting import make_quad_plots

script_path = os.path.dirname(os.path.realpath(__file__))

@timing
def run(gui=False, n_episodes=1, n_steps=None, save_data=True):
    '''The main function running experiments for model-based methods.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''
    # read the additional arguments
    if len(sys.argv) > 1:
        ALGO = sys.argv[1]
    else:
        # ALGO = 'ilqr'
        # ALGO = 'gp_mpc'
        # ALGO = 'gpmpc_acados'
        # ALGO = 'gpmpc_acados_TP'
        # ALGO = 'gpmpc_acados_TRP'
        # ALGO = 'mpc'
        # ALGO = 'mpc_acados'
        # ALGO = 'linear_mpc_acados'
        # ALGO = 'linear_mpc'
        # ALGO = 'lqr'
        # ALGO = 'lqr_c'
        ALGO = 'pid'
    SYS = 'quadrotor_2D_attitude'
    # SYS = 'quadrotor_3D_attitude'
    TASK = 'tracking'
    # TASK = 'stab'
    # PRIOR = '200'
    # PRIOR = '150'
    ADDITIONAL = ''
    # ADDITIONAL = '_9'
    # ADDITIONAL = '_11'
    # ADDITIONAL='_snap'
    PRIOR = '100'
    agent = 'quadrotor' if SYS in ['quadrotor_2D', 'quadrotor_2D_attitude', 'quadrotor_3D_attitude'] else SYS
    SAFETY_FILTER = None
    # SAFETY_FILTER='linear_mpsc'

    # check if the config file exists
    assert os.path.exists(f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml'), f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml does not exist'
    assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
    if SAFETY_FILTER is None:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--seed', '1',
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    else:
        MPSC_COST='one_step_cost'
        assert ALGO != 'gp_mpc', 'Safety filter not supported for gp_mpc'
        assert os.path.exists(f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', agent,
                        '--safety_filter', SAFETY_FILTER,
                        '--overrides',
                            f'./config_overrides/{SYS}_{TASK}{ADDITIONAL}.yaml',
                            f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                            f'./config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml',
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--seed', '2',
                        '--use_gpu', 'True',
                        '--output_dir', f'./{ALGO}/results',
                            ]
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='train', help='main function to run.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config and create output directory
    config = fac.merge()
    if ALGO in ['gpmpc_acados', 'gp_mpc' , 'gpmpc_acados_TP']:
        num_data_max = config.algo_config.num_epochs * config.algo_config.num_samples
        config.output_dir = os.path.join(config.output_dir, PRIOR + '_' + repr(num_data_max))
    # print('output_dir',  config.algo_config.output_dir)
    set_dir_from_config(config)
    config.algo_config.output_dir = config.output_dir
    mkdirs(config.output_dir)

    # Create an environment
    env_func = partial(make,
                       config.task,
                       seed=config.seed,
                       **config.task_config
                       )
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                seed=config.seed,
                **config.algo_config
                )
    
    # Setup safety filter
    if SAFETY_FILTER is not None:
        env_func_filter = partial(make,
                                config.task,
                                seed=config.seed,
                                **config.task_config)
        safety_filter = make(config.safety_filter,
                            env_func_filter,
                            seed=config.seed,
                            **config.sf_config)
        safety_filter.reset()

    all_trajs = defaultdict(list)
    n_episodes = 1 if n_episodes is None else n_episodes

    # Run the experiment.
    for _ in range(n_episodes):
        # Get initial state and create environments
        init_state, _ = random_env.reset()
        # init_state = random_env.reset()
        static_env = env_func(gui=gui, randomized_init=False, init_state=init_state)
        static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

        # Create experiment, train, and run evaluation
        if SAFETY_FILTER is None:  
            if ALGO in ['gpmpc_acados', 'gp_mpc' , 'gpmpc_acados_TP', 'gpmpc_acados_TRP']:
                experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
                if config.algo_config.num_epochs == 1:
                    print('Evaluating prior controller')
                elif config.algo_config.gp_model_path is not None:
                    ctrl.load(config.algo_config.gp_model_path)
                else:
                    # manually launch training 
                    # (NOTE: not using launch_training method since calling plotting before eval will break the eval)
                    experiment.reset()
                    train_runs, test_runs = ctrl.learn(env=static_train_env)
            else:   
                experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
                experiment.launch_training()
        else:
            safety_filter.learn(env=static_train_env)
            mkdirs(f'{script_path}/models/')
            safety_filter.save(path=f'{script_path}/models/{config.safety_filter}_{SYS}_{TASK}_{PRIOR}.pkl')
            ctrl.reset()
            experiment = BaseExperiment(env=static_env, ctrl=ctrl, safety_filter=safety_filter)

        if n_steps is None:
            trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)
        else:
            trajs_data, _ = experiment.run_evaluation(training=True, n_steps=n_steps)

        # plotting training and evaluation results
        # training
        if ALGO in ['gpmpc_acados', 'gp_mpc' , 'gpmpc_acados_TP', 'gpmpc_acados_TRP'] and \
           config.algo_config.gp_model_path is None and \
           config.algo_config.num_epochs > 1:
                if isinstance(static_env, Quadrotor):
                    make_quad_plots(test_runs=test_runs, 
                                    train_runs=train_runs, 
                                    trajectory=ctrl.traj.T,
                                    dir=ctrl.output_dir)
        plot_quad_eval(trajs_data['obs'][0], 
                       trajs_data['action'][0], 
                    #    trajs_data['current_clipped_action'][0],
                       ctrl.env, 
                       config.output_dir)


        # Close environments
        static_env.close()
        static_train_env.close()

        # Merge in new trajectory data
        for key, value in trajs_data.items():
            all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)
    # np.save(f'./data/{ALGO}_ref_traj.npy', all_trajs, allow_pickle=True)
    
    if hasattr(experiment.env, 'dw_model'):
        force_log = experiment.env.dw_model.get_force_log()
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(force_log))/60, force_log)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Downwash force [N]')
        ax.set_title('Downwash force')
        fig.savefig(f'./{config.output_dir}/downwash_force.png')

    if save_data:
        results = {'trajs_data': all_trajs, 'metrics': metrics}
        with open(f'./{config.output_dir}/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)
        # save rmse to a file
        with open(f'./{config.output_dir}/metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            print(f'Metrics saved to ./{config.output_dir}/metrics.txt')

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))
    print(f'pyb_client: {ctrl.env.PYB_CLIENT}')

# def plot_quad_eval(state_stack, input_stack, clipped_action_stack, env, save_path=None):
def plot_quad_eval(state_stack, input_stack, env, save_path=None):
    '''Plots the input and states to determine success.

    Args:
        state_stack (ndarray): The list of observations in the latest run.
        input_stack (ndarray): The list of inputs of in the latest run.
    '''
    model = env.symbolic
    if env.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        x_idx, z_idx = 0, 2
    # elif env.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
    elif env.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10]:
        x_idx, y_idx, z_idx = 0, 2, 4

    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx, figsize=(8, model.nx*1))
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'state_trajectories.png'))

    # Plot inputs
    _, axs = plt.subplots(model.nu, figsize=(8, model.nu*1))
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        # axs[k].plot(times, np.array(clipped_action_stack).transpose()[k, 0:plot_length], color='r')
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'input_trajectories.png'))

    # plot the figure-eight
    fig, axs = plt.subplots(1)
    axs.plot(np.array(state_stack).transpose()[x_idx, 0:plot_length], 
             np.array(state_stack).transpose()[z_idx, 0:plot_length], label='actual')
    axs.plot(reference.transpose()[x_idx, 0:plot_length], 
             reference.transpose()[z_idx, 0:plot_length], color='r', label='desired')
    axs.set_xlabel('x [m]')
    axs.set_ylabel('z [m]')
    axs.set_title('State path in x-z plane')
    axs.legend()
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'state_xz_path.png'))
        print(f'Plots saved to {save_path}')
    if env.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10]:
        fig, axs = plt.subplots(1)
        axs.plot(np.array(state_stack).transpose()[x_idx, 0:plot_length], 
                np.array(state_stack).transpose()[y_idx, 0:plot_length], label='actual')
        axs.plot(reference.transpose()[x_idx, 0:plot_length],
                    reference.transpose()[y_idx, 0:plot_length], color='r', label='desired')
        axs.set_xlabel('x [m]')
        axs.set_ylabel('y [m]')
        axs.set_title('State path in x-y plane')
        axs.legend()
        fig.tight_layout()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'state_xy_path.png'))

    # plt.show()

def wrap2pi_vec(angle_vec):
    '''Wraps a vector of angles between -pi and pi.

    Args:
        angle_vec (ndarray): A vector of angles.
    '''
    for k, angle in enumerate(angle_vec):
        while angle > np.pi:
            angle -= np.pi
        while angle <= -np.pi:
            angle += np.pi
        angle_vec[k] = angle
    return angle_vec


if __name__ == '__main__':
    run()
