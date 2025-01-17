'''This script tests the RL-MPC implementation.'''

import os
import shutil
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Cost, Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(plot=True, training=False, n_episodes=1, n_steps=None, curr_path='.'):
    '''Main function to run RL-MPC experiments.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    system = config.task
    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    results, uncert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    elapsed_time_uncert = results['timestamp'][0][-1] - results['timestamp'][0][0]

    if plot is True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4
        elif system == 'quadrotor_4D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
            graph4_1 = 0
            graph4_2 = 1
        elif system == 'quadrotor_5D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
            graph4_1 = 0
            graph4_2 = 1

        _, ax = plt.subplots()
        ax.plot(results['obs'][0][:, graph1_1], results['obs'][0][:, graph1_2], 'r--', label='RL Trajectory')
        ax.scatter(results['obs'][0][0, graph1_1], results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), results['obs'][0][:, 0], 'r--', label='RL Trajectory')
            ax2.plot(np.linspace(0, 20, results['obs'][0].shape[0]), env.X_GOAL[:, 0], 'b', label='Reference')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(results['obs'][0][:, graph3_1 + 1], results['obs'][0][:, graph3_2 + 1], 'r--', label='RL Trajectory')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(results['obs'][0][:, graph3_1], results['obs'][0][:, graph3_2], 'r--', label='RL Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(env.X_GOAL[:, graph3_1], env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(results['obs'][0][0, graph3_1], results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        if config.task == Environment.QUADROTOR and system == 'quadrotor_2D':
            # _, ax4 = plt.subplots()
            # ax4.plot(results['timestamp'][0][:], results['action'][0][:, graph4_1], 'r', label='Thrust')
            # ax4.set_ylabel(r'Thrust')
            # _, ax5 = plt.subplots()
            # ax5.plot(results['timestamp'][0][:], results['action'][0][:, graph4_2], 'r', label='Pitch')
            # ax5.set_ylabel(r'Pitch')
            _, ax6 = plt.subplots()
            ax6.plot(results['timestamp'][0][:], results['obs'][0][1:, 4], 'r', label='Thrust')
            ax6.set_ylabel(r'Pitch')
            _, ax7 = plt.subplots()
            ax7.plot(results['timestamp'][0][:], results['obs'][0][1:, 5], 'r', label='Pitch')
            ax7.set_ylabel(r'Pitch rate')
        if config.task == Environment.QUADROTOR and system == 'quadrotor_4D':
            _, ax4 = plt.subplots()
            ax4.plot(results['timestamp'][0][:], results['action'][0][:, graph4_1], 'r', label='Action: Thrust')
            ax4.set_ylabel(r'Action: Thrust')
            _, ax5 = plt.subplots()
            ax5.plot(results['timestamp'][0][:], results['action'][0][:, graph4_2], 'r', label='Action: Pitch')
            ax5.set_ylabel(r'Action: Pitch')
            _, ax6 = plt.subplots()
            ax6.plot(results['timestamp'][0][:], results['obs'][0][1:, 4], 'r', label='Obs: Pitch')
            ax6.set_ylabel(r'Obs: Pitch')
            _, ax7 = plt.subplots()
            ax7.plot(results['timestamp'][0][:], results['obs'][0][1:, 5], 'r', label='Obs: Pitch rate')
            ax7.set_ylabel(r'Obs: Pitch rate')
        if config.task == Environment.QUADROTOR and system == 'quadrotor_5D':
            _, ax4 = plt.subplots()
            ax4.plot(results['timestamp'][0][:], results['action'][0][:, graph4_1], 'r', label='Action: Thrust')
            ax4.set_ylabel(r'Action: Thrust')
            _, ax5 = plt.subplots()
            ax5.plot(results['timestamp'][0][:], results['action'][0][:, graph4_2], 'r', label='Action: Pitch')
            ax5.set_ylabel(r'Action: Pitch')
            _, ax6 = plt.subplots()
            ax6.plot(results['timestamp'][0][:], results['obs'][0][1:, 4], 'r', label='Obs: Pitch')
            ax6.set_ylabel(r'Obs: Pitch')

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{curr_path}/perf.png")


if __name__ == '__main__':
    run()
