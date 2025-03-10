'''A PID example on a quadrotor.'''

import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from safe_control_gym.envs.benchmark_env import Environment, Task

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_dir_from_config


def run(gui=False, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running PID experiments.

    Args:
        gui (bool): Whether to display the gui and plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    config.task_config['gui'] = gui

    custom_trajectory = False
    if config.task_config.task == 'traj_tracking' and config.task_config.task_info.trajectory_type == 'custom':
        custom_trajectory = True
        config.task_config.task_info.trajectory_type = 'circle'  # Placeholder
        config.task_config.randomized_init = False
        config.task_config.init_state = np.zeros((12, 1))

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                )

    if custom_trajectory:
        # Set iterations and episode counter.
        ITERATIONS = int(config.task_config['episode_len_sec'] * config.task_config['ctrl_freq']) + 2  # +2 for start and end of reference

        # Curve fitting with waypoints.
        waypoints = np.array([(0, 0, 0), (0.2, 0.5, 0.5), (0.5, 0.1, 0.6), (1, 1, 1), (1.3, 1, 1.2)])
        deg = 6
        t = np.arange(waypoints.shape[0])
        fit_x = np.polyfit(t, waypoints[:, 0], deg)
        fit_y = np.polyfit(t, waypoints[:, 1], deg)
        fit_z = np.polyfit(t, waypoints[:, 2], deg)
        fx = np.poly1d(fit_x)
        fy = np.poly1d(fit_y)
        fz = np.poly1d(fit_z)
        t_scaled = np.linspace(t[0], t[-1], ITERATIONS)
        ref_x = fx(t_scaled)
        ref_y = fy(t_scaled)
        ref_z = fz(t_scaled)

        X_GOAL = np.zeros((ITERATIONS, ctrl.env.symbolic.nx))
        X_GOAL[:, 0] = ref_x
        X_GOAL[:, 2] = ref_y
        X_GOAL[:, 4] = ref_z

        ctrl.env.X_GOAL = X_GOAL
        ctrl.reference = X_GOAL

    obs, _ = ctrl.env.reset()

    if config.task_config.task == 'traj_tracking' and gui is True:
        if config.task_config.quad_type == 2:
            ref_3D = np.hstack([ctrl.env.X_GOAL[:, [0]], np.zeros(ctrl.env.X_GOAL[:, [0]].shape), ctrl.env.X_GOAL[:, [2]]])
        else:
            ref_3D = ctrl.env.X_GOAL[:, [0, 2, 4]]
        # Plot in 3D.
        ax = plt.axes(projection='3d')
        ax.plot3D(ref_3D[:, 0], ref_3D[:, 1], ref_3D[:, 2])
        if custom_trajectory:
            ax.scatter3D(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2])
        plt.show()

        for i in range(10, ctrl.env.X_GOAL.shape[0], 10):
            p.addUserDebugLine(lineFromXYZ=[ref_3D[i - 10, 0], ref_3D[i - 10, 1], ref_3D[i - 10, 2]],
                               lineToXYZ=[ref_3D[i, 0], ref_3D[i, 1], ref_3D[i, 2]],
                               lineColorRGB=[1, 0, 0],
                               physicsClientId=ctrl.env.PYB_CLIENT)

        if custom_trajectory:
            for point in waypoints:
                p.loadURDF(os.path.join(ctrl.env.URDF_DIR, 'gate.urdf'),
                           [point[0], point[1], point[2] - 0.05],
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=ctrl.env.PYB_CLIENT)

    # Run the experiment.
    experiment = BaseExperiment(ctrl.env, ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    iterations = len(trajs_data['action'][0])
    for i in range(iterations):
        # Step the environment and print all returned information.
        obs, reward, done, info, action = trajs_data['obs'][0][i], trajs_data['reward'][0][i], trajs_data['done'][0][i], trajs_data['info'][0][i], trajs_data['action'][0][i]

        # # Print the last action and the information returned at each step.
        # print(i, '-th step.')
        # print(action, '\n', obs, '\n', reward, '\n', done, '\n', info, '\n')

    elapsed_sec = trajs_data['timestamp'][0][-1] - trajs_data['timestamp'][0][0]
    # print(f'\n{iterations} iterations (@{config.task_config.ctrl_freq}Hz) in {elapsed_sec:.2f} seconds, i.e. {iterations / elapsed_sec:.2f} steps/sec for a {(iterations * (1. / config.task_config.ctrl_freq)) / elapsed_sec:.2f}x speedup.\n')

    # print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))

    set_dir_from_config(config)
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
        if config.task_config.quad_type == 4:
            system = 'quadrotor_2D'
        if config.task_config.quad_type == 6:
            system = 'quadrotor_3D'
    else:
        system = config.task

    if True:
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

        _, ax = plt.subplots()
        # ax.plot(trajs_data['obs'][0][:, graph1_1], trajs_data['obs'][0][:, graph1_2], 'r--', label='Agent Trajectory')
        # ax.scatter(trajs_data['obs'][0][0, graph1_1], trajs_data['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        # ax.set_xlabel(r'$\theta$')
        # ax.set_ylabel(r'$\dot{\theta}$')
        # ax.set_box_aspect(0.5)
        # ax.legend(loc='upper right')
        # # save the plot
        # plt.savefig(os.path.join(config.output_dir, 'trajectory_theta_theta_dot.png'))

        # if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
        #     _, ax2 = plt.subplots()
        #     ax2.plot(np.linspace(0, 20, trajs_data['obs'][0].shape[0]), trajs_data['obs'][0][:, 0], 'r--', label='Agent Trajectory')
        #     ax2.plot(np.linspace(0, 20, trajs_data['obs'][0].shape[0]), ctrl.env.X_GOAL[:, 0], 'b', label='Reference')
        #     ax2.set_xlabel(r'Time')
        #     ax2.set_ylabel(r'X')
        #     ax2.set_box_aspect(0.5)
        #     ax2.legend(loc='upper right')
        #     # save the plot
        #     plt.savefig(os.path.join(config.output_dir, 'trajectory_time_x.png'))
        # elif config.task == Environment.QUADROTOR:
        #     _, ax2 = plt.subplots()
        #     ax2.plot(trajs_data['obs'][0][:, graph3_1 + 1], trajs_data['obs'][0][:, graph3_2 + 1], 'r--', label='Agent Trajectory')
        #     ax2.set_xlabel(r'x_dot')
        #     ax2.set_ylabel(r'z_dot')
        #     ax2.set_box_aspect(0.5)
        #     ax2.legend(loc='upper right')
        #     # save the plot
        #     plt.savefig(os.path.join(config.output_dir, 'trajectory_x_dot_z_dot.png'))

        _, ax3 = plt.subplots()
        ax3.plot(trajs_data['obs'][0][:, graph3_1], trajs_data['obs'][0][:, graph3_2], 'r--', label='Agent Trajectory')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(ctrl.env.X_GOAL[:, graph3_1], ctrl.env.X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.scatter(trajs_data['obs'][0][0, graph3_1], trajs_data['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        # save the plot
        plt.savefig(os.path.join(config.output_dir, 'trajectory_x.png'))


if __name__ == '__main__':
    run()
