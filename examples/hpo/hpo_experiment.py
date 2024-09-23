'''Template hyperparameter optimization/hyperparameter evaluation script.'''
import os
from functools import partial

import yaml

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.hyperparameters.hpo_optuna import HPO as HPO_optuna
from safe_control_gym.hyperparameters.hpo_vizier import HPO as HPO_vizier
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_dir_from_config, set_seed_from_config


def hpo(config):
    '''Hyperparameter optimization.

    Usage:
        * to start HPO, use with `--func hpo`.
    '''

    # change the cost function for rl methods
    if config.algo == 'ppo':
        config.task_config.cost = 'rl_reward'
        config.task_config.obs_goal_horizon = 1
        config.normalized_rl_action_space = True
        if 'disturbances' in config.task_config:
            if config.task_config.disturbances is not None:
                raise ValueError('Double check with this setup.')
                config.task_config.disturbances.observation[0]['std'] += [0, 0, 0, 0, 0, 0]
        config.algo_config.log_interval = 10000000
        config.algo_config.eval_interval = 10000000
    elif config.algo == 'gp_mpc' or config.algo == 'ilqr':
        config.task_config.info_in_reset = False
    else:
        raise ValueError('Only ppo, gp_mpc, gpmpc_acados, and ilqr are supported for now.')
    
    # Experiment setup.
    if config.hpo_config.hpo:
        set_dir_from_config(config)
    set_seed_from_config(config)
    set_device_from_config(config)

    # initialize safety filter
    if 'safety_filter' not in config:
        config.safety_filter = None
        config.sf_config = None

    # HPO
    if config.sampler == 'optuna':
        hpo = HPO_optuna(config.algo,
                        config.task,
                        config.load_study,
                        config.output_dir,
                        config.task_config,
                        config.hpo_config,
                        config.algo_config,
                        config.safety_filter,
                        config.sf_config,
                        )
    elif config.sampler == 'vizier':
        hpo = HPO_vizier(config.algo,
                        config.task,
                        config.load_study,
                        config.output_dir,
                        config.task_config,
                        config.hpo_config,
                        config.algo_config,
                        config.safety_filter,
                        config.sf_config,
                        )
    else:
        raise ValueError('Only Optuna and Vizier are supported for now.')


    if config.hpo_config.hpo:
        hpo.hyperparameter_optimization()
        print('Hyperparameter optimization done.')


MAIN_FUNCS = {'hpo': hpo}


if __name__ == '__main__':
    # Make config.
    fac = ConfigFactory()
    fac.add_argument('--func', type=str, default='hpo', help='main function to run.')
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    fac.add_argument('--sampler', type=str, default='optuna', help='which package to use in HPO.')
    fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
    # merge config
    config = fac.merge()

    # Execute.
    func = MAIN_FUNCS.get(config.func, None)
    if func is None:
        raise Exception(f'Main function {config.func} not supported.')
    func(config)
