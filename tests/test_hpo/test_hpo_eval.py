import os
import sys

import pytest

from examples.hpo.hpo_experiment import eval
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT


@pytest.mark.parametrize('SYS', ['quadrotor_2D_attitude'])
@pytest.mark.parametrize('TASK', ['tracking'])
@pytest.mark.parametrize('ALGO', ['pid', 'lqr', 'ilqr', 'mpc_acados', 'ppo'])
@pytest.mark.parametrize('PRIOR', [''])
@pytest.mark.parametrize('SAFETY_FILTER', ['', 'linear_mpsc'])
def test_hpo_eval(SYS, TASK, ALGO, PRIOR, SAFETY_FILTER):
    '''Test HPO evaluation.
    '''

    # output_dir
    output_dir = f'./examples/hpo/results/{ALGO}'
    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')

    SYS_NAME = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS

    if ALGO == 'pid' or ALGO == 'lqr' or ALGO == 'ilqr' or ALGO == 'gpmpc_acados' or ALGO == 'linear_mpc' or ALGO == 'mpc_acados' or ALGO == 'gpmpc_acados_TP':
        PRIOR = '100'
    elif ALGO == 'gp_mpc':
        PRIOR = '200'

    # check if the config file exists
    TASK_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{SYS}_{TASK}_eval.yaml'
    ALGO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'
    HPO_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{ALGO}_{SYS}_hpo.yaml'
    assert os.path.exists(TASK_CONFIG_PATH), f'{TASK_CONFIG_PATH} does not exist'
    assert os.path.exists(ALGO_CONFIG_PATH),  f'{ALGO_CONFIG_PATH} does not exist'
    assert os.path.exists(HPO_CONFIG_PATH),  f'{HPO_CONFIG_PATH} does not exist'

    if SAFETY_FILTER == 'linear_mpsc':
        if ALGO != 'ilqr':
            pytest.skip('SAFETY_FILTER is only supported for ilqr')
            raise ValueError('SAFETY_FILTER is only supported for ilqr')
        SAFETY_FILTER_CONFIG_PATH = f'./examples/hpo/{SYS_NAME}/config_overrides/{SAFETY_FILTER}_{SYS}_{TASK}_{PRIOR}.yaml'
        assert os.path.exists(SAFETY_FILTER_CONFIG_PATH), f'{SAFETY_FILTER_CONFIG_PATH} does not exist'
        MPSC_COST = 'one_step_cost'
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                            SAFETY_FILTER_CONFIG_PATH,
                        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}',
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]
    else:
        sys.argv[1:] = ['--algo', ALGO,
                        '--task', SYS_NAME,
                        '--overrides',
                            TASK_CONFIG_PATH,
                            ALGO_CONFIG_PATH,
                            HPO_CONFIG_PATH,
                        '--output_dir', output_dir,
                        '--seed', '7',
                        '--use_gpu', 'True'
                        ]

    fac = ConfigFactory()
    fac.add_argument('--load_study', type=bool, default=False, help='whether to load study from a previous HPO.')
    config = fac.merge()
    config.hpo_config.repetitions = 1

    eval(config)

    # delete output_dir
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')