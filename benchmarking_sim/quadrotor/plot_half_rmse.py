import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make

script_dir = os.path.dirname(os.path.abspath('__file__'))
print('script_dir', script_dir)
data_folder = 'gpmpc_acados_TP/results/100_200/temp'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# get the subfolders in the data_folder
data_folder_path = os.path.join(script_dir, data_folder)
assert os.path.exists(data_folder_path), 'data_folder_path does not exist'
print('data_folder_path', data_folder_path)
subfolders = [f.path for f in os.scandir(data_folder_path) if f.is_dir()]
print('subfolders', subfolders)



# get the config
ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
# PRIOR = '200_hpo'
PRIOR = '100'
agent = 'quadrotor' if SYS == 'quadrotor_2D' or SYS == 'quadrotor_2D_attitude' else SYS
SAFETY_FILTER = None

# check if the config file exists
assert os.path.exists(f'./config_overrides/{SYS}_{TASK}.yaml'), f'./config_overrides/{SYS}_{TASK}.yaml does not exist'
assert os.path.exists(f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml'), f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml does not exist'
if SAFETY_FILTER is None:
    sys.argv[1:] = ['--algo', ALGO,
                    '--task', agent,
                    '--overrides',
                        f'./config_overrides/{SYS}_{TASK}.yaml',
                        f'./config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}.yaml',
                    '--seed', '2',
                    '--use_gpu', 'True',
                    '--output_dir', f'./{ALGO}/results',
                        ]
fac = ConfigFactory()
fac.add_argument('--func', type=str, default='train', help='main function to run.')
fac.add_argument('--n_episodes', type=int, default=1, help='number of episodes to run.')
# merge config and create output directory
config = fac.merge()
# Create an environment
env_func = partial(make,
                    config.task,
                    seed=config.seed,
                    **config.task_config
                    )
random_env = env_func(gui=False)
X_GOAL = random_env.X_GOAL

# load the pkl files
res = []
for subfolder in subfolders:
    data_path = os.path.join(subfolder, 'gpmpc_acados_TP_data_quadrotor_traj_tracking.pkl')
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        res.append(data)
    else:
        print(f'{data_path} does not exist')

traj_data = [res[i]['trajs_data']['obs'][0] for i in range(len(res))]

# compute tracking error
error = []
for i in range(len(traj_data)):
    error.append(np.linalg.norm(traj_data[i][:, [0,2]] - X_GOAL[:-1, [0,2]], axis=1))
error = np.array(error)

# compute rmse
rmse = np.sqrt(np.mean(error**2, axis=1))
rmse_half = np.sqrt(np.mean(error[:, int(error.shape[1]/2):]**2, axis=1))