
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from safe_control_gym.utils.configuration import ConfigFactory
from functools import partial
from safe_control_gym.utils.registration import make

# get the config
ALGO = 'mpc_acados'
SYS = 'quadrotor_2D_attitude'
TASK = 'tracking'
# PRIOR = '200'
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
print('X_GOAL.shape', X_GOAL.shape)

# get the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

script_path = os.path.dirname(os.path.realpath(__file__))
gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_rti/temp'
# gp_model_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/gpmpc_acados/results/200_300_rti/temp'
# get all directories in the gp_model_path
gp_model_dirs = [d for d in os.listdir(gp_model_path) if os.path.isdir(os.path.join(gp_model_path, d))]
gp_model_dirs = [os.path.join(gp_model_path, d) for d in gp_model_dirs]

traj_data_name = 'gpmpc_acados_data_quadrotor_traj_tracking.pkl'
data_name = [os.path.join(d, traj_data_name) for d in gp_model_dirs]

# print(data_name)
# data = np.load(data_name[0], allow_pickle=True)
# print(data.keys())
# print(data['trajs_data'].keys())
# print(data['trajs_data']['obs'][0].shape) # (541, 6)
data = []
for d in data_name:
    data.append(np.load(d, allow_pickle=True))
traj_data = [d['trajs_data']['obs'][0] for d in data]
traj_data = np.array(traj_data)
print(traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
mean_traj_data = np.mean(traj_data, axis=0)
print(mean_traj_data.shape) # (mean_541, 6)

### plot the ilqr data
ilqr_data_path = '/home/mingxuan/Repositories/scg_tsung/benchmarking_sim/quadrotor/ilqr/results/temp'
ilqr_data_dirs = [d for d in os.listdir(ilqr_data_path) if os.path.isdir(os.path.join(ilqr_data_path, d))]
ilqr_traj_data_name = 'ilqr_data_quadrotor_traj_tracking.pkl'
ilqr_traj_data_name = [os.path.join(d, ilqr_traj_data_name) for d in ilqr_data_dirs]

ilqr_data = []
for d in ilqr_traj_data_name:
    ilqr_data.append(np.load(os.path.join(ilqr_data_path, d), allow_pickle=True))
ilqr_traj_data = [d['trajs_data']['obs'][0] for d in ilqr_data]
ilqr_traj_data = np.array(ilqr_traj_data)
print(ilqr_traj_data.shape) # (10, 541, 6) seed, time_step, obs
# take average of all seeds
ilqr_mean_traj_data = np.mean(ilqr_traj_data, axis=0)
print(ilqr_mean_traj_data.shape) # (mean_541, 6)


# plot the state path x, z [0, 2]
mean_points = mean_traj_data[:, [0, 2]]
mean_points_ilqr = ilqr_mean_traj_data[:, [0, 2]]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(X_GOAL[:, 0], X_GOAL[:, 2], color='gray', linestyle='-.', label='Reference')
ax.plot(mean_points[:,0], mean_points[:,1], label='GP-MPC', color=colors[0])
ax.plot(mean_points_ilqr[:,0], mean_points_ilqr[:,1], label='iLQR', color=colors[2])
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$z$ [m]')
ax.set_title('State path in $x$-$z$ plane')
# set the super title
fig.suptitle('Figure-eight reference tracking', fontsize=18)
fig.tight_layout()
ax.legend()

# plot the convex hull of each steps
k = 1.1 # padding factor
gpmpc_hull_color = 'lightskyblue'
ilqr_hull_color = 'lightgreen'

for i in range(traj_data.shape[1] - 1):
# for i in range(1):
    points_of_step = traj_data[:, i, [0, 2]]
    hull = ConvexHull(points_of_step)
    cent = np.mean(points_of_step, axis=0)
    # print('cent', cent)
    pts = points_of_step[hull.vertices]
    # print(pts.shape)
    poly = Polygon(k*(pts - cent) + cent, closed=True,
                   capstyle='round', facecolor=gpmpc_hull_color, alpha=1.0)
    ax.add_patch(poly)
    # plt.gca().add_patch(poly)

    # also connect the points of the next step
    points_of_next_step = traj_data[:, i+1, [0, 2]]
    points_all = np.concatenate((points_of_step, points_of_next_step), axis=0)
    hull_all = ConvexHull(points_all)
    cent_all = np.mean(points_all, axis=0)
    pts_all = points_all[hull_all.vertices]
    poly_all = Polygon(k*(pts_all - cent_all) + cent_all, closed=True,
                   capstyle='round', facecolor=gpmpc_hull_color, alpha=1.0)
    ax.add_patch(poly_all)

    # ilqr
    points_of_step_ilqr = ilqr_traj_data[:, i, [0, 2]]
    hull_ilqr = ConvexHull(points_of_step_ilqr)
    cent_ilqr = np.mean(points_of_step_ilqr, axis=0)
    pts_ilqr = points_of_step_ilqr[hull_ilqr.vertices]

    poly_ilqr = Polygon(k*(pts_ilqr - cent_ilqr) + cent_ilqr, closed=True,
                   capstyle='round', facecolor=ilqr_hull_color, alpha=1.0)
    ax.add_patch(poly_ilqr)

    points_of_next_step_ilqr = ilqr_traj_data[:, i+1, [0, 2]]
    points_all_ilqr = np.concatenate((points_of_step_ilqr, points_of_next_step_ilqr), axis=0)
    hull_all_ilqr = ConvexHull(points_all_ilqr)
    cent_all_ilqr = np.mean(points_all_ilqr, axis=0)
    pts_all_ilqr = points_all_ilqr[hull_all_ilqr.vertices]
    poly_all_ilqr = Polygon(k*(pts_all_ilqr - cent_all_ilqr) + cent_all_ilqr, closed=True,
                   capstyle='round', facecolor=ilqr_hull_color, alpha=1.0)
    ax.add_patch(poly_all_ilqr)









fig.savefig(os.path.join(script_path, 'xz_path_performance.png'))


