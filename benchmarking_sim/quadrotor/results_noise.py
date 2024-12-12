import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import munch

from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
s = 2 # times of std

# roll out
# additionals_list = ['', '_slow', '_fast']
# additionals_list = ['_'+repr(i) for i in range(9, 16)]
# start_seed = [i for i in range(1, 100, 10)]
# for algo in ['pid', 'lqr', 'ilqr']:
    # for additional in additionals_list:
        # for start_seed in start_seed:

# additional = sys.argv[1]
# start_seed = int(sys.argv[2])
# algo = sys.argv[3]
# noise_factor = int(sys.argv[4])
algo = sys.argv[1]

# task_description = munch.munchify({
#     'additional': additional,
#     'algo': algo,
#     'eval_task': 'rollout',
#     'start_seed': start_seed,
#     })
# run_rollouts(task_description)

# noise factor test
additional = ''
noise_factor_list = np.arange(1, 201, 10)
noise_factor_list[0] = 1
# for algo in ['pid', 'lqr', 'ilqr']:
for noise_factor in noise_factor_list:
    for start_seed in range(1, 11):
        task_description = munch.munchify({
            'additional': additional,
            'algo': algo,
            'noise_factor': noise_factor,
            'eval_task': 'noise',
            'num_seed': 1,
            'start_seed': start_seed,
            })
        run_rollouts(task_description)