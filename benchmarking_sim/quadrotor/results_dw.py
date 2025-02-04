import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import munch

from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)

# additional = sys.argv[1]
# start_seed = int(sys.argv[2])
# algo = sys.argv[3]
# noise_factor = int(sys.argv[4])
algo = sys.argv[1]

# noise factor test
# additional = ''
additional = '_downwash'
# noise_factor = 1
# for dw_height_scale in np.arange(0.0, 1.0, 0.05):
for dw_height in np.arange(1.5, 4.0, 0.2):
    for start_seed in range(1, 11):
    # for start_seed in range(1, 2): # for testing
        task_description = munch.munchify({
            'additional': additional,
            'algo': algo,
            # 'noise_factor': noise_factor,
            # 'dw_height_scale': dw_height_scale,
            'dw_height': dw_height,
            'eval_task': 'downwash',
            'num_seed': 1,
            'start_seed': start_seed,
            'SYS': 'quadrotor_2D_attitude',
            })
        run_rollouts(task_description)