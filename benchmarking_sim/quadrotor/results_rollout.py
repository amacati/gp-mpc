import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import munch

from benchmarking_sim.quadrotor.benchmark_util.utils import run_rollouts

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)

additional = sys.argv[1]
start_seed = int(sys.argv[2])
algo = sys.argv[3]
gp_model_tag = sys.argv[4]
# try:
#     gp_model_tag = sys.argv[4]
# except:
#     gp_model_tag = '100_200'

# noise_factor = int(sys.argv[4])

task_description = munch.munchify({
    'additional': additional,
    'algo': algo,
    'eval_task': 'rollout',
    'start_seed': start_seed,
    # 'SYS': 'quadrotor_3D_attitude',
    'SYS': 'quadrotor_2D_attitude',
    'gp_model_tag': gp_model_tag,
    })
run_rollouts(task_description)
