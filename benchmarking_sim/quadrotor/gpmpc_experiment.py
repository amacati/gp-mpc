
import os
import sys

import numpy as np

from benchmarking_sim.quadrotor.mb_experiment import run

script_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    # runtime_list = []
    # num_seed = 3
    # start_seed = 7
    # for seed in range(start_seed, num_seed + start_seed):
    # # for seed in [9, 10]:
    #     run(seed=seed)
    #     runtime_list.append(run.elapsed_time)
    # print(f'Average runtime for {num_seed} runs: \
    #       {np.mean(runtime_list):.3f} sec')

    runtime_list = []
    num_seed = 1
    start_seed = 1 # [1, 5, 6, 8, 9, 11, 12]
    suceeded = 0
    seed = start_seed
    while suceeded < num_seed:  
        if seed > 2:
            print(f'Only {suceeded} out of {num_seed} runs succeeded')
            break
        try:
            sys.argv[1:] = ['gpmpc_acados_TP', # specify the controller
                            # '_dwr',
                            # '_obsr',
                            '_procr'
                            ]
            run(seed=seed)
            runtime_list.append(run.elapsed_time)
            suceeded += 1
        except Exception as e:
            # log the error and complete trackback
            print('Error:', e)
            # the error and complete trackback
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            # dump the error to a file
            with open(f'./error_{seed}.txt', 'w') as f:
                f.write(f'Error: {e}\n')
                f.write(f'{exc_type} {fname} {exc_tb.tb_lineno}\n')
        seed += 1

    print(f'Average runtime for {num_seed} runs: \
          {np.mean(runtime_list):.3f} sec')

