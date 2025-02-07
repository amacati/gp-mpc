#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='dppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_DATA='nominal'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

NS=1
T=11
H=0

# RL Experiment
python3 ./rl_experiment.py \
            --task ${SYS_NAME} \
            --algo ${ALGO} \
            --use_gpu \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --seed 0 \
            --kv_overrides \
                algo_config.training=False \
                task_config.normalized_rl_action_space=False \
                task_config.randomized_init=True \
                task_config.task_info.num_cycles=2 \
                task_config.episode_len_sec=${T} \
                task_config.noise_scale=${NS} \
                task_config.downwash_height=${H} \
                task_config.external_param=${EP}