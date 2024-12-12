#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
# SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

ALGO='ppo'
#ALGO='sac'
#ALGO='dppo'
# ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

NS=1
T=11

# RL Experiment
#for N in {1}:
for NS in {0,1,5,10,15,20,25}
#for T in {9,10,11,12,13,14,15}
do
  for SEED in {0..0}
  do
    python3 ./rl_experiment.py \
      --task ${SYS_NAME} \
      --algo ${ALGO} \
      --use_gpu \
      --overrides \
          ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
          ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
      --experiment_type 'performance' \
      --seed ${SEED} \
      --kv_overrides \
          algo_config.training=False \
          task_config.normalized_rl_action_space=False \
          task_config.randomized_init=True \
          task_config.task_info.num_cycles=2 \
          task_config.episode_len_sec=${T} \
          task_config.noise_scale=${NS} \
      --pretrain_path ./Results/New_batch/default_param/${SYS}_${ALGO}_data4/seed${SEED}_*/
  done
done