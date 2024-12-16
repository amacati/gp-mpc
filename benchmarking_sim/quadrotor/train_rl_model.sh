#!/bin/bash

SYS='quadrotor_2D_attitude'
TASK='tracking'

ALGO='ppo'


if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
rm -r -f ./unsafe_rl_temp_data/

# Train the unsafe controller/agent.
python3 ../../safe_control_gym/experiments/train_rl_controller.py \
    --algo ${ALGO} \
    --task ${SYS_NAME} \
    --overrides \
        ./config_overrides/${ALGO}_${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}_${TASK}.yaml \
    --output_dir ./unsafe_rl_temp_data/ \
    --seed 2 \
    --kv_overrides \
        task_config.init_state=None \
        task_config.randomized_init=True \
        algo_config.pretrained=./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt

# Move the newly trained unsafe model.
# mv ./unsafe_rl_temp_data/model_best.pt ./models/${ALGO}/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
# rm -r -f ./unsafe_rl_temp_data/
