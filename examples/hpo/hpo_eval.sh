#!/bin/bash

cd ~/safe-control-gym

parallel_jobs=$1 # Number of parallel jobs
localOrHost=$2
sys=$3 # cartpole, or quadrotor_2D_attitude
sys_name=${sys%%_*} # cartpole, or quadrotor
algo=$4
prior=$5
safety_filter=$6 # True or False
task=$7 # stab, or tracking
FOLDER="./examples/hpo/hpo/${algo}"
OUTPUT_DIR=(${FOLDER})

# activate the environment
if [ "$localOrHost" == 'local' ]; then
    source /home/tsung/anaconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'host0' ]; then
    source /home/tueilsy-st01/anaconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'hostx' ]; then
    source /home/tsung/miniconda3/etc/profile.d/conda.sh
    conda activate safe
elif [ "$localOrHost" == 'cluster' ]; then
    echo "Doing experiment in cluster..."
else
    echo "Please specify the machine to run the experiment."
    exit 1
fi

# echo config path
echo "task config path: ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}_eval.yaml"
echo "algo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml"
echo "hpo config path: ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml"

# training unseen seeds that are unseen during hpo (hpo only saw seeds in [0, 10000])
seeds=(22403 84244)
# seeds=(22403 84244 98825 40417 58454 47838 56715 77833 19880 59009 
#        47722 81354 63825 13296 10779 98122 86221 89144 35192 24759)

count=0

for seed in "${seeds[@]}"; do

    # Run the process in the background
    python ./examples/hpo/hpo_experiment.py \
        --algo "${algo}" \
        --task "${sys_name}" \
        --overrides ./examples/hpo/${sys_name}/config_overrides/${sys}_${task}_eval.yaml \
                    ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_${task}_${prior}.yaml \
                    ./examples/hpo/${sys_name}/config_overrides/${algo}_${sys}_hpo.yaml \
        --output_dir "${OUTPUT_DIR}" \
        --seed "${seed}" \
        --func eval \
        --use_gpu True &

    # Increment count
    count=$((count + 1))

    # Check if we have hit the limit of parallel jobs
    if (( count % parallel_jobs == 0 )); then
        # Wait for all background jobs to finish before continuing
        wait
    fi

done