

# for STARTSEED in 1 21 41 61 81
# for STARTSEED in 40
# do 
#     for ADDITIOANL in '_11' # '_10' '_12' '_13' '_14'
#     do
#         python3 mb_experiment_rollout.py $STARTSEED 20 $ADDITIOANL 'gpmpc_acados'
#     done
# done

# for algo in 'mpc_acados' # 'linear_mpc' 
# do
#     for STARTSEED in 21 41 61 81
#     do 
#         for ADDITIOANL in  '_13' '_14' # '_10' '_12'
#         do
#         python3 mb_experiment_rollout.py $STARTSEED 20 $ADDITIOANL $algo
#         done
#     done
# done

# for ADDITIOANL in '_9' '_11' '_13' '_15'
# for ADDITIOANL in '_10' '_12' '_14'
for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
do
    for STARTSEED in 1 11 21 31 41 51 61 71 81 91 
    do 
        # for algo in 'lqr' 'ilqr' # 'pid' 
        # for algo in 'linear_mpc' 
        for algo in 'ilqr' # 'lqr'
        do
            python3 results_rollout.py $ADDITIOANL $STARTSEED $algo 1
        done
    done
done

for ADDITIOANL in ''
do
    for STARTSEED in 1 
    do 
        for algo in 'ilqr'
        # for algo in 'linear_mpc_acados'
        do
            python3 results_noise.py $algo
        done
    done
done