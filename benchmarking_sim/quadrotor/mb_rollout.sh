
# for algo in 'ilqr'
# for algo in 'pid'
for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
do
    python3 results_noise.py $algo 'obs_noise'
    python3 results_noise.py $algo 'proc_noise'

for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
do
    python3 results_dw.py $algo
done

done
# for ADDITIOANL in '_9' '_11' '_13' '_15'
# for ADDITIOANL in '_10' '_12' '_14'
for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
# for ADDITIOANL in '_11'
do
    for STARTSEED in 1 11 21 31 41 # 51 61 71 81 91 
    do 
        # for algo in 'lqr' 'ilqr' # 'pid' 
        # for algo in 'linear_mpc' 
        # for algo in 'ilqr' # 'lqr'
        # for algo in 'pid'
        for algo in 'mpc_acados' 'fmpc' 'linear_mpc_acados' 'lqr' 'ilqr' 'pid'
        do
            python3 results_rollout.py $ADDITIOANL $STARTSEED $algo 1
        done
    done
done
