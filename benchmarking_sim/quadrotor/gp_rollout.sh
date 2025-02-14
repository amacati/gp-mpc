
for GP_TAG in '100_200_proc_ns=5' '100_200_proc_ns=15' # '100_200_proc_ns=25'
# for GP_TAG in '100_200_ob_ns=5' '100_200_ob_ns=15' # '100_200_ob_ns=25'
# for GP_TAG in '100_200_dw_h=1dot5' '100_200_dw_h=2dot5' # '100_200_dw_h=4'
# for GP_TAG in '100_200_tr'
do
    # for ADDITIOANL in '_11'
    for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
    do
        for STARTSEED in 1 11 21 31 41 # 51 61 71 81 91 
        do 
            # for algo in 'lqr' 'ilqr' # 'pid' 
            # for algo in 'linear_mpc' 
            # for algo in 'ilqr' # 'lqr'
            # for algo in 'pid'
            for algo in 'gpmpc_acados_TP'
            do
                python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
            done
        done
    done

    python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG

    python3 results_noise.py 'gpmpc_acados_TP' $GP_TAG
done
