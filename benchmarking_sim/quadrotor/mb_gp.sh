

for RAND_TYPE in '_dw_h=1dot5' '_dw_h=2' '_dw_h=2dot5' '_dw_h=3'
# for RAND_TYPE in '_ob_ns=5' '_ob_ns=15' '_ob_ns=25'  
# for RAND_TYPE in  '_proc_ns=25' # '_proc_ns=5' '_proc_ns=15'
# for RAND_TYPE in '_tr'
do
    for STARTSEED in 1
    do 
        for algo in 'gpmpc_acados_TP'
        do
            python3 gpmpc_experiment.py $algo $RAND_TYPE
        done
    done

    # python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
    # python3 results_noise.py 'gpmpc_acados_TP' $GP_TAG 'obs_noise'
    # python3 results_noise.py 'gpmpc_acados_TP' $GP_TAG 'proc_noise'
    # for ADDITIOANL in '_9' '_10' '_11' '_12' '_13' '_14' '_15'
    # do
    #     for STARTSEED in 1 11 21 31 41 # 51 61 71 81 91 
    #     do 
    #         for algo in 'gpmpc_acados_TP'
    #         do
    #             python3 results_rollout.py $ADDITIOANL $STARTSEED $algo $GP_TAG
    #         done
    #     done
    # done
done

python3 ../del_acados_files.py