

for ADDITIOANL in '_dw_h=1dot5' '_dw_h=2' '_dw_h=2dot5' '_dw_h=3'
# for ADDITIOANL in '_ob_ns=5' '_ob_ns=15' '_ob_ns=25'  
# for ADDITIOANL in  '_proc_ns=25' # '_proc_ns=5' '_proc_ns=15'
# for ADDITIOANL in '_tr'
do
    for STARTSEED in 1
    do 
        for algo in 'gpmpc_acados_TP'
        do
            python3 gpmpc_experiment.py $algo $ADDITIOANL
        done
    done

    # python3 results_dw.py 'gpmpc_acados_TP' $GP_TAG
done

python3 ../del_acados_files.py