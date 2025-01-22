

for ADDITIOANL in ''
do
    for STARTSEED in 1 
    do 
        for algo in 'pid'
        do
            python3 results_dw.py $algo
        done
    done
done