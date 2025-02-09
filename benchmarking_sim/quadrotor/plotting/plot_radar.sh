
for algo in 'iLQR' 'PID' 'LQR' \
            'GP-MPC' 'Nonlinear-MPC' 'Linear-MPC' 'F-MPC' \
            'PPO' 'SAC' 'DPPO' 

do
    python3 plot_radar.py $algo
done 