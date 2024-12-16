import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# plot the reference trajectory
script_dir = os.path.dirname(__file__)

ref_file = os.path.join(script_dir, 'ilqr_ref_traj.npy')
ref_traj = np.load(ref_file, allow_pickle=True).item()

print(ref_traj.keys())

# 6 states: x, z, vx, vz, theta, omega
# 2 control inputs: F, theta_cmd
#
nx = 6
nu = 2
state_labels = ['x', 'vx', 'z', 'vz', 'theta', 'omega']
control_labels = ['F', 'theta_cmd']

fig, axs = plt.subplots(6, 1, figsize=(8, 12))
for i in range(nx):
    axs[i].plot(ref_traj['obs'][0][:, i], label=f'{state_labels[i]}')
    axs[i].legend()

fig, axs = plt.subplots(2, 1, figsize=(8, 4))
for i in range(nu):
    axs[i].plot(ref_traj['current_physical_action'][0][:, i], label=f'{control_labels[i]}')
    axs[i].legend()

# plot x-z path
plt.figure(figsize=(8, 8))
plt.plot(ref_traj['obs'][0][:, 0], ref_traj['obs'][0][:, 2])
plt.xlabel('x')
plt.ylabel('z')
plt.title('x-z path')

plt.show()