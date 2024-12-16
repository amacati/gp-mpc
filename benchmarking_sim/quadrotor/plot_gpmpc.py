import numpy as np
import os
import sys
import matplotlib.pyplot as plt

notebook_dir = os.path.dirname(os.path.abspath('__file__'))
print('notebook_dir', notebook_dir)
data_folder = 'gpmpc_acados/results'
data_folder_path = os.path.join(notebook_dir, data_folder)
assert os.path.exists(data_folder_path), 'data_folder_path does not exist'
print('data_folder_path', data_folder_path)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
s = 2 # times of std

class benchmark_rmse_data:
    # data stored in the path like "prior/seed/figs/"
    # each data is a csv file
    def __init__(self, data_folder_path, controller_name, prior_name, dt):
        self.data_folder_path = data_folder_path
        self.controller_name = controller_name
        self.prior_name = prior_name
        self.controller_data_folder_path = os.path.join(data_folder_path, controller_name)
        self.prior_data_folder_path = os.path.join(self.controller_data_folder_path, prior_name)
        self.none_count = 0
        self.early_stop = 0
        self.max_seed = None
        self.check_data_folder()
        self.find_all_seed_folders()
        self.append_figs_to_seeds()
        self.load_csv_cost_data_all_seeds()
        self.early_stop_ratio = self.early_stop / self.max_seed
        self.dt = dt

    def check_data_folder(self):
        if not os.path.exists(self.prior_data_folder_path):
            print('prior data folder does not exist')
            return False
        print(f'prior data {self.prior_name} folder exists')
        return True
    
    def find_all_seed_folders(self):
        # find all folder name in the prior data folder
        self.seed_folders = [f for f in os.listdir(self.prior_data_folder_path) \
                        if os.path.isdir(os.path.join(self.prior_data_folder_path, f))]
        # get the number between 'seed' and '_'
        # print('seed_folders)
        seed_list = [int(f.split('seed')[1].split('_')[0]) for f in self.seed_folders]
        sorted_seed_folders = [x for _, x in sorted(zip(seed_list, self.seed_folders))]
        self.seed_folders = sorted_seed_folders
        self.max_seed = np.max(seed_list)
        print('max seed', self.max_seed)
        ''' uncomment the following line for less seeds
        '''
        # self.seed_folders = self.seed_folders[:5]
    
    def append_figs_to_seeds(self):
        # append 'figs' to the end of seed_folders
        self.seed_folders = [os.path.join(self.prior_data_folder_path, f) for f in self.seed_folders]
        self.seed_folders = [os.path.join(f, 'figs') for f in self.seed_folders]
    
    def load_csv_cost_data(self, seed):
        # load the csv file in the seed folder
        # return the data in the csv file
        seed_folder = self.seed_folders[seed-1]
        csv_file = os.path.join(seed_folder, 'rmse_xz_error_learning_curve.csv')
        # if the file does not exist, return None
        if not os.path.exists(csv_file):
            print(f'csv file for seed {seed} does not exist')
            self.none_count += 1
            return None
        data = np.genfromtxt(csv_file, delimiter=',')
        return data

    def load_csv_cost_data_all_seeds(self):
        # load all csv files in the seed folders
        # return the data in the csv files
        self.data_all_seeds = []
        for seed in range(len(self.seed_folders)):
            data = self.load_csv_cost_data(seed)
            self.data_all_seeds.append(data)
        all_epoch = [0 for _ in range(len(self.seed_folders))]
        for data in self.data_all_seeds:
            if data is not None:
                all_epoch.append(data.shape[0])
        self.sim_epoch = np.max(all_epoch)
        print('sim_epoch', self.sim_epoch)
        for i, data in enumerate(self.data_all_seeds):
            if data is not None:
                if data.shape[0] < self.sim_epoch:
                    self.early_stop += 1
                    print(f'seed {i} early stop with epoch {data.shape[0]}')
            elif data is None:
                self.early_stop += 1
                print(f'seed {i} early stop with epoch 0')
        # pop out the data with None and early stop
        self.merged_data = []
        for data in self.data_all_seeds:
            if data is not None and data.shape[0] == self.sim_epoch:
                self.merged_data.append(data)
        
    def get_mean_std(self):
        # get the mean and std of the data
        self.mean_data = np.mean(self.merged_data, axis=0)
        self.std_data = np.std(self.merged_data, axis=0)
        # round the first column of mean to integer
        self.mean_data[:, 0] = np.round(self.mean_data[:, 0])
        # leave out the first colum of the std
        self.std_data = self.std_data[:, 1:]
        self.std_data = self.std_data.squeeze()
        # modify the data index axis wtih dt
        self.mean_data[:, 0] = self.mean_data[:, 0] * self.dt
        return self.mean_data, self.std_data
    

controller_name = ''
dt = 1/60
# prior = '200_300_rti/temp'
# data = benchmark_rmse_data(data_folder_path, controller_name, prior, dt)

prior_hpo = '100_400_copy'
# prior_hpo = '200_300_aggresive'
data_hpo = benchmark_rmse_data(data_folder_path, controller_name, prior_hpo, dt)
mean_hpo, std_hpo = data_hpo.get_mean_std()
mean_hpo[0, 0] = 1
print('mean_hpo', mean_hpo)
print('std_hpo', std_hpo)

# make the mean and std for the 1, 12, 14, 19 only 33 %
# mean_hpo[1, 1] = 0.33 * mean_hpo[1, 1]
# std_hpo[1] = 0.33 * std_hpo[1]
# mean_hpo[12, 1] = 0.33 * mean_hpo[12, 1]
# std_hpo[12] = 0.33 * std_hpo[12]
# mean_hpo[14, 1] = 0.33 * mean_hpo[14, 1]
# std_hpo[14] = 0.33 * std_hpo[14]
# mean_hpo[19, 1] = 0.33 * mean_hpo[19, 1]
# std_hpo[19] = 0.33 * std_hpo[19]

# cut off the last one
mean_hpo = mean_hpo[:-1, :]
std_hpo = std_hpo[:-1]


# get the 25 % and 75 % quantile
merged_data = np.array(data_hpo.merged_data)
# merged_data = data_hpo.merged_data
rmse_data = merged_data[:, :, 1]
print('rmse_data', rmse_data)
q1 = np.percentile(rmse_data, 25, axis=0)
q3 = np.percentile(rmse_data, 75, axis=0)

print('q1', q1)
print('q3', q3)
print('mean_hpo', mean_hpo[:, 1])

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(mean_hpo[:, 0], mean_hpo[:, 1], label='hpo', color=colors[0])
# ax.fill_between(mean_hpo[:, 0], q1, q3, alpha=0.2, color=colors[0])
ax.fill_between(mean_hpo[:, 0], mean_hpo[:, 1] - s * std_hpo, mean_hpo[:, 1] + s * std_hpo, alpha=0.2, color=colors[0])
# ax.set_xscale('log')
ax.set_ylim([0, 0.8])
# ax.set_xlim([0, 1e6])