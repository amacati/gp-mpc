import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

script_dir = os.path.dirname(__file__)

# get the pyplot default color wheel
prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# plot_colors = {
#     'GP-MPC': colors[0],
#     'PPO': colors[1],
#     'SAC': colors[3],
#     # 'iLQR': 'darkgray',
#     'DPPO': colors[6],
#     'Linear-MPC': colors[2],
#     'MPC': colors[-1],
#     'MAX': 'none',
#     'MIN': 'none',
# }

plot_colors = {
    'GP-MPC': 'royalblue',
    'PPO': 'darkorange',
    'SAC': 'red',
    'DPPO': 'tab:pink',
    'PID': 'darkgray',
    'Linear MPC': 'green',
    'Nonlinear MPC': 'cadetblue',
    'F-MPC': 'darkblue',
    "iLQR": "slateblue",
    'LQR': 'blueviolet',
    'MAX': 'none',
    'MIN': 'none',
}

axis_label_fontsize = 30
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20


def spider(df, *, id_column, title=None, subtitle=None, max_values=None, padding=1.25, plt_name=''):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient='list')
    ids = df[id_column].tolist()

    lower_padding = (padding - 1) / 2
    # upper_padding = 1 + lower_padding * 2
    upper_padding = 1 + 7 * lower_padding
    # upper_padding = 1.05
    flip_flag = True

    if max_values is None:
        max_values = {key: upper_padding * max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] + lower_padding for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    print('tiks:', tiks)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True), )
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]

        # Invert the values to have the higher values in the center
        values[0:5] = 1 - np.array(values[0:5])

        values += values[:1]  # Close the plot for a better look
        # values = 1 - np.array(values) 
        if model_name in ['MAX', 'MIN']:
            ax.plot(angles, values, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )
            continue
        else:
            ax.plot(angles, values, label=model_name, color=plot_colors[model_name], )
            ax.scatter(angles, values, facecolor=plot_colors[model_name], )
            ax.fill(angles, values, alpha=0.15, color=plot_colors[model_name], )
        for _x, _y, t in zip(angles, values, actual_values):
            if _x == angles[2]: # inference time
                t = f'{t:.1E}' if isinstance(t, float) else str(t)
                # t = f'{np.format_float_scientific(t, precision=1)}' if isinstance(t, float) else str(t)
            elif _x == angles[4]:  # sampling complexity
                if t == int(1):
                    # _y = 0.01
                    t = '0'
                else:
                    # write number in 1e5 format
                    t = f'{t:.1E}'  # if isinstance(t, float) else str(t)
            # elif _x == angles[0]:
            #     t = f'{t:.2f}' if isinstance(t, float) else str(t)
            else:
                t = f'{t:.3f}' if isinstance(t, float) else str(t)
            if t == '1': t = 'Model-free'
            if t == '40': t = '   Linear\n   model'
            if t == '80': t = 'Nonlinear\n   model'

            t = t.center(10, ' ')
            if _x == angles[3]:
                ax.text(_x + 0.2, _y + 0.15, t, size=small_text_size)
            elif _x == angles[0]:
                if flip_flag:
                    ax.text(_x + 0.2, _y - 0.1, t, size=small_text_size)
                else:
                    ax.text(_x - 0.2, _y + 0.0, t, size=small_text_size)
                flip_flag = False
            elif model_name == 'GP-MPC':
                if _x == angles[0]:
                    if flip_flag:
                        ax.text(_x + 0.05, _y - 0.1, t, size=small_text_size)
                    else:
                        ax.text(_x - 0.05, _y, t, size=small_text_size)
                    flip_flag = False
                elif _x == angles[4]:
                    ax.text(_x, _y - 0.05, t, size=small_text_size)
                else:
                    ax.text(_x, _y - 0.01, t, size=small_text_size)

            elif model_name == 'DPPO':
                # if _x == angles[5]:
                #     ax.text(_x, _y-0.1, t, size=small_text_size)
                # if _x == angles[0]: # generalization performance
                #     ax.text(_x-0.1, _y-0.05, t, size=small_text_size)
                # elif _x == angles[2]: # inference time
                #     ax.text(_x+0.1, _y-0.05, t, size=small_text_size)
                if _x == angles[4]:  # sampling complexity
                    ax.text(_x, _y + 0.15, t, size=small_text_size)
                else:
                    ax.text(_x, _y - 0.01, t, size=small_text_size)
            else:
                ax.text(_x, _y - 0.01, t, size=small_text_size)

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05, color='lightgray')
    # ax.fill(angles[0:3], np.ones(3), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=text_fontsize)
    if title is not None: plt.suptitle(title, fontsize=supertitle_fontsize)
    if subtitle is not None: plt.title(subtitle, fontsize=subtitle_fontsize)
    # plt.show()
    fig_save_path = os.path.join(script_dir, f'{plt_name}_radar.pdf')
    fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    print(f'figure saved as {fig_save_path}')


radar = spider

num_axis = 6
gen_performance = [0.07240253210609013, 0.031425261835490235,  # GP-MPC
                   0.09475888182425916, 0.034306490036127714,  # Linear-MPC
                   0.06669989755434953, 0.023313325828377546,  # MPC
                   0.07809048366149708, 0.03448930158456177, # F-MPC
                   0.1368606, 0.13115713,  # PPO
                   0.13443702, 0.10112919,  # SAC
                   0.16563129, 0.16684679,  # DPPO
                   0.175546733464246, 0.06473299142106152, # PID
                   0.48860034297750127, 0.026522666742796974, # iLQR
                   0.21703079983074353, 0.044875452075620124, # LQR
                   ]
performance = [0.049872511450839645, 0.049872511450839645,  # GP-MPC
               0.06284182159429033, 0.06284182159429033,  # Linear-MPC
               0.04421752503119518, 0.04421752503119518,  # MPC
               0.05599325343755744, 0.05599325343755744,  # F-MPC
               0.0537718862849458, 0.0537718862849458,  # PPO
               0.07645108154247876, 0.07645108154247876,  # SAC
               0.056197160779967906, 0.056197160779967906,  # DPPO
               0.1128442698191488, 0.1128442698191488, # PID
               0.0492655107844662, 0.04752121868080676, # iLQR
               0.12846739566058812, 0.12846739566058812, # LQR
               ]
inference_time = [0.0090775150246974736, 0.0090775150246974736, # GP-MPC
                  0.00159305, 0.00159305, # Linear-MPC
                  0.0061547613, 0.0061547613, # MPC
                  0.0054, 0.0054, # F-MPC
                  0.00020738168999000832, 0.00020738168999000832, # PPO
                  0.00024354409288477016, 0.00024354409288477016, # SAC
                  0.0001976909460844817, 0.0001976909460844817, # DPPO
                  0.0003089414, 0.0003089414, # PID
                  3.943804538999999e-06, 3.943804538999999e-06, # iLQR
                  4.951303655e-06, 4.951303655e-06, # LQR
                  ]
model_complexity = [80, 80, # GP-MPC
                    40, 40, # Linear-MPC
                    80, 80, # MPC
                    80, 80, # F-MPC
                    1, 1, # PPO
                    1, 1, # SAC
                    1, 1, # DPPO
                    1, 1, # PID
                    80, 80, # iLQR
                    40, 40, # LQR 
                    ]
sampling_complexity = [int(660), int(660),
                       int(1), int(1),
                       int(1), int(1),
                       int(1), int(1),
                       int(2.5 * 1e5), int(2.5 * 1e5), # PPO
                       int(2.1 * 1e5), int(2.1 * 1e5), # SAC
                       int(3.2 * 1e5), int(3.2 * 1e5), # DPPO
                       int(1), int(1),
                       int(1), int(1),
                       int(1), int(1),
                       ]
robustness = [120, 120, # GP-MPC
              90, 90, # Linear-MPC
              90, 90, # MPC
              100, 100, # F-MPC
              15, 15, # PPO
              30, 30, # SAC
              5, 5, # DPPO
              110, 110, # PID
              40, 40, # iLQR
              100, 100, # LQR
              ]
data = [gen_performance, performance, inference_time, model_complexity, sampling_complexity, robustness]
max_values = [0.01, 0.01, 1e-5, 1, 1, 120]
min_values = [0.2, 0.2, 1e-2, 80, 3.e5, 1]

for i, d in enumerate(data):
    data[i].append(max_values[i])
    data[i].append(min_values[i])

# append the max and min values to the data
algos = ['GP-MPC', 'GP-MPC',
         'Linear MPC', 'Linear MPC',
         'Nonlinear MPC', 'Nonlinear MPC',
         'F-MPC', 'F-MPC',
         'PPO', 'PPO',
         'SAC', 'SAC',
         'DPPO', 'DPPO',
         'PID', 'PID',
         'iLQR', 'iLQR',
         'LQR', 'LQR',
         'MAX', 'MIN']

# read the argv
if len(sys.argv) > 1:
    masks_algo = [int(i) for i in sys.argv[1:]]
    masks_algo.append(6)
    masks_algo.append(7)
else:
    # masks_algo = [8, 9, -2, -1]
    # masks_algo = [18, 19, -2, -1] # LQR
    # masks_algo = [16, 17, -2, -1] # iLQR
    # masks_algo = [14, 15, -2, -1] # PID
    # masks_algo = [12, 13, -2, -1] # DPPO
    # masks_algo = [10, 11, -2, -1] # SAC
    masks_algo = [8, 9, -2, -1] # PPO
    # masks_algo = [6, 7, -2, -1] # F-MPC
    # masks_algo = [4, 5, -2, -1] # Nonlinear MPC
    # masks_algo = [2, 3, -2, -1] # Linear MPC
    # masks_algo = [0, 1, -2, -1] # GP-MPC
    # masks_algo = [6, 7, -2, -1] # F-MPC
data = np.array(data)[:, masks_algo]
data = data.tolist()
algos = [algos[i] for i in masks_algo]
print(algos)

spider(
    pd.DataFrame({
        # 'x': [*'ab'],
        'x': algos,
        '$\qquad\qquad\qquad\quad$  Generalization\n $\qquad\qquad\qquad\quad$ performance\n\n':
            data[0],
        '$\qquad\qquad\qquad\quad$ Performance\n':
            data[1],
        # '$\quad\quad\quad\quad\quad\qquad$(Figure-8 tracking)': [3.94646538e-02, 0.03],
        'Inference\ntime\n\n':
            data[2],
        'Model                \nknowledge                ':
            [int(data[3][i]) for i in range(len(data[3]))],
        '\n\n\nSampling\ncomplexity':
            data[4],
        '\n\nRobustness':
            [int(data[5][i]) for i in range(len(data[5]))],
    }),

    id_column='x',
    # title='   Overall Comparison',
    # title = algos[0],
    title=None,
    # subtitle='(Normalized linear scale)',
    padding=1.1,
    # padding=1,
    plt_name=algos[0],
)
