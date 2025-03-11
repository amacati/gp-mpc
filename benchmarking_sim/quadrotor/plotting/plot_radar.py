import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

script_dir = os.path.dirname(__file__)


def load_metric(transfer_metric, method):
    ctrl = tag_ctrl_list[method]
    res = np.load(f"{script_dir}/../data/{ctrl}_{SYS}_gen_results.npy", allow_pickle=True).item()
    transfer_metric[method] = {"rmse": [], "rmse_std": [], "inference_time": []}
    for T in episode_len_list:
        T = "_" + str(T)
        transfer_metric[method]["rmse"].append(res[T]["mean_rmse"])
        transfer_metric[method]["rmse_std"].append(res[T]["std_rmse"])
    transfer_metric[method]["rmse"] = np.array(transfer_metric[method]["rmse"])
    transfer_metric[method]["rmse_std"] = np.array(transfer_metric[method]["rmse_std"])
    transfer_metric[method]["inference_time"] = np.mean(res["inference_time"])
    return transfer_metric


# get the pyplot default color wheel
prop_cycle = plt.rcParams["axes.prop_cycle"]
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
SYS = "quadrotor_2D_attitude"
tag_ctrl_list = {
    "iLQR": "ilqr",
    "LQR": "lqr",
    "PID": "pid",
    "Linear MPC": "linear_mpc_acados",
    "Nonlinear MPC": "mpc_acados",
    "F-MPC": "fmpc",
    "GP-MPC": "gpmpc_acados_TP" if SYS == "quadrotor_2D_attitude" else "gpmpc_acados_TRP",
}
transfer_metric = {}
episode_len_list = [9, 10, 11, 12, 13, 14, 15]
transfer_metric = load_metric(transfer_metric, "iLQR")
transfer_metric = load_metric(transfer_metric, "F-MPC")
transfer_metric = load_metric(transfer_metric, "Nonlinear MPC")
transfer_metric = load_metric(transfer_metric, "Linear MPC")
transfer_metric = load_metric(transfer_metric, "PID")
transfer_metric = load_metric(transfer_metric, "LQR")
transfer_metric = load_metric(transfer_metric, "GP-MPC")

plot_colors = {
    "GP-MPC": "royalblue",
    "PPO": "darkorange",
    "SAC": "red",
    "DPPO": "tab:pink",
    "PID": "darkgray",
    "Linear MPC": "green",
    "Nonlinear MPC": "cadetblue",
    "F-MPC": "darkblue",
    "iLQR": "slateblue",
    "LQR": "blueviolet",
    "MAX": "none",
    "MIN": "none",
}

axis_label_fontsize = 30
text_fontsize = 30
supertitle_fontsize = 30
subtitle_fontsize = 30
small_text_size = 20


def spider(df, *, id_column, title=None, subtitle=None, max_values=None, padding=1.25, plt_name=""):
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient="list")
    ids = df[id_column].tolist()

    lower_padding = (padding - 1) / 2
    # upper_padding = 1 + lower_padding * 2
    upper_padding = 1 + 7 * lower_padding
    # upper_padding = 1.05
    flip_flag = True

    if max_values is None:
        max_values = {key: upper_padding * max(value) for key, value in data.items()}

    normalized_data = {
        key: np.array(value) / max_values[key] + lower_padding for key, value in data.items()
    }
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    print("tiks:", tiks)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw=dict(polar=True),
    )
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]

        # Invert the values to have the higher values in the center
        values[0:5] = 1 - np.array(values[0:5])

        values += values[:1]  # Close the plot for a better look
        # values = 1 - np.array(values)
        if model_name in ["MAX", "MIN"]:
            ax.plot(
                angles,
                values,
                color=plot_colors[model_name],
            )
            ax.scatter(
                angles,
                values,
                facecolor=plot_colors[model_name],
            )
            ax.fill(
                angles,
                values,
                alpha=0.15,
                color=plot_colors[model_name],
            )
            continue
        else:
            ax.plot(
                angles,
                values,
                label=model_name,
                color=plot_colors[model_name],
            )
            ax.scatter(
                angles,
                values,
                facecolor=plot_colors[model_name],
            )
            ax.fill(
                angles,
                values,
                alpha=0.15,
                color=plot_colors[model_name],
            )
        for _x, _y, t in zip(angles, values, actual_values):
            if _x == angles[2]:  # inference time
                t = f"{t:.1E}" if isinstance(t, float) else str(t)
                # t = f'{np.format_float_scientific(t, precision=1)}' if isinstance(t, float) else str(t)
            elif _x == angles[4]:  # sampling complexity
                if t == int(1):
                    # _y = 0.01
                    t = "0"
                else:
                    # write number in 1e5 format
                    t = f"{t:.1E}"  # if isinstance(t, float) else str(t)
            # elif _x == angles[0]:
            #     t = f'{t:.2f}' if isinstance(t, float) else str(t)
            else:
                t = f"{t:.3f}" if isinstance(t, float) else str(t)
            if t == "1":
                t = "Model-free"
            if t == "40":
                t = "   Linear\n   model"
            if t == "80":
                t = "Nonlinear\n   model"

            t = t.center(10, " ")
            if _x == angles[3]:
                ax.text(_x + 0.2, _y + 0.15, t, size=small_text_size)
            elif _x == angles[0]:
                if flip_flag:
                    ax.text(_x + 0.2, _y - 0.1, t, size=small_text_size)
                else:
                    ax.text(_x - 0.2, _y + 0.0, t, size=small_text_size)
                flip_flag = False
            elif model_name == "GP-MPC":
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

            elif model_name == "DPPO":
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

    ax.fill(angles, np.ones(num_vars + 1), alpha=0.05, color="lightgray")
    # ax.fill(angles[0:3], np.ones(3), alpha=0.05)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks, fontsize=axis_label_fontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2), fontsize=text_fontsize)
    if title is not None:
        plt.suptitle(title, fontsize=supertitle_fontsize)
    if subtitle is not None:
        plt.title(subtitle, fontsize=subtitle_fontsize)
    # plt.show()
    fig_save_path = os.path.join(script_dir, f"radar_{plt_name}.pdf")
    fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"figure saved as {fig_save_path}")


radar = spider

num_axis = 6


gen_performance = [
    transfer_metric["GP-MPC"]["rmse"][0],
    transfer_metric["GP-MPC"]["rmse"][-1],  # GP-MPC
    transfer_metric["Linear MPC"]["rmse"][0],
    transfer_metric["Linear MPC"]["rmse"][-1],  # Linear-MPC
    transfer_metric["Nonlinear MPC"]["rmse"][0],
    transfer_metric["Nonlinear MPC"]["rmse"][-1],  # MPC
    transfer_metric["F-MPC"]["rmse"][0],
    transfer_metric["F-MPC"]["rmse"][-1],  # F-MPC
    0.14263671,
    0.10946506,  # PPO
    0.11052249,
    0.07096967,  # SAC
    0.14777681,
    0.11818952,  # DPPO
    transfer_metric["PID"]["rmse"][0],
    transfer_metric["PID"]["rmse"][-1],  # PID
    transfer_metric["iLQR"]["rmse"][0],
    transfer_metric["iLQR"]["rmse"][-1],  # iLQR
    transfer_metric["LQR"]["rmse"][0],
    transfer_metric["LQR"]["rmse"][-1],  # LQR
]
performance = [
    transfer_metric["GP-MPC"]["rmse"][2],
    transfer_metric["GP-MPC"]["rmse"][2],  # GP-MPC
    transfer_metric["Linear MPC"]["rmse"][2],
    transfer_metric["Linear MPC"]["rmse"][2],  # Linear-MPC
    transfer_metric["Nonlinear MPC"]["rmse"][2],
    transfer_metric["Nonlinear MPC"]["rmse"][2],  # MPC
    transfer_metric["F-MPC"]["rmse"][2],
    transfer_metric["F-MPC"]["rmse"][2],  # F-MPC
    0.04944359,
    0.04944359,  # PPO
    0.06911531,
    0.06911531,  # SAC
    0.05183557,
    0.05183557,  # DPPO
    transfer_metric["PID"]["rmse"][2],
    transfer_metric["PID"]["rmse"][2],  # PID
    transfer_metric["iLQR"]["rmse"][2],
    transfer_metric["iLQR"]["rmse"][2],  # iLQR
    transfer_metric["LQR"]["rmse"][2],
    transfer_metric["LQR"]["rmse"][2],  # LQR
]
inference_time = [
    transfer_metric["GP-MPC"]["inference_time"],
    transfer_metric["GP-MPC"]["inference_time"],  # GP-MPC
    transfer_metric["Linear MPC"]["inference_time"],
    transfer_metric["Linear MPC"]["inference_time"],  # Linear-MPC
    transfer_metric["Nonlinear MPC"]["inference_time"],
    transfer_metric["Nonlinear MPC"]["inference_time"],  # MPC
    transfer_metric["F-MPC"]["inference_time"],
    transfer_metric["F-MPC"]["inference_time"],  # F-MPC
    0.00020738168999000832,
    0.00020738168999000832,  # PPO
    0.00024354409288477016,
    0.00024354409288477016,  # SAC
    0.0001976909460844817,
    0.0001976909460844817,  # DPPO
    transfer_metric["PID"]["inference_time"],
    transfer_metric["PID"]["inference_time"],  # PID
    transfer_metric["iLQR"]["inference_time"],
    transfer_metric["iLQR"]["inference_time"],  # iLQR
    transfer_metric["LQR"]["inference_time"],
    transfer_metric["LQR"]["inference_time"],  # LQR
]
model_complexity = [
    80,
    80,  # GP-MPC
    40,
    40,  # Linear-MPC
    80,
    80,  # MPC
    80,
    80,  # F-MPC
    1,
    1,  # PPO
    1,
    1,  # SAC
    1,
    1,  # DPPO
    1,
    1,  # PID
    80,
    80,  # iLQR
    40,
    40,  # LQR
]
sampling_complexity = [
    int(1320),
    int(1320),
    int(1),
    int(1),
    int(1),
    int(1),
    int(1),
    int(1),
    int(0.5 * 1e5),
    int(0.5 * 1e5),  # PPO
    int(2 * 1e5),
    int(2 * 1e5),  # SAC
    int(1 * 1e5),
    int(1 * 1e5),  # DPPO
    int(1),
    int(1),
    int(1),
    int(1),
    int(1),
    int(1),
]
robustness = [
    100,
    100,
    90,
    90,
    100,
    100,
    100,
    100,
    10,
    10,
    50,
    50,
    10,
    10,
    100,
    100,
    70,
    70,
    100,
    100,
]
data = [
    gen_performance,
    performance,
    inference_time,
    model_complexity,
    sampling_complexity,
    robustness,
]
max_values = [0.01, 0.01, 1e-5, 1, 1, 120]
min_values = [0.2, 0.2, 1e-2, 80, 3.0e5, 1]

for i, d in enumerate(data):
    data[i].append(max_values[i])
    data[i].append(min_values[i])

# append the max and min values to the data
algos = [
    "GP-MPC",
    "GP-MPC",
    "Linear MPC",
    "Linear MPC",
    "Nonlinear MPC",
    "Nonlinear MPC",
    "F-MPC",
    "F-MPC",
    "PPO",
    "PPO",
    "SAC",
    "SAC",
    "DPPO",
    "DPPO",
    "PID",
    "PID",
    "iLQR",
    "iLQR",
    "LQR",
    "LQR",
    "MAX",
    "MIN",
]

# read the argv
if len(sys.argv) > 1:
    # masks_algo = [int(i) for i in sys.argv[1:]]
    algo = sys.argv[1]
    # masks_algo.append(6)
    # masks_algo.append(7)
    if algo == "GP-MPC":
        masks_algo = [0, 1, -2, -1]
    elif algo == "PPO":
        masks_algo = [8, 9, -2, -1]
    elif algo == "SAC":
        masks_algo = [10, 11, -2, -1]
    elif algo == "DPPO":
        masks_algo = [12, 13, -2, -1]
    elif algo == "PID":
        masks_algo = [14, 15, -2, -1]
    elif algo == "iLQR":
        masks_algo = [16, 17, -2, -1]
    elif algo == "LQR":
        masks_algo = [18, 19, -2, -1]
    elif algo == "F-MPC":
        masks_algo = [6, 7, -2, -1]
    elif algo == "Nonlinear-MPC":
        masks_algo = [4, 5, -2, -1]
    elif algo == "Linear-MPC":
        masks_algo = [2, 3, -2, -1]

else:
    # masks_algo = [18, 19, -2, -1] # LQR
    masks_algo = [8, 9, -2, -1]
    # masks_algo = [16, 17, -2, -1] # iLQR
    # masks_algo = [14, 15, -2, -1] # PID
    # masks_algo = [12, 13, -2, -1] # DPPO
    # masks_algo = [10, 11, -2, -1] # SAC
    masks_algo = [8, 9, -2, -1]  # PPO
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
    pd.DataFrame(
        {
            # 'x': [*'ab'],
            "x": algos,
            "$\qquad\qquad\qquad\quad$  Generalization\n $\qquad\qquad\qquad\quad$ performance\n\n": data[
                0
            ],
            "$\qquad\qquad\qquad\quad$ Performance\n": data[1],
            # '$\quad\quad\quad\quad\quad\qquad$(Figure-8 tracking)': [3.94646538e-02, 0.03],
            "Inference\ntime\n\n": data[2],
            "Model                \nknowledge                ": [
                int(data[3][i]) for i in range(len(data[3]))
            ],
            "\n\n\nSampling\ncomplexity": data[4],
            "\n\nRobustness": [int(data[5][i]) for i in range(len(data[5]))],
        }
    ),
    id_column="x",
    # title='   Overall Comparison',
    # title = algos[0],
    title=None,
    # subtitle='(Normalized linear scale)',
    padding=1.1,
    # padding=1,
    plt_name=algos[0],
)
