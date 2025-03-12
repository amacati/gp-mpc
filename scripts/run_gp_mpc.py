import os
import pickle
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FormatStrFormatter
from munch import munchify

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.mpc.plotting import make_quad_plots
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config

script_path = os.path.dirname(os.path.realpath(__file__))


def load_config():
    # TODO: Load the whole config from a single file, remove ConfigFactory
    config_path = Path(__file__).parent / "gp_mpc_config.yaml"
    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))

    seed = "1"
    ALGO = "gpmpc_acados_TRP"
    CTRL_ADD = ""
    SYS = "quadrotor_3D_attitude"
    TASK = "tracking"
    PRIOR = "100"
    agent = "quadrotor"

    # check if the config file exists
    root_path = Path(__file__).parent / "config_overrides"
    additional_cfg = root_path / f"{SYS}_{TASK}.yaml"
    ctrl_cfg = root_path / f"{ALGO}_{SYS}_{TASK}_{PRIOR}{CTRL_ADD}.yaml"
    print(f"additional_cfg: {additional_cfg}, ctrl_cfg: {ctrl_cfg}")
    assert additional_cfg.exists(), f"{additional_cfg} does not exist"
    assert ctrl_cfg.exists(), f"{ctrl_cfg} does not exist"
    sys.argv[1:] = [
        "--algo",
        ALGO,
        "--task",
        agent,
        "--overrides",
        f"./scripts/config_overrides/{SYS}_{TASK}.yaml",
        f"./scripts/config_overrides/{ALGO}_{SYS}_{TASK}_{PRIOR}{CTRL_ADD}.yaml",
        "--seed",
        seed,
        "--use_gpu",
        "True",
        "--output_dir",
        f"./{ALGO}/results",
    ]
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--n_episodes", type=int, default=1, help="number of episodes to run.")
    config = fac.merge()
    num_data_max = config.run.num_epochs * config.algo_config.num_samples
    config.output_dir = str(Path(config.output_dir) / f"{PRIOR}_{num_data_max}")
    set_dir_from_config(config)
    config.algo_config.output_dir = config.output_dir
    mkdirs(config.output_dir)
    return config


def run():
    """The main function running experiments for model-based methods."""
    config = load_config()
    # Create an environment
    env_func = partial(make, config.task, seed=config.seed, **config.task_config)
    random_env = env_func(gui=False)

    # Create controller.
    ctrl = make(config.algo, env_func, seed=config.seed, **config.algo_config)

    all_trajs = defaultdict(list)

    # Run the experiment.
    # Get initial state and create environments
    init_state, _ = random_env.reset()
    static_env = env_func(gui=False, randomized_init=False, init_state=init_state)
    static_train_env = env_func(gui=False, randomized_init=False, init_state=init_state)

    # Create experiment, train, and run evaluation
    experiment = BaseExperiment(env=static_env, ctrl=ctrl, train_env=static_train_env)
    experiment.reset()

    train_runs, test_runs = ctrl.learn(
        env=static_train_env,
        num_epochs=config.run.num_epochs,
        num_train_episodes_per_epoch=config.run.num_train_episodes_per_epoch,
        num_test_episodes_per_epoch=config.run.num_test_episodes_per_epoch,
    )

    trajs_data, _ = experiment.run_evaluation(training=True, n_episodes=1)

    # plotting training and evaluation results
    make_quad_plots(
        test_runs=test_runs,
        train_runs=train_runs,
        trajectory=ctrl.traj.T,
        dir=ctrl.output_dir,
    )

    # Close environments
    static_env.close()
    static_train_env.close()

    # Merge in new trajectory data
    for key, value in trajs_data.items():
        all_trajs[key] += value

    ctrl.close()
    random_env.close()
    metrics = experiment.compute_metrics(all_trajs)
    all_trajs = dict(all_trajs)

    results = {"trajs_data": all_trajs, "metrics": metrics}

    save_path = Path(__file__).parents[1] / "saves"
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "gpmpc_results.pkl", "wb") as file:
        pickle.dump(results, file)

    with open(save_path / "metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    plot_quad_eval(results, experiment.env, save_path)

    with open(save_path / "rand_hist.txt", "w") as file:
        for key, value in ctrl.rand_hist.items():
            file.write(f"{key}: {value}\n")


def plot_quad_eval(res, env, save_path: Path):
    """Plots the input and states to determine success.

    Args:
        state_stack (ndarray): The list of observations in the latest run.
        input_stack (ndarray): The list of inputs of in the latest run.
    """
    state_stack = res["trajs_data"]["obs"][0]
    input_stack = res["trajs_data"]["action"][0]
    model = env.symbolic
    x_idx, y_idx, z_idx = 0, 2, 4

    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    action_bound = env.action_space

    # Plot states
    fig, axs = plt.subplots(model.nx, figsize=(8, model.nx * 1))
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label="actual")
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color="r", label="desired")
        axs[k].set(ylabel=env.STATE_LABELS[k] + f"\n[{env.STATE_UNITS[k]}]")
        axs[k].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title("State Trajectories")
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc="lower right")
    axs[-1].set(xlabel="time (sec)")
    fig.tight_layout()

    plt.savefig(save_path / "state_trajectories.png")

    # Plot inputs
    _, axs = plt.subplots(model.nu, figsize=(8, model.nu * 1))
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        # axs[k].plot(times, np.array(clipped_action_stack).transpose()[k, 0:plot_length], color='r')
        axs[k].set(ylabel=f"input {k}")
        axs[k].hlines(action_bound.high[k], 0, times[-1], color="gray", linestyle="--")
        axs[k].hlines(action_bound.low[k], 0, times[-1], color="gray", linestyle="--")
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f"\n[{env.ACTION_UNITS[k]}]")
        axs[k].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[0].set_title("Input Trajectories")
    axs[-1].set(xlabel="time (sec)")
    fig.tight_layout()

    plt.savefig(save_path / "input_trajectories.png")

    # plot the figure-eight
    fig, axs = plt.subplots(2, figsize=(8, 8))
    axs[0].plot(
        np.array(state_stack).transpose()[x_idx, 0:plot_length],
        np.array(state_stack).transpose()[z_idx, 0:plot_length],
        label="actual",
    )
    axs[0].plot(
        reference.transpose()[x_idx, 0:plot_length],
        reference.transpose()[z_idx, 0:plot_length],
        color="r",
        label="desired",
    )
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("z [m]")
    axs[0].set_title("State path in x-z plane")
    axs[0].legend()

    error = []
    for i in range(1, len(res["trajs_data"]["info"][0])):
        error.append(np.sqrt(res["trajs_data"]["info"][0][i]["mse"]))
    error = np.array(error)
    rmse = res["metrics"]["rmse"]
    # plot the tracking error
    axs[1].plot(times, error)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("tracking error [m]")
    axs[1].set_title(f"Tracking error {rmse:.4f} m")

    fig.tight_layout()

    plt.savefig(save_path / "state_xz_path.png")

    fig, axs = plt.subplots(1)
    axs.plot(
        np.array(state_stack).transpose()[x_idx, 0:plot_length],
        np.array(state_stack).transpose()[y_idx, 0:plot_length],
        label="actual",
    )
    axs.plot(
        reference.transpose()[x_idx, 0:plot_length],
        reference.transpose()[y_idx, 0:plot_length],
        color="r",
        label="desired",
    )
    axs.set_xlabel("x [m]")
    axs.set_ylabel("y [m]")
    axs.set_title("State path in x-y plane")
    axs.legend()
    fig.tight_layout()

    plt.savefig(save_path / "state_xy_path.png")


if __name__ == "__main__":
    tstart = time.time()
    run()
    tend = time.time()
    print(f"Experiment took {tend - tstart:.2f} seconds")
