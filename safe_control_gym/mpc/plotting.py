"""GP-MPC lotting utilities."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_runtime(test_runs, train_runs):
    """get the mean, std, and max runtime"""
    # NOTE: only implemented for single episode
    # NOTE: the first step is popped out because of the ipopt initial guess

    num_epochs = len(train_runs.keys())
    num_train_samples_by_epoch = []  # number of training data
    mean_runtime = np.zeros(num_epochs)
    std_runtime = np.zeros(num_epochs)
    max_runtime = np.zeros(num_epochs)

    runtime = []
    for epoch in range(num_epochs):
        num_samples = len(train_runs[epoch].keys())
        num_train_samples_by_epoch.append(num_samples)
        # num_steps = len(test_runs[epoch][0]['runtime'])
        runtime = test_runs[epoch][0][0]["inference_time_data"][0][1:]  # remove the first step

        mean_runtime[epoch] = np.mean(runtime)
        std_runtime[epoch] = np.std(runtime)
        max_runtime[epoch] = np.max(runtime)

    runtime_result = {
        "mean": mean_runtime,
        "std": std_runtime,
        "max": max_runtime,
        "num_train_samples": num_train_samples_by_epoch,
    }

    return runtime_result


def plot_runtime(runtime, num_points_per_epoch, save_dir: Path):
    mean_runtime = runtime["mean"]
    std_runtime = runtime["std"]
    max_runtime = runtime["max"]
    # num_train_samples = runtime['num_train_samples']
    plt.plot(num_points_per_epoch, mean_runtime, label="mean")
    plt.fill_between(
        num_points_per_epoch,
        mean_runtime - std_runtime,
        mean_runtime + std_runtime,
        alpha=0.3,
        label="1-std",
    )
    plt.plot(num_points_per_epoch, max_runtime, label="max", color="r")
    plt.legend()
    plt.xlabel("Train Steps")
    plt.ylabel("Runtime (s) ")

    plt.savefig(save_dir / "runtime.png")
    plt.cla()
    plt.clf()
    data = np.vstack((num_points_per_epoch, mean_runtime, std_runtime, max_runtime)).T
    np.savetxt(save_dir / "runtime.csv", data, delimiter=",", header="Train Steps, Mean, Std, Max")


def plot_runs(
    all_runs,
    num_epochs,
    episode=0,
    ind=0,
    ylabel="x position",
    save_dir: Path | None = None,
    traj=None,
):
    # plot the reference trajectory
    if traj is not None:
        plt.plot(traj[:, ind], label="Reference", color="gray", linestyle="--")
    # plot the prior controller
    plt.plot(all_runs[0][0][episode]["state"][0][:, ind], label="prior MPC")
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode][0]["state"][0][:, ind], label=f"GP-MPC {epoch}")
    plt.title(ylabel)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir / f"ep{episode}_ind{ind}_state.png")
    else:
        plt.show()
    plt.cla()
    plt.clf()


def plot_runs_input(
    all_runs,
    num_epochs,
    episode=0,
    ind=0,
    ylabel="x position",
    save_dir: Path | None = None,
):
    # plot the prior controller
    plt.plot(all_runs[0][episode][0]["action"][0][:, ind], label="prior MPC")
    # plot each learning epoch
    for epoch in range(1, num_epochs):
        # plot the first episode of each epoch
        plt.plot(all_runs[epoch][episode][0]["action"][0][:, ind], label=f"GP-MPC {epoch}")
    plt.title(ylabel)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir / f"ep{episode}_ind{ind}_action.png")
    else:
        plt.show()
    plt.clf()


def plot_learning_curve(avg_rewards, num_points_per_epoch, stem, save_dir: Path):
    samples = num_points_per_epoch  # data number
    rewards = np.array(avg_rewards)
    plt.plot(samples, rewards)
    plt.title("Avg Episode" + stem)
    plt.xlabel("Training Steps")
    plt.ylabel(stem)
    plt.savefig(save_dir / (stem + ".png"))
    plt.cla()
    plt.clf()
    data = np.vstack((samples, rewards)).T
    np.savetxt(save_dir / (stem + ".csv"), data, delimiter=",", header="Train steps,Cost")


def plot_xyz_trajectory(runs, ref, save_dir: Path):
    num_epochs = len(runs)
    fig, ax = plt.subplots(3, 1)

    # x-y plane
    ax[0].plot(ref[:, 0], ref[:, 2], label="Reference", color="gray", linestyle="--")
    ax[0].plot(runs[0][0][0]["obs"][0][:, 0], runs[0][0][0]["obs"][0][:, 2], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[0].plot(
            runs[epoch][0][0]["obs"][0][:, 0],
            runs[epoch][0][0]["obs"][0][:, 2],
            label="GP-MPC %s" % epoch,
        )
    ax[0].set_title("X-Y plane path")
    ax[0].set_xlabel("X [m]")
    ax[0].set_ylabel("Y [m]")
    ax[0].legend()
    # x-z plane
    ax[1].plot(ref[:, 0], ref[:, 4], label="Reference", color="gray", linestyle="--")
    ax[1].plot(runs[0][0][0]["obs"][0][:, 0], runs[0][0][0]["obs"][0][:, 4], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[1].plot(
            runs[epoch][0][0]["obs"][0][:, 0],
            runs[epoch][0][0]["obs"][0][:, 4],
            label="GP-MPC %s" % epoch,
        )
    ax[1].set_title("X-Z plane path")
    ax[1].set_xlabel("X [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].legend()
    # y-z plane
    ax[2].plot(ref[:, 2], ref[:, 4], label="Reference", color="gray", linestyle="--")
    ax[2].plot(runs[0][0][0]["obs"][0][:, 2], runs[0][0][0]["obs"][0][:, 4], label="prior MPC")
    for epoch in range(1, num_epochs):
        ax[2].plot(
            runs[epoch][0][0]["obs"][0][:, 2],
            runs[epoch][0][0]["obs"][0][:, 4],
            label="GP-MPC %s" % epoch,
        )
    ax[2].set_title("Y-Z plane path")
    ax[2].set_xlabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].legend()

    fig.savefig(save_dir / "xyz_path.png")
    plt.cla()
    plt.clf()


def make_quad_plots(test_runs, train_runs, trajectory, save_dir):
    num_steps, nx = test_runs[0][0][0]["state"][0].shape
    nu = test_runs[0][0][0]["action"][0].shape[1]
    # trim the traj steps to mach the evaluation steps
    trajectory = trajectory[0:num_steps, :]
    num_epochs = len(test_runs)
    num_episodes = len(test_runs[0])
    fig_dir = save_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=False)
    num_points_per_epoch = []
    for episode_i in range(num_episodes):
        plot_xyz_trajectory(test_runs, trajectory, fig_dir)
        for ind in range(nx):
            ylabel = "x%s" % ind
            plot_runs(
                test_runs,
                num_epochs,
                episode=episode_i,
                ind=ind,
                ylabel=ylabel,
                save_dir=fig_dir,
                traj=trajectory,
            )
        for ind in range(nu):
            ylabel = "u%s" % ind
            plot_runs_input(
                test_runs, num_epochs, episode=episode_i, ind=ind, ylabel=ylabel, save_dir=fig_dir
            )
    num_points = 0
    num_points_per_epoch.append(num_points)
    for epoch in range(1, num_epochs):
        num_train_episodes = len(train_runs[epoch])
        for episode in range(num_train_episodes):
            num_points += train_runs[epoch][episode][0]["obs"][0].shape[0]
        num_points_per_epoch.append(num_points)

    rmse_error = [test_runs[epoch][0][1]["rmse"] for epoch in range(num_epochs)]
    plot_learning_curve(rmse_error, num_points_per_epoch, "rmse_error_learning_curve", fig_dir)
    runtime_result = get_runtime(test_runs, train_runs)
    plot_runtime(runtime_result, num_points_per_epoch, fig_dir)
