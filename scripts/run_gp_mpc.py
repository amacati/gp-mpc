import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.ticker import FormatStrFormatter
from munch import munchify
from tqdm import tqdm

from safe_control_gym.mpc.gpmpc import GPMPC
from safe_control_gym.mpc.plotting import make_quad_plots
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdir_date


def load_config():
    config_path = Path(__file__).parent / "gp_mpc_config.yaml"
    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))
    root_dir = Path(__file__).parents[1] / config.save_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = mkdir_date(root_dir)
    config.gpmpc.output_dir = config.save_dir
    return config


def run_evaluation(env, ctrl: GPMPC, seed: int) -> dict:
    episode_data = defaultdict(list)
    ctrl.reset()
    obs, _ = env.reset(seed=seed)

    episode_data["obs"].append(obs)
    episode_data["state"].append(env.state)

    env.action_space.seed(seed)
    ctrl_data = defaultdict(list)
    inference_time_data = []

    while True:
        time_start = time.perf_counter()
        action = ctrl.select_action(obs)
        inference_time_data.append(time.perf_counter() - time_start)
        obs, reward, done, _ = env.step(action)
        step_data = {
            "obs": obs,
            "action": action,
            "done": done,
            "reward": reward,
            "length": 1,
            "state": env.state,
        }
        for key, val in step_data.items():
            episode_data[key].append(val)
        if done:
            break
    for key, val in episode_data.items():
        episode_data[key] = np.array(val)

    episode_data["controller_data"] = dict(ctrl_data)
    episode_data["inference_time_data"] = inference_time_data
    return episode_data


def sample_data(data, n_samples: int, rng):
    """Sample data from a list of observations and actions."""
    n = data["action"].shape[0]
    assert isinstance(data["action"], np.ndarray)
    assert isinstance(data["obs"], np.ndarray)
    idx = rng.choice(n - 1, n_samples, replace=False) if n_samples < n else np.arange(n - 1)
    obs = np.array(data["obs"])
    actions = np.array(data["action"])
    return obs[idx, ...], actions[idx, ...], obs[idx + 1, ...]


def learn(
    n_epochs: int,
    ctrl: GPMPC,
    train_env,
    test_env,
    test_size: float,
    lr: float,
    gp_iterations: int,
    train_seed: int,
    test_seed: int,
    samples_per_epoch: int,
):
    """Performs multiple epochs learning."""
    train_runs, test_runs = {}, {}
    # Generate n unique random integers for epoch seeds and one for evaluation
    rng = np.random.default_rng(train_seed)
    # To make the results reproducible across runs with varying number of epochs, we create seeds
    # for 1e6 epochs and then use the first n_epochs of them. This guarantees that the same seeds
    # are used for the episodes no matter how many epochs are run. We could also reseed the rng
    # after sampling and sample each episode independently, but this prevents us from using replace.
    assert n_epochs < int(1e6), f"Number of epochs must be less than 1e6, got {n_epochs}"
    epoch_seeds = rng.choice(np.iinfo(np.int32).max, size=int(1e6), replace=False)[: n_epochs + 1]

    pbar = tqdm(range(n_epochs), desc="GP-MPC", dynamic_ncols=True)
    # Run prior
    train_runs[0] = run_evaluation(train_env, ctrl.prior_ctrl, seed=int(epoch_seeds[0]))
    test_runs[0] = run_evaluation(test_env, ctrl.prior_ctrl, seed=test_seed)
    x_train, y_train = np.zeros((0, 7)), np.zeros((0, 3))  # 7 inputs, 3 outputs

    for epoch in range(1, n_epochs + 1):
        # Gather training data and train the GP
        state, actions, next_state = sample_data(train_runs[epoch - 1], samples_per_epoch, rng)
        inputs, targets = ctrl.preprocess_data(state, actions, next_state)
        x_train = np.vstack((x_train, inputs))  # Add to the existing training dataset
        y_train = np.vstack((y_train, targets))
        t3 = time.perf_counter()
        ctrl.train_gp(x=x_train, y=y_train, lr=lr, iterations=gp_iterations, test_size=test_size)
        t4 = time.perf_counter()
        # Test new policy.
        test_runs[epoch] = run_evaluation(test_env, ctrl, test_seed)
        t5 = time.perf_counter()
        # Gather training data
        train_runs[epoch] = run_evaluation(train_env, ctrl, int(epoch_seeds[epoch]))
        t6 = time.perf_counter()
        # Print timing table
        print("\nExecution Times (seconds):")
        print(f"{'Operation':<25} {'Time (s)':<10}")
        print("-" * 35)
        print(f"{'Train GP':<25} {t4 - t3:>10.2f}")
        print(f"{'Test GPMPC Performance':<25} {t5 - t4:>10.2f}")
        print(f"{'Collect GP Data':<25} {t6 - t5:>10.2f}")
        pbar.update(1)

    return train_runs, test_runs


def run():
    """The main function running experiments for model-based methods."""
    config = load_config()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    env_func = partial(make, config.task, seed=config.seed, **config.task_config)
    # Create a random initial state for all experiments

    # Create controller.
    ctrl = GPMPC(env_func, seed=config.seed, **config.gpmpc)

    # Run the experiment.
    # Get initial state and create environments
    train_seed = int(rng.integers(np.iinfo(np.int32).max))
    train_env = env_func(randomized_init=True, seed=train_seed)
    test_seed = int(rng.integers(np.iinfo(np.int32).max))
    test_env = env_func(randomized_init=True, seed=test_seed)

    train_runs, test_runs = learn(
        n_epochs=config.run.num_epochs,
        ctrl=ctrl,
        train_env=train_env,
        test_env=test_env,
        test_size=config.train.test_size,
        lr=config.train.lr,
        gp_iterations=config.train.iterations,
        train_seed=train_seed,
        test_seed=test_seed,
        samples_per_epoch=config.train.samples_per_epoch,
    )
    train_env.close()
    test_env.close()

    # plotting training and evaluation results
    make_quad_plots(
        test_runs=test_runs, train_runs=train_runs, trajectory=ctrl.traj.T, save_dir=config.save_dir
    )

    # Run evaluation on a seed different from the test seed
    eval_env = env_func(gui=False, randomized_init=False, seed=test_seed + 1)
    trajs_data = run_evaluation(eval_env, ctrl, seed=test_seed)
    eval_env.close()

    plot_quad_eval(trajs_data, eval_env, config.save_dir)


def plot_quad_eval(trajectories, env, save_path: Path):
    """Plots the input and states to determine success.

    Args:
        state_stack (ndarray): The list of observations in the latest run.
        input_stack (ndarray): The list of inputs of in the latest run.
    """
    state_stack = trajectories["obs"]
    input_stack = trajectories["action"]
    model = env.symbolic
    nx, dt = model.nx, model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, dt * plot_length, plot_length)

    reference = env.X_GOAL

    # Plot states
    fig, axs = plt.subplots(nx, figsize=(8, nx * 1))
    for k in range(nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label="actual")
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color="r", label="desired")
        axs[k].set(ylabel=env.STATE_LABELS[k] + f"\n[{env.STATE_UNITS[k]}]")
        axs[k].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if k != nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title("State Trajectories")
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc="lower right")
    axs[-1].set(xlabel="time (sec)")
    fig.tight_layout()

    plt.savefig(save_path / "state_trajectories.png")


if __name__ == "__main__":
    tstart = time.perf_counter()
    run()
    print(f"Experiment took {time.perf_counter() - tstart:.2f} seconds")
