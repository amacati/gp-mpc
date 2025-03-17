import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FormatStrFormatter
from munch import munchify
from tqdm import tqdm

from safe_control_gym.mpc.gpmpc_acados_TRP import GpMpcAcadosTrp
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
    config.algo_config.output_dir = config.save_dir
    return config


def run_evaluation(env, ctrl, seed: int) -> dict:
    episode_data = defaultdict(list)
    ctrl.reset()
    obs, info = env.reset(seed=seed)

    step_data = dict(obs=obs, info=info, state=env.state)
    for key, val in step_data.items():
        episode_data[key].append(val)

    env.action_space.seed(seed)
    ctrl_data = defaultdict(list)
    inference_time_data = []

    while True:
        time_start = time.perf_counter()
        action = ctrl.select_action(obs, info)
        inference_time_data.append(time.perf_counter() - time_start)
        obs, reward, done, info = env.step(action)
        step_data = dict(
            obs=obs, action=action, done=done, info=info, reward=reward, length=1, state=env.state
        )
        for key, val in step_data.items():
            episode_data[key].append(val)
        if done:
            break
    for key, val in episode_data.items():
        episode_data[key] = np.array(val)

    episode_data["controller_data"] = munchify(dict(ctrl_data))
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
    ctrl,
    train_env,
    test_env,
    test_data_ratio: float,
    train_seed: int,
    test_seed: int,
):
    """Performs multiple epochs learning."""
    train_runs, test_runs = {}, {}
    # Generate n unique random integers for epoch seeds and one for evaluation
    rng = np.random.default_rng(train_seed)
    epoch_seeds = [int(i) for i in rng.choice(np.iinfo(np.int32).max, size=n_epochs, replace=False)]
    pbar = tqdm(range(n_epochs), desc="GP-MPC", dynamic_ncols=True)
    # Run prior
    train_runs[0] = munchify(run_evaluation(train_env, ctrl, seed=epoch_seeds[0]))
    test_runs[0] = munchify(run_evaluation(test_env, ctrl, seed=test_seed))
    pbar.update(1)

    for epoch in range(1, n_epochs):
        # Gather training data and train the GP
        x_seq, actions, x_next_seq = sample_data(train_runs[epoch - 1], ctrl.num_samples, rng)
        train_inputs, train_targets = ctrl.preprocess_data(x_seq, actions, x_next_seq)
        ctrl.train_gp(
            input_data=train_inputs, target_data=train_targets, test_data_ratio=test_data_ratio
        )

        # Test new policy.
        ctrl.x_prev = test_runs[epoch - 1]["obs"][: ctrl.T + 1, :].T
        ctrl.u_prev = test_runs[epoch - 1]["action"][: ctrl.T, :].T
        run_results = run_evaluation(test_env, ctrl, test_seed)
        test_runs[epoch] = munchify(run_results)

        # Gather training data
        ctrl.x_prev = train_runs[epoch - 1]["obs"][: ctrl.T + 1, :].T
        ctrl.u_prev = train_runs[epoch - 1]["action"][: ctrl.T, :].T
        run_results = run_evaluation(train_env, ctrl, epoch_seeds[epoch])
        train_runs[epoch] = munchify(run_results)
        pbar.update(1)

    train_env.close()
    test_env.close()
    return train_runs, test_runs


def run():
    """The main function running experiments for model-based methods."""
    config = load_config()
    rng = np.random.default_rng(config.seed)
    env_func = partial(make, config.task, seed=config.seed, **config.task_config)
    # Create a random initial state for all experiments

    # Create controller.
    ctrl = GpMpcAcadosTrp(env_func, seed=config.seed, **config.algo_config)

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
        test_data_ratio=config.train.test_data_ratio,
        train_seed=train_seed,
        test_seed=test_seed,
    )

    # plotting training and evaluation results
    make_quad_plots(
        test_runs=test_runs, train_runs=train_runs, trajectory=ctrl.traj.T, save_dir=config.save_dir
    )

    # Run evaluation on a seed different from the test seed
    eval_env = env_func(gui=False, randomized_init=False, seed=test_seed + 1)
    trajs_data = run_evaluation(eval_env, ctrl, seed=test_seed)
    eval_env.close()
    ctrl.close()

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
