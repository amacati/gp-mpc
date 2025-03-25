import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import crazyflow  # noqa: F401, register environments
import gymnasium
import numpy as np
import torch
import yaml
from crazyflow.sim.physics import ang_vel2rpy_rates
from crazyflow.sim.symbolic import symbolic_attitude
from gymnasium.wrappers.vector.jax_to_numpy import JaxToNumpy
from munch import munchify
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from gpmpc.gpmpc import GPMPC
from gpmpc.plotting import make_quad_plots, plot_quad_eval


def load_config():
    config_path = Path(__file__).parent / "gp_mpc_config.yaml"
    with open(config_path, "r") as file:
        config = munchify(yaml.safe_load(file))
    root_dir = Path(__file__).parents[1] / config.save_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = mkdir_date(root_dir)
    config.gpmpc.output_dir = config.save_dir
    return config


def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten the observation dictionary into a 1D numpy array."""
    x, y, z = obs["pos"]
    rpy = R.from_quat(obs["quat"]).as_euler("xyz")
    vx, vy, vz = obs["vel"]
    rpy_rates = ang_vel2rpy_rates(obs["ang_vel"], obs["quat"])
    obs = np.array([x, vx, y, vy, z, vz, *rpy, *rpy_rates])
    return obs


def run_evaluation(env, ctrl: GPMPC, seed: int) -> dict:
    episode_data = defaultdict(list)
    ctrl.reset()
    obs, _ = env.reset(seed=seed)
    obs = flatten_obs(obs)

    episode_data["obs"].append(obs)

    env.action_space.seed(seed)
    ctrl_data = defaultdict(list)
    inference_time_data = []

    while True:
        time_start = time.perf_counter()
        action = ctrl.select_action(obs)
        inference_time_data.append(time.perf_counter() - time_start)
        # Vector environment expects a batched action (world size 1) in float32
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32).reshape(1, -1))
        obs = flatten_obs(obs)
        done = terminated or truncated
        step_data = {"obs": obs, "action": action, "done": done, "reward": reward, "length": 1}
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
    env: JaxToNumpy,
    eval_size: float,
    lr: float,
    gp_iterations: int,
    seed: int,
    samples_per_epoch: int,
):
    """Performs multiple epochs learning."""
    train_runs, test_runs = {}, {}
    # Generate n unique random integers for epoch seeds and one for evaluation
    rng = np.random.default_rng(seed)
    eval_seed = int(rng.integers(np.iinfo(np.int32).max))
    # To make the results reproducible across runs with varying number of epochs, we create seeds
    # for 1e6 epochs and then use the first n_epochs of them. This guarantees that the same seeds
    # are used for the episodes no matter how many epochs are run. We could also reseed the rng
    # after sampling and sample each episode independently, but this prevents us from using replace.
    assert n_epochs < int(1e6), f"Number of epochs must be less than 1e6, got {n_epochs}"
    epoch_seeds = rng.choice(np.iinfo(np.int32).max, size=int(1e6), replace=False)[: n_epochs + 1]

    pbar = tqdm(range(n_epochs), desc="GP-MPC", dynamic_ncols=True)
    # Run prior
    train_runs[0] = run_evaluation(env, ctrl.prior_ctrl, seed=int(epoch_seeds[0]))
    test_runs[0] = run_evaluation(env, ctrl.prior_ctrl, seed=eval_seed)
    x_train, y_train = np.zeros((0, 7)), np.zeros((0, 3))  # 7 inputs, 3 outputs

    for epoch in range(1, n_epochs + 1):
        # Gather training data and train the GP
        state, actions, next_state = sample_data(train_runs[epoch - 1], samples_per_epoch, rng)
        inputs, targets = ctrl.preprocess_data(state, actions, next_state)
        x_train = np.vstack((x_train, inputs))  # Add to the existing training dataset
        y_train = np.vstack((y_train, targets))
        t3 = time.perf_counter()
        ctrl.train_gp(x=x_train, y=y_train, lr=lr, iterations=gp_iterations, test_size=eval_size)
        t4 = time.perf_counter()
        # Test new policy.
        test_runs[epoch] = run_evaluation(env, ctrl, eval_seed)
        t5 = time.perf_counter()
        # Gather training data
        train_runs[epoch] = run_evaluation(env, ctrl, int(epoch_seeds[epoch]))
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

    # TODO: Add the information from config.gpmpc.prior_info to the symbolic model
    prior_model = symbolic_attitude(dt=0.02, params=config.gpmpc.prior_params)

    # Run the experiment.
    # Get initial state and create environments
    env = JaxToNumpy(gymnasium.make_vec("DroneFigureEightXY-v0", num_envs=1))
    traj = env.unwrapped.trajectory.T

    # Create controller.
    ctrl = GPMPC(prior_model, traj=traj, seed=config.seed, **config.gpmpc)

    train_runs, test_runs = learn(
        n_epochs=config.run.num_epochs,
        ctrl=ctrl,
        env=env,
        eval_size=config.train.eval_size,
        lr=config.train.lr,
        gp_iterations=config.train.iterations,
        seed=config.seed,
        samples_per_epoch=config.train.samples_per_epoch,
    )

    # plotting training and evaluation results
    make_quad_plots(
        test_runs=test_runs, train_runs=train_runs, trajectory=ctrl.traj.T, save_dir=config.save_dir
    )

    # Run tests on a seed different from the test seed
    trajs_data = run_evaluation(env, ctrl, seed=config.seed + 1)
    env.close()

    dt = ctrl.model.dt
    plot_quad_eval(trajs_data, traj, dt, config.save_dir)


def mkdir_date(path: Path) -> Path:
    """Make a unique directory within the given directory with the current time as name.

    Args:
        path: Parent folder path.
    """
    assert path.is_dir(), f"Path {path} is not a directory"
    save_dir = path / datetime.now().strftime("%Y_%m_%d_%H_%M")
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        t = 1
        while save_dir.is_dir():
            curr_date_unique = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_({t})"
            save_dir = path / (curr_date_unique)
            t += 1
        save_dir.mkdir(parents=True)
    return save_dir


if __name__ == "__main__":
    tstart = time.perf_counter()
    run()
    print(f"Experiment took {time.perf_counter() - tstart:.2f} seconds")
