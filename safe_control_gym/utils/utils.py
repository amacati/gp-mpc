"""Miscellaneous utility functions."""
from datetime import datetime
from pathlib import Path

import gymnasium as gym


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


def unwrap_wrapper(env, wrapper_class):
    """Retrieve a ``VecEnvWrapper`` object by recursively searching."""
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env, wrapper_class):
    """Check if a given environment has been wrapped with a given wrapper."""
    return unwrap_wrapper(env, wrapper_class) is not None
