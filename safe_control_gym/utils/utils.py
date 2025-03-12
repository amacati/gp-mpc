"""Miscellaneous utility functions."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import munch
import yaml


def mkdirs(*paths):
    """Makes a list of directories."""

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    """Converts string token to int, float or str."""
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except TypeError:
        return token


def read_file(path: Path, sep=","):
    """Loads content from a file (json, yaml, csv, txt).

    For json & yaml files returns a dict.
    For csv & txt returns list of lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    suffix = path.suffix.lower()
    supported_formats = {".json", ".yaml", ".yml", ".csv", ".txt"}

    if suffix not in supported_formats:
        raise ValueError(f"Unsupported format: {suffix}.")

    # load file based on format
    if suffix == ".json":
        with path.open("r") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        with path.open("r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    else:  # csv or txt
        data = []
        separator = sep if suffix == ".csv" else " "
        with path.open("r") as f:
            for line in f:
                line_tokens = [eval_token(t) for t in line.strip().split(separator)]
                # if only single item in line
                if len(line_tokens) == 1:
                    line_tokens = line_tokens[0]
                if len(line_tokens) > 0:
                    data.append(line_tokens)

    return data


def merge_dict(source_dict, update_dict):
    """Merges updates into source recursively."""
    for k, v in update_dict.items():
        if k in source_dict and isinstance(source_dict[k], dict) and isinstance(v, dict):
            merge_dict(source_dict[k], v)
        else:
            source_dict[k] = v


def set_dir_from_config(config):
    """Creates a output folder for experiment (and save config files).

    Naming format: {root (e.g. results)}/{tag (exp id)}/{seed}_{timestamp}_{git commit id}
    """
    # Make run folder (of a seed run for an experiment)
    seed = str(config.seed) if config.seed is not None else "-"
    timestamp = str(datetime.datetime.now().strftime("%b-%d-%H-%M-%S"))
    try:
        commit_id = (
            subprocess.check_output(["git", "describe", "--tags", "--always"])
            .decode("utf-8")
            .strip()
        )
        commit_id = str(commit_id)
    except BaseException:
        commit_id = "-"
    run_dir = f"seed{seed}_{timestamp}_{commit_id}"
    # Make output folder.
    config.output_dir = os.path.join(config.output_dir, config.tag, run_dir)
    mkdirs(config.output_dir)
    # Save config.
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)
    # Save command.
    with open(os.path.join(config.output_dir, "cmd.txt"), "a") as file:
        file.write(" ".join(sys.argv) + "\n")


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
