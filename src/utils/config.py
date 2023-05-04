import operator
import os
from argparse import Namespace
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from yaml.loader import SafeLoader

from utils.custom_print import print_exception, print_info_config, print_warn

_path_t = Union[str, os.PathLike, Path]

CONFIGS_DIR = Path("../configs")
DATA_CONFIGS_FP = CONFIGS_DIR / "data.yaml"


def _load_config(path: _path_t) -> Dict[str, Any]:
    """
    Load a YAML config file.
    :param path: Path to the config file.
    :return: A dictionary with the config.
    """
    with open(path) as f:
        return yaml.load(f, Loader=SafeLoader)


def _add_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a dataset config to the config.
    :param config: The config.
    :return: The config with the dataset config added.
    """
    dataset_name = config["dataset"]["name"]
    dataset_augmentation = config["dataset"]["augmentation"]
    data_config = _load_config(DATA_CONFIGS_FP)
    config["dataset"] = config["dataset"] | data_config[dataset_name]
    config["dataset"]["dir"] = Path(data_config["base_dir"]) / config["dataset"]["dir"]
    config["dataset"]["beton_dir"] = Path(data_config["base_dir"]) / config["dataset"]["beton_dir"]

    if dataset_augmentation is not None and dataset_augmentation != "None":
        config["dataset"]["augmentation"] = data_config[dataset_augmentation]
    return config


def get_from_nested_dict(data_dict: Dict, key_list: Union[List[str] | str]):
    """
    Get a value from a nested dictionary.
    :param data_dict: The dictionary to get the value from.
    :param key_list: A list of keys to get to the value.
    """
    if isinstance(key_list, str):
        key_list = [key_list]
    return reduce(operator.getitem, key_list, data_dict)


def set_in_nested_dict(data_dict: Dict, key_list: Union[List[str] | str], value: Any):
    """
    Set a value in a nested dictionary.
    :param data_dict: The dictionary to set the value in.
    :param key_list: A list of keys to get to the value.
    :param value: The value to set.
    """
    if isinstance(key_list, str):
        key_list = [key_list]
    get_from_nested_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def _add_cli_args(config: Dict[str, Any], cli_args: Namespace) -> Dict[str, Any]:
    """
    Add command line arguments to the config.
    :param config: The config.
    :param cli_args: Command line arguments.
    :return: The config with the command line arguments added.
    """
    for key, value in vars(cli_args).items():
        if value is None:
            continue
        try:
            set_in_nested_dict(config, key.split(":"), value)
        except Exception as e:
            print_exception(e)
            print_warn(f"Could not set {key} to {value} in config.")
        if "cli_args" not in config:
            config["cli_args"] = {}
        config["cli_args"][key] = value
    return config


def get_config(
        config_name: str,
        cli_args: Namespace = None,
) -> Dict[str, Any]:
    """
    Get the config.
    :param config_name: Name of the config.
    :param cli_args: Command line arguments.
    :return: A dictionary with the config.
    """
    config = _load_config(CONFIGS_DIR / f"{config_name}.yaml")
    config = _add_data_config(config)
    if cli_args is not None:
        config = _add_cli_args(config, cli_args)
    print_info_config(config, "Config")
    return config
