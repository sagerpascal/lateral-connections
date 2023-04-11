import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from typing import Dict, Any, Union
import os
from utils.custom_print import print_info_config

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


def add_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
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

    print_info_config(config, "Config")

    return config


def get_config(
        config_name: str,
) -> Dict[str, Any]:
    """
    Get the config.
    :param config_name: Name of the config.
    :return: A dictionary with the config.
    """
    config = _load_config(CONFIGS_DIR / f"{config_name}.yaml")
    config = add_data_config(config)
    return config

