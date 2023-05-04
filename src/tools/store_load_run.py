from typing import Any, Dict, Optional

from lightning.fabric import Fabric

from utils.custom_print import print_info


def merge_configs(
        config: Dict[str, Optional[Any]],
        config_old: Dict[str, Optional[Any]]
) -> Dict[str, Optional[Any]]:
    """
    Merge two configurations.
    :param config: Current configuration
    :param config_old: Old configuration
    :return: Configuration dict
    """
    config['run']['current_epoch'] = config_old['run']['current_epoch']
    return config


def save_run(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        components: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save the current run.
    :param config: Configuration dict
    :param fabric: The fabric
    :param components: Dict with all components to be saved
    :return: Path to the saved run
    """
    state = {
        "config": config,
    }
    if components is not None:
        state = {**state, **components}
    fabric.save(config['run']['store_state_path'], state)
    print_info(f"Saved run to {config['run']['store_state_path']}", "Run State Saved")
    return config['run']['store_state_path']


def load_run(
        config: Dict[str, Optional[Any]],
        fabric: Fabric
) -> (Dict[str, Optional[Any]], Dict[str, Any]):
    """
    Load a run.
    :param fabric: Fabric instance
    :return: The configuration dict, and a dict with all components
    """
    state = fabric.load(config['run']['load_state_path'])
    print_info(f"Loaded run from {config['run']['load_state_path']}", "Run State Loaded")
    config_old = state.pop("config")
    config = merge_configs(config, config_old)
    return config, state
