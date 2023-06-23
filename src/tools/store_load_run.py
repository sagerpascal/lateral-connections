import warnings
from typing import Any, Dict, Optional

from deepdiff import DeepDiff
from lightning.fabric import Fabric

from utils.custom_print import print_info


def _warn_different_configs(config: Dict[str, Optional[Any]], config_old: Dict[str, Optional[Any]]):
    """
    Print a warning if the two configurations differ.
    :param config: Current configuration
    :param config_old: Old configuration
    """
    diff = DeepDiff(config_old, config)
    keys_ignored = ["store_state_path", "load_state_path", "current_epoch", "n_epochs", "['logging']['wandb']['active']", "'logging:wandb:active'", "['run']['plots']['enable']", "'run:plots:enable'"]
    removed_items = [r for r in list(diff.get("dictionary_item_removed", [])) if
                     not bool([k for k in keys_ignored if k in r])]
    added_items = [a for a in list(diff.get("dictionary_item_added", [])) if
                   not bool([k for k in keys_ignored if k in a])]
    changed_items = {c: d for c, d in diff.get("values_changed", {}).items() if
                     not bool([k for k in keys_ignored if k in c])}

    if len(removed_items) + len(added_items) + len(changed_items) > 0:
        warnings.warn(
            f"Configurations Differ:\n\tRemoved Items: {removed_items}\n\tAdded Items: {added_items}\n\tChanged "
            f"Items: {changed_items}")
        input("Press Enter to continue...")

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
    _warn_different_configs(config, config_old)
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
