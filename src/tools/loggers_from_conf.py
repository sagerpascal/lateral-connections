from typing import Dict, List, Optional

from lightning.fabric.loggers import Logger
from lightning.pytorch.loggers import WandbLogger

from src.utils import print_logs


class ConsoleLogger:
    """
    Logger that prints to console
    """

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        """
        Log metrics to console
        :param metrics: The metrics to log
        :param step: The current step
        """
        logs_by_prefix = {"general": {'step': step}}
        for k, v in metrics.items():
            if "/" in k:
                prefix, metric = k.split("/", 1)
                if prefix not in logs_by_prefix:
                    logs_by_prefix[prefix] = {}
                logs_by_prefix[prefix][metric] = v
            else:
                logs_by_prefix["general"][k] = v

        for k, v in logs_by_prefix.items():
            print_logs(v, title=k)


def loggers_from_conf(conf: Dict) -> List[Logger]:
    """
    Create a list of loggers from a config dict.
    :param conf: Config dict
    :return: List of loggers
    """
    loggers = []
    for logger_name, logger_conf in conf['logging'].items():
        if not logger_conf['active']:
            continue
        if logger_name.lower() == "wandb":
            loggers.append(WandbLogger(project=logger_conf['project'],
                                       save_dir=logger_conf['save_dir'],
                                       log_model=logger_conf['log_model'],
                                       config=conf,
                                       job_type=logger_conf['job_type'],
                                       group=logger_conf['group']))
        elif logger_name.lower() == "console":
            loggers.append(ConsoleLogger())
        else:
            raise NotImplementedError(f"Logger {logger_name} not implemented.")
    return loggers
