from typing import Dict, List
from lightning.fabric.loggers import Logger
from lightning.pytorch.loggers import WandbLogger

def loggers_from_conf(conf: Dict) -> List[Logger]:
    """
    Create a list of loggers from a config dict.
    :param conf: Config dict
    :return: List of loggers
    """
    loggers = []
    for logger in conf['logging']:
        for logger_name, logger_conf in logger.items():
            if logger_name.lower() == "wandb":
                wb_conf = logger_conf['wandb']
                loggers.append(WandbLogger(project=wb_conf['project'],
                                           save_dir=wb_conf['save_dir'],
                                           log_model=wb_conf['log_model'],
                                           config=conf,
                                           job_type=wb_conf['job_type'],
                                           group=wb_conf['group']))
            else:
                raise NotImplementedError(f"Logger {logger_name} not implemented.")
    return loggers
