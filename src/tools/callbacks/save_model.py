from pathlib import Path
from typing import Literal

import wandb

from tools.store_load_run import save_run
from utils.custom_print import print_info_best_model


class SaveBestModelCallback:
    """
    Callback to save the best model based on a metric.
    """

    def __init__(self,
                 metric_key: str,
                 mode: Literal["min", "max"] = "min",
                 ):
        """
        Initialize the callback.
        :param metric_key: Key of the metric to be used
        :param mode: Mode of the metric (`min` or `max`)
        """
        self.metric_key = metric_key
        self.mode = mode
        self.best_metric = None

    def on_epoch_end(self, logs, fabric, components, config, *args, **kwargs):
        """
        Save the model if the metric is better than the previous best metric.
        :param logs: Logs of the epoch.
        :param fabric: Fabric instance
        :param components: Components to be saved
        """
        if self.best_metric is None:
            self.best_metric = logs[self.metric_key]
        if (self.mode == "min" and logs[self.metric_key] <= self.best_metric) or (
                self.mode == "max" and logs[self.metric_key] >= self.best_metric):
            self.best_metric = logs[self.metric_key]

            print_info_best_model(f"Model achieved {self.best_metric:.4f} on {self.metric_key}", "New best model")

            path = save_run(config, fabric, components)
            if config['logging']['wandb']['active']:
                wandb.save(path, base_path=str(Path(path).parent.absolute()), policy="now")
