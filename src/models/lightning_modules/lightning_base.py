from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import wandb
from lightning.fabric import Fabric

import tools


class BaseLitModule(pl.LightningModule):
    """
    Lightning Base Module
    """

    def __init__(
            self,
            conf: Dict[str, Optional[Any]],
            fabric: Fabric,
            logging_prefixes: Optional[List[str]] = None,
    ):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        :param logging_prefixes: Prefixes for logging.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        if logging_prefixes is None:
            logging_prefixes = ["train", "val"]
        self.logging_prefixes = logging_prefixes
        self.metrics, self.avg_meters = self.configure_meters()
        if conf['logging'].get('wandb', {}).get('active', False):
            self.configure_wandb_metrics()
        self.current_epoch_ = 0

    def log_step(self,
                 processed_values: Optional[Dict[str, torch.Tensor]] = None,
                 metric_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 prefix: Optional[str] = "",
                 ):
        """
        Log the values and the metrics.
        :param processed_values: Values that are already processed and can be logged directly with an average value
        meter. Must be dictionaries with "meter_key" and "value".
        :param metric_pairs: Pairs of values that are fed into a metric meter. Must be tuples of (v1, v2), typically,
        v1 is a prediction and v2 is a target.
        :param prefix: Optional prefix for the logging.
        """
        if processed_values is not None:
            for k, v in processed_values.items():
                meter_name = f"{prefix}/{k}" if prefix != "" else f"{k}"
                if meter_name not in self.avg_meters:
                    self.avg_meters[meter_name] = tools.AverageMeter()
                self.avg_meters[meter_name](v)

        if metric_pairs is not None:
            for v1, v2 in metric_pairs:
                for m_name, metric in self.metrics.items():
                    if m_name.startswith(prefix) or (prefix == "" and not "/" in m_name):
                        metric(v1, v2)

    def configure_meters(self) -> (Dict[str, tools.AverageMeter], Dict[str, tools.AverageMetricWrapper]):
        """
        Configure (create instances) the metrics.
        :return: Dictionary of metric meters and dictionary of loss meters.
        """
        metrics, avg_meters = {}, {}
        for prefix in self.logging_prefixes:
            metrics.update({f"{prefix}/{k}": v for k, v in tools.metrics_from_conf(self.conf, self.fabric).items()})

        return metrics, avg_meters

    def configure_wandb_metrics(self):
        """
        Configure the metrics for wandb.
        """
        for prefix in self.logging_prefixes:
            wandb.define_metric(f'{prefix}/loss', summary='min')
            for m_name, in self.metrics.keys():
                wandb.define_metric(f'{prefix}/{m_name}', summary='max')

    def log_(self) -> Dict[str, float]:
        """
        Log the metrics.
        :return: Dictionary of logs.
        """
        logs = {'epoch': self.current_epoch_}
        for m_name, m in self.avg_meters.items():
            logs[m_name] = m.mean
            m.reset()
        for m_name, m in self.metrics.items():
            logs[m_name] = m.mean
            m.reset()
        self.log_dict(logs)
        return logs

    def on_epoch_end(self) -> Dict[str, float]:
        """
        Callback at the end of an epoch (log data).
        :return: Dictionary of logs.
        """
        logs = self.log_()
        self.current_epoch_ += 1
        return logs
