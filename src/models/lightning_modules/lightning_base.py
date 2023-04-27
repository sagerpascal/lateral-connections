import lightning.pytorch as pl
from typing import Optional, Dict, Any, List
import torch
import tools
import wandb
from lightning.fabric import Fabric


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
        self.metrics, self.loss_meters = self.configure_meters()
        if conf['logging'].get('wandb', {}).get('active', False):
            self.configure_wandb_metrics()
        self.current_epoch_ = 0

    def log_step(self, loss: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = ""):
        """
        Log the loss and the metrics.
        :param loss: Loss of the step.
        :param y_hat: Predictions of the model.
        :param y: Targets.
        :param prefix: Prefix for the logging.
        """
        for m_name, metric in self.loss_meters.items():
            if m_name.startswith(prefix):
                metric(loss)
        for m_name, metric in self.metrics.items():
            if m_name.startswith(prefix):
                metric(y_hat, y)

    def configure_meters(self) -> (Dict[str, tools.AverageMeter], Dict[str, tools.AverageMetricWrapper]):
        """
        Configure (create instances) the metrics.
        :return: Dictionary of metric meters and dictionary of loss meters.
        """
        metrics = {}
        for prefix in self.logging_prefixes:
            metrics.update({f"{prefix}/{k}": v for k, v in tools.metrics_from_conf(self.conf, self.fabric).items()})
        loss_meters = {f"{prefix}/loss": tools.AverageMeter() for prefix in self.logging_prefixes}

        return metrics, loss_meters

    def configure_wandb_metrics(self):
        """
        Configure the metrics for wandb.
        """
        for prefix in self.logging_prefixes:
            wandb.define_metric(f'{prefix}/loss', summary='min')
            for m_name, in self.metrics.keys():
                wandb.define_metric(f'{prefix}/{m_name}', summary='max')

    def log_(self):
        """
        Log the metrics.
        """
        logs = {'epoch': self.current_epoch_}
        for prefix in self.logging_prefixes:
            logs[f"{prefix}/loss"] = self.loss_meters[f"{prefix}/loss"].mean
            self.loss_meters[f"{prefix}/loss"].reset()
        for m_name, m in self.metrics.items():
            logs[m_name] = m.mean
            m.reset()
        self.log_dict(logs)

    def on_epoch_end(self):
        """
        Callback at the end of an epoch (log data).
        """
        self.log_()
        self.current_epoch_ += 1
