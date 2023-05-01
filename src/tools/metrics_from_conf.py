from typing import Dict, Union
import torchmetrics
from src.tools import AverageMeter
from lightning.fabric import Fabric
import torch

CLASSIFICATION_METRICS = ['Accuracy', 'AUROC', 'AveragePrecision', 'F1Score', 'Precision', 'Recall', 'Dice']
REGRESSION_METRICS = ['MeanSquaredError', 'MeanAbsoluteError', 'MeanSquaredLogError', 'MeanAbsolutePercentageError']
SUPPORTED_METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS


class AverageMetricWrapper:
    """
    Wrapper for torchmetrics to compute the average value.
    """

    def __init__(self, metric):
        """
        Constructor
        :param metric: Metric to wrap
        """
        super().__init__()
        self.metric = metric
        self.avg_meter = AverageMeter()

    def __call__(self, *args, **kwargs):
        """
        Call the metric and add the result to the average meter
        """
        if isinstance(args[0], torch.Tensor):
            weight = args[0].shape[0]
        result = self.metric(*args, **kwargs)
        self.avg_meter.add(result, weight=weight)
        return result

    def add(self, *args, **kwargs):
        """
        Add the result to the average meter (same as __call__)
        """
        return self.__call__(*args, **kwargs)

    def reset(self):
        """
        Reset the values from the average is computed
        """
        self.avg_meter.reset()

    @property
    def mean(self) -> float:
        """
        Return the mean value
        :return:
        """
        return self.avg_meter.mean


def _classification_metrics_from_conf(
        config: Dict,
        fabric: Fabric,
) -> Dict[str, Union[torchmetrics.Metric | AverageMetricWrapper]]:
    """
    Create a list of torchmetrics from a config dict.
    :param config: Config dict
    :param fabric: Fabric instance
    :return: Dict of torchmetrics with metric names as keys
    """
    result = {}
    for metric in config['metrics']:
        for m_name, metric_conf in metric.items():
            assert metric_conf['type'] in SUPPORTED_METRICS, f'Unknown metric type {metric_conf["type"]} check ' \
                                                             f'(https://torchmetrics.readthedocs.io/en/latest/, ' \
                                                             f'the metric may be available but not yet implemented ' \
                                                             f'here)'
            class_ = getattr(torchmetrics, metric_conf['type'])
            if 'params' not in metric_conf:
                metric_conf['params'] = {}
            if metric_conf['type'] in CLASSIFICATION_METRICS and 'num_classes' in class_.__new__.__code__.co_varnames:
                metric_conf['params'] = metric_conf['params'] | {'num_classes': config['dataset']['num_classes']}
            metric = class_(**metric_conf['params']).to(fabric.device)
            if 'meter' in metric_conf:
                if metric_conf['meter'] == 'avg' or metric_conf['meter'] == 'mean':
                    metric = AverageMetricWrapper(metric)
                else:
                    raise NotImplementedError(f"Meter {metric_conf['meter']} not implemented.")
            else:
                raise NotImplementedError(f"Meter must be defined for config")
            result[m_name] = metric

    return result


def metrics_from_conf(
        config: Dict,
        fabric: Fabric,
) -> Dict[str, Union[torchmetrics.Metric | AverageMetricWrapper]]:
    """
    Create a list of torchmetrics from a config dict.
    :param config: Config file
    :param fabric: Fabric instance
    :return: Dict of torchmetrics with metric names as keys
    """
    return _classification_metrics_from_conf(config, fabric)
