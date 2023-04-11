from typing import Dict, Union
import torchmetrics
from tools import AverageMeter


class AverageMetricWrapper:
    """
    Wrapper for torchmetrics to compute the average value.
    """
    def __init__(self, metric):
        """
        Constructor
        :param metric: Metric to wrap
        """
        self.metric = metric
        self.avg_meter = AverageMeter()

    def __call__(self, *args, **kwargs):
        """
        Call the metric and add the result to the average meter
        """
        result = self.metric(*args, **kwargs)
        self.avg_meter.add(result)
        return result

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


def metrics_from_conf(
        config: Dict,
) -> Dict[Union[torchmetrics.Metric | AverageMetricWrapper]]:
    """
    Create a list of torchmetrics from a config dict.
    :param config: Config dict
    :return: Dict of torchmetrics with metric names as keys
    """
    result = {}
    for m_name, metric_conf in config['metrics'].items():
        class_ = getattr(torchmetrics, metric_conf['type'])
        metric = class_(**metric_conf['params'])
        if 'meter' in metric_conf:
            if metric_conf['meter'] == 'avg':
                metric = AverageMetricWrapper(m_name)
        result[m_name] = metric

    return result
