from numbers import Number
from typing import Union
import numpy as np
import torch

numeric = Union[int, float, complex, np.number, torch.tensor()]


class AverageMeter:
    """
    Computes and stores the average value.
    """

    def __init__(self):
        """
        Constructor
        """
        self.count = 0
        self.mean = 0

    def reset(self):
        """
        Reset the meter
        """
        self.count = 0
        self.mean = 0

    def add(self, value: numeric, weight: numeric = 1):
        """
        Add a new value to the meter
        :param value: Value to add
        :param weight: Weight of the value (typically the batch size)
        """
        self.mean = (self.mean * self.count + value * weight) / (self.count + weight)
        self.count += weight
