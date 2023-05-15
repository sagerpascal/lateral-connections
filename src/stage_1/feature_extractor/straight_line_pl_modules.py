from typing import Any, Dict, Optional

import lightning.pytorch as pl
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor

from data import plot_images


class Conv2dFixedFilters(nn.Module):
    """
    Fixed 2D convolutional layer with 4 filters that detect straight lines.
    """

    def __init__(self, fabric: Fabric):
        """
        Initializes the layer with the fixed filters.
        :param fabric: Fabric instance.
        """
        super(Conv2dFixedFilters, self).__init__()
        self.weight = torch.tensor([[[[-1, +2, -1],
                                      [-1, +2, -1],
                                      [-1, +2, -1]]],
                                    [[[-1, -1, +2],
                                      [-1, +2, -1],
                                      [+2, -1, -1]]],
                                    [[[-1, -1, -1],
                                      [+2, +2, +2],
                                      [-1, -1, -1]]],
                                    [[[+2, -1, -1],
                                      [-1, +2, -1],
                                      [-1, -1, +2]]],
                                    ], dtype=torch.float32, requires_grad=False).to(fabric.device)
        # self.weight = torch.tensor([[[[-1, +1, -1],
        #                               [-1, +1, -1],
        #                               [-1, +1, -1]]],
        #                             [[[-1, -1, -1],
        #                               [+1, +1, +1],
        #                               [-1, -1, -1]]],
        #                             ], dtype=torch.float32, requires_grad=False).to(fabric.device)
        self.weight = self.weight / 3

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(F.conv2d(x[:, idx, ...], self.weight, padding="same"))
            return torch.stack(result, dim=1)
        else:
            return F.conv2d(x, self.weight, padding="same")


class FixedDogFilter(nn.Module):
    """
    Fixed 2D convolutional layer with a DoG filter.
    """

    def __init__(self, filter_size, fabric: Fabric):
        """
        Initializes the layer with the fixed filters.
        :param filter_size: Size of the DoG filter.
        :param fabric: Fabric instance.
        """
        super().__init__()
        gk1 = np.zeros((filter_size, filter_size))
        gk1[filter_size // 2, filter_size // 2] = 1
        gk2 = (scipy.ndimage.gaussian_filter(gk1, sigma=.5) - scipy.ndimage.gaussian_filter(gk1, sigma=1.0))
        self.dog = torch.Tensor(gk2[np.newaxis, np.newaxis, :, :]).to(fabric.device)  # Adding two singleton dimensions for input and output channels (1 each)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the DoG filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(F.conv2d(x[:, idx, ...], self.dog, padding="same"))
            return torch.stack(result, dim=1)
        else:
            return F.conv2d(x, self.dog, padding="same")


class FixedFilterFeatureExtractor(pl.LightningModule):
    """
    PyTorch Lightning module that uses a CNN with a fixed filter.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        self.model = self.configure_model()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input image.
        :return: reconstructed features
        """
        return self.model(x)

    def configure_model(self) -> nn.Module:
        """
           Configures the model.
           :return: The model.
           """
        return Conv2dFixedFilters(self.fabric)
        # return FixedDogFilter(filter_size=12, fabric=self.fabric)

    def visualize_encodings(self, x: Tensor):
        """
        Visualize the features of a batch of images.
        :param x: Input image.
        """
        if x.shape[0] > 20:
            x = x[:20, ...]
        z = self.forward(x).clip(0., 1.)
        images, masks, titles = [], [], []
        for b in range(x.shape[0]):
            images_, titles_ = [x[b]], [f"Orig."]
            for f in range(z.shape[1]):
                images_.append(z[b, f:f+1, ...])
                titles_.append(f"Filter {f+1}")
            images.extend(images_)
            titles.extend(titles_)
        plot_images(images=images, titles=titles, max_cols=5, vmin=0., vmax=1.)
