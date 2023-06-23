from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision import utils

from data import plot_images


class Conv2dFixedFilters(nn.Module):
    """
    Fixed 2D convolutional layer with 4 filters that detect straight lines.
    """

    def __init__(
            self,
            fabric: Fabric,
            add_bg_channel: Optional[bool] = False,
            optimized_filter_lines: Optional[bool] = False
    ):
        """
        Initializes the layer with the fixed filters.
        :param fabric: Fabric instance.
        :param add_bg_channel: Whether to add a background channel to the input.
        :param optimized_filter_lines: Whether to use a filter that is optimized for detecting straight lines
        """
        super(Conv2dFixedFilters, self).__init__()

        if optimized_filter_lines:
            self.weight = torch.tensor([[[[-1, +2, -1],
                                          [-1, +2, -1],
                                          [-1, +2, -1]]],
                                        [[[+0, -2, +2],
                                          [-2, +2, -2],
                                          [+2, -2, +0]]],
                                        [[[-1, -1, -1],
                                          [+2, +2, +2],
                                          [-1, -1, -1]]],
                                        [[[+2, -2, +0],
                                          [-2, +2, -2],
                                          [+0, -2, +2]]],
                                        ], dtype=torch.float32, requires_grad=False).to(fabric.device)
            self.weight = self.weight / 6
        else:
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
            self.weight = self.weight / 3

        self.add_bg_channel = add_bg_channel

    def apply_conv(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        x = F.conv2d(x, self.weight, padding="same")
        if self.add_bg_channel:
            background = torch.where(torch.sum(x, dim=1, keepdim=True) == 0., 1., 0.)
            x = torch.cat([x, background], dim=1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(self.apply_conv(x[:, idx, ...]))
            return torch.stack(result, dim=1)
        else:
            return self.apply_conv(x).unsqueeze(1)


class FixedDogFilter(nn.Module):
    """
    Fixed 2D convolutional layer with a DoG filter.
    """

    def __init__(self, filter_size, fabric: Fabric, add_bg_channel: Optional[bool] = False):
        """
        Initializes the layer with the fixed filters.
        :param filter_size: Size of the DoG filter.
        :param fabric: Fabric instance.
        :param add_bg_channel: Whether to add a background channel to the input.
        """
        super().__init__()
        gk1 = np.zeros((filter_size, filter_size))
        gk1[filter_size // 2, filter_size // 2] = 1
        gk2 = (scipy.ndimage.gaussian_filter(gk1, sigma=.5) - scipy.ndimage.gaussian_filter(gk1, sigma=1.0))
        self.dog = torch.Tensor(gk2[np.newaxis, np.newaxis, :, :]).to(
            fabric.device)  # Adding two singleton dimensions for input and output channels (1 each)
        self.add_bg_channel = add_bg_channel

    def apply_conv(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the fixed filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        x = F.conv2d(x, self.weight, padding="same")
        if self.add_bg_channel:
            background = torch.where(torch.sum(x, dim=1, keepdim=True) == 0., 1., 0.)
            x = torch.cat([x, background], dim=1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a 2D convolution with the DoG filters.
        :param x: Image to perform the convolution on.
        :return: Extracted features.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(self.apply_conv(x[:, idx, ...]))
            return torch.stack(result, dim=1)
        else:
            return self.apply_conv(x).unsqueeze(1)


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
        self.bin_threshold = conf["feature_extractor"]["bin_threshold"]
        self.model = self.configure_model()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input image.
        :return: reconstructed features
        """
        return self.model(x)

    def binarize_features(self, x: Tensor) -> Tensor:
        """
        Convert float tensor to binary tensor
        :param x: Features as float
        :return: Binarized Features
        """
        return torch.where(x > self.bin_threshold, 1., 0.)

    def configure_model(self) -> nn.Module:
        """
           Configures the model.
           :return: The model.
           """
        return Conv2dFixedFilters(self.fabric, add_bg_channel=self.conf["feature_extractor"]["add_bg_channel"],
                                  optimized_filter_lines=self.conf["feature_extractor"]["optimized_filter_lines"])
        # return FixedDogFilter(filter_size=12, fabric=self.fabric)

    def plot_model_weights(self, show_plot: Optional[bool] = False) -> List[Path]:
        """
        Plot a histogram of the model weights.
        :param show_plot: Whether to show the plot.
        :return: List of paths to the plots.
        """

        def _hist_plot(ax, weight, title):
            bins = 20
            min, max = torch.min(weight).item(), torch.max(weight).item()
            hist = torch.histc(weight, bins=bins, min=min, max=max)
            x = np.linspace(min, max, bins)
            ax.bar(x, hist, align='center')
            ax.set_xlabel(f'Bins form {min:.4f} to {max:.4f}')
            ax.set_title(title)

        def _plot_weights(ax, weight, title):
            weight_img_list = [weight[i, j].unsqueeze(0) for j in range(weight.shape[1]) for i in
                               range(weight.shape[0])]
            # Order is [(0, 0), (1, 0), ..., (3, 0), (0, 1), ..., (3, 7)]
            # The columns show the output channels, the rows the input channels
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True, pad_value=1)
            ax.imshow(grid.permute(1, 2, 0), interpolation='none')
            ax.set_title(title)

        files = []
        for layer, weight in [('feature extractor', self.model.weight)]:
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(axs[1], weight[:20, :20, ...].detach().cpu(), f"Weight matrix ({layer})")
            plt.tight_layout()

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None and fig_fp != "None":
                fp = Path(fig_fp) / f'weights_{layer}.png'
                plt.savefig(fp)
                files.append(fp)

            if show_plot:
                plt.show()

            plt.close()
        return files

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
                images_.append(z[b, f:f + 1, ...])
                titles_.append(f"Filter {f + 1}")
            images.extend(images_)
            titles.extend(titles_)
        plot_images(images=images, titles=titles, max_cols=5, vmin=0., vmax=1.)
