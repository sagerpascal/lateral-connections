from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
from torchvision import utils

from data import plot_images
from tools import AverageMeter, bin2dec
from utils import create_video_from_images_ffmpeg, print_logs

HEBBIAN_ALGO = Literal['instar', 'oja', 'vanilla']
W_INIT = Literal['random', 'zeros', 'identity']


# TODO: According to Christoph: The number of connections is important (not only the connection strength) -> Simple
#  measure: Limit the weight
# TODO: Limit the weight per kernel

class LateralLayer(nn.Module):

    def __init__(self,
                 fabric: Fabric,
                 in_channels: int,
                 out_channels: int,
                 locality_size: Optional[int] = 2,
                 lr: Optional[float] = 0.0005,
                 hebbian_rule: Optional[HEBBIAN_ALGO] = 'vanilla',
                 ):
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.

        This layer comprises two convolutional operations to implement Hebbian learning: a fixed, binary convolution
        that rearranges every input patch into a single column vector, followed by a 1 × 1 convolution that contains
        the actual weights. More precisely, suppose our original convolution has input size h × w × n_i (where h and w
        are the height and width of the convolutional filter, and n_i is the number of channels in the input), with
        n_o output channels. Then, we can first pass the input through a fixed convolution of input size h × w × n_i
        with hwn_i output channels, with a fixed weight vector set to 1 for the weights that links input x, y, i to
        output xyi (where x, y and i run from 1 to h, w and n_i respectively) and 0 everywhere else. This rearranges
        (and duplicates) the values of each input patch of the original convolution into single, non-overlapping column
        vectors. Afterwards we can apply the actual weights of the original convolution with a simple 1 × 1 convolution,
        which can be performed by a simple tensor product with appropriate broadcasting if necessary.

        This somewhat clunkier method does require two convolutional steps, as well as additional memory usage.
        However, it also provides finer-grained control.

        Proposed in https://arxiv.org/pdf/2107.01729.pdf

        :param fabric: Fabric instance.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param locality_size: Size of the locality, i.e. how many neurons are connected to each other. For
        example, if locality_size = 2, then each neuron is connected to 5 neurons on each side of it.
        :param lr: Learning rate.
        :param hebbian_rule: Which Hebbian rule to use.
        """
        super().__init__()
        self.fabric = fabric
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.locality_size = locality_size
        self.neib_size = 2 * self.locality_size + 1
        self.kernel_size = (self.neib_size, self.neib_size)
        self.lr = lr
        self.hebbian_rule = hebbian_rule

        self.W_rearrange = self._init_rearrange_weights()
        self.W_lateral = nn.Parameter(self._init_lateral_weights())
        self.ts = 0
        self.x_lateral_norm_prev = None

    def get_weights(self):
        return self.W_lateral.reshape((self.out_channels, self.in_channels) + self.kernel_size)

    def new_sample(self):
        """
        To be called when a new sample is fed into the network.
        """
        pass

    def update_ts(self, ts):
        """
        Set the current timestep (relevant for sparsity rate)
        :param ts: The current timestep
        """
        self.ts = ts

    def _init_rearrange_weights(self) -> Tensor:
        """
        Initialize the transpose weights.
        :return: Tensor of shape (in_channels * kernel_size[0] * kernel_size[1], in_channels, kernel_size[0],
        kernel_size[1])
        """
        W_r_shape = (self.in_channels * self.kernel_size[0] * self.kernel_size[1], self.in_channels) + self.kernel_size
        W_rearrange = torch.zeros(W_r_shape, device=self.fabric.device, requires_grad=False)

        for i in range(self.in_channels):
            for j in range(self.kernel_size[0]):
                for k in range(self.kernel_size[1]):
                    ijk = i * self.kernel_size[0] * self.kernel_size[1] + j * self.kernel_size[1] + k
                    W_rearrange[ijk, i, j, k] = 1

        return W_rearrange

    def rearrange_input(self, x: Tensor) -> Tensor:
        """
        Rearrange the input tensor x into a single column vector.
        If x is of shape (batch_size, in_channels, height, width), then the output is of shape (batch_size,
        in_channels * kernel_size[0] * kernel_size[1], height, width).

        If the output is of shape (C, 5, 5), then the rearrangement is as follows:
        # C = 0: all inputs of kernel weight at kernel position (0,0) and input channel 0
        # C = 1: all inputs of kernel weight at kernel position (0,1) and input channel 0
        # C = 4: all inputs of kernel weight at kernel position (0,4) and input channel 0
        # C = 5: all inputs of kernel weight at kernel position (1,0) and input channel 0
        # C = 24: all inputs of kernel weight at kernel position (4,4) and input channel 0
        # C = 25: all inputs of kernel weight at kernel position (0,0) and input channel 1

        i.e. x_rearranged[0] is a matrix of size (w,h) containing all inputs that affect the kernel weight at position
        (0,0) and input channel 0. Thus, it looks like this (where p is a padding value and i_xy is the input value at
        position (x,y)):

        p, p,        p,        p,        p, ...,        p
        p, p,        p,        p,        p, ...,        p
        p, p,     i_00,     i_01,     i_02, ..., i_0(w-2)
        p, p,     i_10,     i_11,     i_12, ..., i_1(w-2)
        p, p,     i_20,     i_21,     i_22, ..., i_2(w-2)
        ...,       ...,      ...,      ..., ...,     ...
        p, p, i_(h-2)0, i_(h-2)1, i_(h-2)2, ..., i_0(w-2)

        Note: only until h-2/w-2 because top right weight does not see more, i.e. is never convolved to the input at
        the bottom right.
        """
        x_rearranged = F.conv2d(x, self.W_rearrange, padding="same")
        return x_rearranged

    def _init_lateral_weights(self) -> Tensor:
        W_lateral = torch.zeros((self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1], 1, 1),
                                device=self.fabric.device, requires_grad=False)

        for co in range(self.out_channels):
            for ci in range(self.in_channels):
                if ci == co:
                    cii = ci * self.kernel_size[0] * self.kernel_size[1] + self.locality_size * self.kernel_size[
                        1] + self.locality_size
                    W_lateral[co, cii, 0, 0] = 1

        return W_lateral

    def hebbian_update(self, x: Tensor, y: Tensor):
        """
        Update the weights according to the Hebbian rule.
        :param x: Input tensor of shape (batch_size, in_channels * kernel_size[0] * kernel_size[1], height, width).
        :param y: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # assert False, "Not implemented yet"
        assert torch.all((x == 0.) | (x == 1.)), "x not binary"
        assert torch.all((y == 0.) | (y == 1.)), "y not binary"
        x_v = x.permute(0, 2, 3, 1).reshape(-1, 1, x.shape[1])
        y_v = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1], 1)
        pos_co_activation = torch.matmul(y_v, x_v)
        # neg_co_activation = torch.matmul(y_v, 1 - x_v) + torch.matmul(1 - y_v, x_v)
        assert torch.all(pos_co_activation >= 0) and torch.all(pos_co_activation <= 1), "pos_co_activation not in [0,1]"
        # assert torch.all(neg_co_activation >= 0.) and torch.all(
        #     neg_co_activation <= 1), "neg_co_activation not in [0,1]"
        # assert not torch.any(
        #     (pos_co_activation > 0) * (neg_co_activation > 0)), "pos_co_activation and neg_co_activation overlap"

        if self.hebbian_rule == "vanilla":
            # update = torch.mean((pos_co_activation - neg_co_activation), dim=0)
            update = torch.mean(pos_co_activation, dim=0)
            # TODO: is this normalization necessary?
            # update.reshape((self.out_channels, self.in_channels) + self.kernel_size)
            # why is there a value somwhere expect the diagonal??
            update = torch.where(update > 0., update, 0.)
            update = (update - update.min()) / (update.max() - update.min() + 1e-10)
            updated_weights = torch.where(update.reshape((self.out_channels, self.in_channels) + self.kernel_size) > 0)
            #if len(torch.where((updated_weights[0] != updated_weights[1]) & ((4+updated_weights[0]) != updated_weights[1]))[0]) > 0:
            #    print("updated_weights not diagonal")

            self.W_lateral.data += self.lr * update.view(self.W_lateral.shape)
            self.W_lateral.data = self.W_lateral.data / (1e-10 + torch.sqrt(
                torch.sum(self.W_lateral.data ** 2, dim=[1, 2, 3], keepdim=True)))  # Weight normalization
        else:
            raise NotImplementedError(f"Hebbian rule {self.hebbian_rule} not implemented.")

    def make_gaussian(self, size, fwhm=3, center=None):
        x = torch.arange(0, size, 1, dtype=torch.float)
        y = x.unsqueeze(1)
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return torch.exp(-4 * torch.log(torch.Tensor([2])) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        with torch.no_grad():
            x_rearranged = self.rearrange_input(x)

            # TODO: What happens if we set the actual input =0 during training and timestep > 0? -> we could try

            assert torch.all((x_rearranged == 0.) | (x_rearranged == 1.)), "x_rearranged not binary"
            x_lateral = F.conv2d(x_rearranged, self.W_lateral, padding="same")

            # Normalize by dividing through the max. possible activation (if all weights were 1)
            x_max_act = F.conv2d(x_rearranged, torch.ones_like(self.W_lateral.data), padding="same")
            min_support = self.kernel_size[0]
            # x_max_act[x_max_act > 0] = torch.where(x_max_act[x_max_act > 0] < min_support, min_support,
            # x_max_act[x_max_act > 0])
            x_max_act = torch.where(x_max_act < min_support, min_support, x_max_act)
            x_lateral_norm = x_lateral / x_max_act

            # TODO: Delme
            # if not hasattr(self, "dist_filter"):
            #     self.dist_filter = self.makeGaussian(7, 5).to(self.W_lateral.device)
            # delme_weight = (2 - torch.ones_like(self.W_lateral.data).reshape((self.out_channels, self.in_channels)
            # + self.kernel_size) * self.dist_filter.view((1,1) + self.kernel_size)).reshape(self.W_lateral.shape)
            # x_lateral_norm = (x_lateral / (1e-10+F.conv2d(x_rearranged, delme_weight, padding="same")))
            x_lateral_norm = x_lateral_norm / (1e-10 + torch.sum(self.W_lateral.data, dim=(1, 2, 3)).view(1, -1, 1, 1))

            # reduce weight at a certain point if it is too high (does not help a lot...)
            # x_lateral_norm = torch.where(10 * x_lateral_norm <= 1, 10 * x_lateral_norm, 1 - (x_lateral_norm * 10 - 1))

            x_lateral_norm_s = x_lateral_norm.shape
            x_lateral_norm /= (1e-10 + x_lateral_norm.view(-1, x_lateral_norm_s[2] * x_lateral_norm_s[3]).max(1)[0].view(
                x_lateral_norm_s[:2] + (1, 1)))

            # TODO: Test with / without average (two lines below)
            # TODO: Test with using x_lateral_bin_prev instead of x_lateral_norm_prev
            if self.ts > 0:
                x_lateral_norm = (x_lateral_norm + self.ts * self.x_lateral_norm_prev) / (self.ts + 1)
            self.x_lateral_norm_prev = x_lateral_norm

            # TODO: Over timesteps; increase probability at beginning a little bit and slightly decrease this
            #  additional boost over time -> sparsity over time
            # TODO: In weights of lateral support: Mask out center pixel so that it only depends on the neighborhood
            x_lateral_bin = (x_lateral_norm ** 3 >= 0.5).float()
            # x_lateral_bin = torch.bernoulli(torch.clip(x_lateral_norm ** 5, 0, 1))

            # TODO:
            # if self.training and self.ts == 4:
            if self.training:
                self.hebbian_update(x_rearranged, x_lateral_bin)

            stats = {
                "l1/avg_support_active": x_lateral[x_lateral_bin > 0].mean().item(),
                "l1/std_support_active": x_lateral[x_lateral_bin > 0].std().item(),
                "l1/min_support_active": x_lateral[x_lateral_bin > 0].min().item() if torch.sum(x_lateral_bin > 0) > 0 else 0,
                "l1/max_support_active": x_lateral[x_lateral_bin > 0].max().item() if torch.sum(x_lateral_bin > 0) > 0 else 0,
                "l1/avg_support_inactive": x_lateral[x_lateral_bin <= 0].mean().item(),
                "l1/std_support_inactive": x_lateral[x_lateral_bin <= 0].std().item(),
                "l1/min_support_inactive": x_lateral[x_lateral_bin <= 0].min().item() if torch.sum(x_lateral_bin <= 0) > 0 else 0,
                "l1/max_support_inactive": x_lateral[x_lateral_bin <= 0].max().item() if torch.sum(x_lateral_bin <= 0) > 0 else 0,
                "l1/norm_factor": torch.mean(x_lateral / (1e-10 + x_lateral_norm)).item()
            }

            x_lateral /= x_lateral.view(-1, x_lateral_norm_s[2] * x_lateral_norm_s[3]).max(1)[0].view(
                x_lateral_norm_s[:2] + (1, 1))
            return torch.cat([x_lateral, x_lateral_norm], dim=1), x_lateral_bin, stats


class LateralLayerEfficient(nn.Module):

    def __init__(self,
                 fabric: Fabric,
                 in_channels: int,
                 out_channels: int,
                 locality_size: Optional[int] = 2,
                 lr: Optional[float] = 0.0005,
                 mask_out_bg: Optional[bool] = False,
                 mov_avg_rate_prev_view: Optional[float] = 0.,
                 mov_avg_rate_prev_time: Optional[float] = 0.,
                 mov_avg_rate_prev: Optional[float] = 0.,
                 hebbian_rule: Optional[HEBBIAN_ALGO] = 'instar',
                 use_bias: Optional[bool] = True,
                 k: Optional[int] = 1,
                 thr_rate: Optional[float] = 0.05,
                 target_rate: Optional[float] = None,
                 w_init: Optional[W_INIT] = 'identity',
                 ):
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.
        Source: https://github.com/ThomasMiconi/HebbianCNNPyTorch/tree/main

        :param fabric: Fabric instance.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param locality_size: Size of the locality, i.e. how many neurons are connected to each other. For
        example, if locality_size = 2, then each neuron is connected to 5 neurons on each side of it.
        :param lr: Learning rate.
        :param mask_out_bg: Whether to mask out the background.
        :param mov_avg_rate_prev_view: The rate of the previous view at the same timestep to use for moving average.
        :param mov_avg_rate_prev_time: The rate of the previous timestep and the same view to use for moving average.
        :param mov_avg_rate_prev: The rate of the previous activation to use for moving average.
        :param hebbian_rule: Which Hebbian rule to use.
        :param use_bias: Whether to use bias.
        :param k: Number of active neurons per channel (k winner take all).
        :param thr_rate: The threshold rate, defines how fast the bias adapts to data. Only relevant if `use_bias=True`
        :param target_rate: The target rate of the neurons (how many should be activate), default is k/out_channels.
                            Only relevant if `use_bias=True`
        :param w_init: How to initialize the weights.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.locality_size = locality_size
        self.neib_size = 2 * self.locality_size + 1
        self.lr = lr
        self.mask_out_bg = mask_out_bg
        self.hebbian_rule = hebbian_rule
        self.use_bias = use_bias
        self.k = k
        self.mov_avg_rate_prev_view = mov_avg_rate_prev_view
        self.mov_avg_rate_prev_time = mov_avg_rate_prev_time
        self.mov_avg_rate_prev = mov_avg_rate_prev
        self.thr_rate = thr_rate
        self.target_rate = self.k / self.out_channels if target_rate is None else target_rate

        self.step = 0
        self.ts = None
        self.prev_activations = {}
        self.prev_activation = None

        if isinstance(self.target_rate, list):
            self.target_rate = torch.tensor(self.target_rate, device=fabric.device).view(1, -1, 1, 1)

        assert self.k <= self.out_channels, "k must be smaller than out_channels"
        assert self.k > 0, "k must be greater than 0"

        if self.mask_out_bg:
            self.target_rate = self.target_rate / 10

        weight_shape = (self.out_channels, self.in_channels, self.neib_size, self.neib_size)
        if w_init == "random":
            self.W = torch.randn(weight_shape, requires_grad=True, device=fabric.device)
        elif w_init == "zeros":
            self.W = torch.zeros(weight_shape, requires_grad=True, device=fabric.device)
        elif w_init == "identity":
            self.W = torch.zeros(weight_shape, device=fabric.device)
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    if oc == ic:
                        self.W[oc, ic, self.locality_size, self.locality_size] = 1
            self.W.requires_grad = True

        self.W.data = self.W.data / (1e-10 + torch.sqrt(torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))
        self.b = torch.zeros((1, self.out_channels, 1, 1), requires_grad=False).to(fabric.device)

        self.W = nn.Parameter(self.W)
        self.b = nn.Parameter(self.b)

        self.optimizer = torch.optim.Adam([self.W, ], lr=self.lr)

    def new_sample(self):
        """
        To be called when a new sample is fed into the network.
        """
        self.prev_activations = {}
        self.prev_activation = None

    def update_ts(self, ts):
        """
        Set the current timestep (relevant for sparsity rate)
        :param ts: The current timestep
        """
        self.ts = ts

    def update_k(self, k):
        self.k = k
        self.target_rate = self.k / self.out_channels
        assert False, "Should this be called?"

    def get_weights(self):
        return self.W

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()

        prelimy = F.conv2d(x, self.W, padding="same")
        prelimy = prelimy * (prelimy > 0.6)

        # Then we compute the "real" output (y) of each cell, with winner-take-all competition
        with torch.no_grad():
            realy = (prelimy - self.b)

            if self.mov_avg_rate_prev_view > 0 and self.ts in self.prev_activations:
                realy = (1 - self.mov_avg_rate_prev_view) * realy + self.mov_avg_rate_prev_view * \
                        self.prev_activations[self.ts]
            if self.mov_avg_rate_prev_time > 0 and self.ts - 1 in self.prev_activations:
                realy = (1 - self.mov_avg_rate_prev_time) * realy + self.mov_avg_rate_prev_time * \
                        self.prev_activations[self.ts - 1]
            if self.mov_avg_rate_prev and self.prev_activation is not None:
                realy = (1 - self.mov_avg_rate_prev) * realy + self.mov_avg_rate_prev * self.prev_activation
            self.prev_activation = realy.detach()
            self.prev_activations[self.ts] = realy.detach()

            # TODO: In order that the channels have more distinct features, we could limit the number of activations
            #  per channel to 1.5x the input activations of the same channel?
            # TODO: We could use different Filters, e.g. some layers can access only some input filters (number of
            #  combinations: 2^in_channels)

            internal_activations = realy.detach()

            # # k winner take all
            # smallest_value_per_channel = torch.amin(realy, dim=(2, 3)) + 0.0001
            # tk = torch.topk(realy.data, self.k, dim=1, largest=True)[0]
            # realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
            # if self.mask_out_bg:
            #     realy.data[realy.data <= smallest_value_per_channel.view(
            #         smallest_value_per_channel.shape + (1, 1))] = 0  # mask out background...
            #
            # realy.data = (realy.data > 0.).float()

            realy.data = (realy.data > 0.6).float()  # torch.bernoulli(realy.data)

            # Adaptive Threshold
            # if self.step <= 100_000:
            #     ratio = max((100_000 - self.step) / 100_000, 0)
            #     realy.data = ((x[:, :x.shape[1] // 2, ...] * ratio + realy.data * (1 - ratio)) > 0.5).float()
        #
        #     realy.data = (realy.data > 0.).float()

        # # binary output
        # threshold = (torch.sort(realy.view(realy.shape[0], -1), dim=1, descending=True)[0][:,
        #              int(realy.numel() / realy.shape[0] / realy.shape[1] * 0.2)])
        #
        # realy.data = (realy.data > threshold.view(realy.shape[0], 1, 1, 1)).float()

        # Then we compute the surrogate output yforgrad, whose gradient computations produce the desired Hebbian output
        # Note: We must not include thresholds here, as this would not produce the expected gradient expressions. The
        # actual values will come from realy, which does include thresholding.
        if self.hebbian_rule == "instar":
            yforgrad = prelimy - 1 / 2 * torch.sum(self.W * self.W, dim=(1, 2, 3))[None, :, None,
                                         None]  # Instar rule, dw ~= y(x-w)
        elif self.hebbian_rule == "oja":
            yforgrad = prelimy - 1 / 2 * torch.sum(self.W * self.W, dim=(1, 2, 3))[None, :, None,
                                         None] * realy.data  # Oja's rule, dw ~= y(x-yw)
        elif self.hebbian_rule == "vanilla":
            yforgrad = prelimy
        else:
            assert False, "Unknown hebbian rule"

        yforgrad.data = realy.data  # We force the value of yforgrad to be the "correct" y

        loss = torch.sum(-1 / 2 * yforgrad * yforgrad)
        if self.training:
            loss.backward()
            if self.step > 100:  # No weight modifications before batch 100 (burn-in) or after learning
                # epochs (during data accumulation for training / testing)
                self.optimizer.step()
                self.W.data = self.W.data / (1e-10 + torch.sqrt(
                    torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))  # Weight normalization

        if self.use_bias:
            with torch.no_grad():
                # Threshold adaptation is based on realy, i.e. the one used for plasticity. Always binarized (firing vs.
                # not firing).
                self.b += self.thr_rate * (
                        torch.mean((realy.data > 0).float(), dim=(0, 2, 3))[None, :, None, None] - self.target_rate)

        self.step += 1
        return internal_activations, realy.detach()


class LateralLayerEfficientNetwork1L(nn.Module):
    """
    A model with two lateral layers. The first one increases the number of channels from [channels of the feature
    extractor] to [channels of the lateral network]
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dict.
        :param fabric: Fabric instance.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        self.avg_value_meter = {}

        lm_conf = self.conf["lateral_model"]
        self.out_channels = self.conf["lateral_model"]["channels"]
        self.in_channels = self.conf["feature_extractor"]["out_channels"] + self.out_channels
        if self.conf["feature_extractor"]["add_bg_channel"]:
            self.in_channels += 1
            self.out_channels += 1

        if lm_conf["l1_type"] == "lateral_efficient":
            l1_t = LateralLayerEfficient
        elif lm_conf["l1_type"] == "lateral_flex":
            l1_t = LateralLayer
        else:
            assert False, "Unknown lateral layer type"

        self.l1 = l1_t(
            self.fabric,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **lm_conf["l1_params"],
        )

    def new_sample(self):
        self.l1.new_sample()

    def update_ts(self, ts):
        self.l1.update_ts(ts)

    def get_layer_weights(self):
        return {"L1": self.l1.get_weights()}

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass over multiple timesteps
        :param x: The input tensor (extracted features of the image)
        :return: Extracted features thar were fed into the lateral layer (binarized version of features from feature
        extractor), Features extracted by the lateral layers (binarized), Features extracted by the lateral layers (as
        float)
        """
        # TODO: New
        act, act_bin, logs = self.l1(x)
        for k, v in logs.items():
            if k not in self.avg_value_meter:
                self.avg_value_meter[k] = AverageMeter()
            self.avg_value_meter[k](v, weight=x.shape[0])
        return act, act_bin

        # # TODO: Move this loop into s1_toy_example
        # self.new_sample()
        # z = None

    #
    # input_features, lateral_features, lateral_features_f = [], [], []
    # for view_idx in range(x.shape[1]):
    #     # prepare input view
    #     x_view = x[:, view_idx, ...]
    #     x_view = torch.where(x_view > 0., 1., 0.)
    #     input_features.append(x_view)
    #
    #     if z is None:
    #         z = torch.zeros((x_view.shape[0], self.l1.out_channels, x_view.shape[2], x_view.shape[3]),
    #                         device=x.device)
    #
    #     t = 0
    #     features, features_float = [], []
    #     for t in range(self.conf["lateral_model"]["max_timesteps"]):
    #         self.l1.update_ts(t)
    #         x_in = torch.cat([x_view, z], dim=1)
    #         z_float, z = self.l1(x_in)
    #         features.append(z)
    #         features_float.append(z_float)
    #
    #     lateral_features.append(torch.stack(features, dim=1))
    #     lateral_features_f.append(torch.stack(features_float, dim=1))
    #
    # return torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), torch.stack(lateral_features_f,
    #                                                                                             dim=1)

    def get_model_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics of the model weights.
        :return: Dictionary with statistics.
        """
        stats = {}
        for layer, weight in self.get_layer_weights().items():
            non_zero_mask = weight != 0
            stats_ = {
                f"l1/weight_mean_{layer}": torch.mean(weight).item(),
                f"l1/weight_std_{layer}": torch.std(weight).item(),
                f"l1/weight_mean_{layer}_(0_ignored)": (
                        torch.sum(weight * non_zero_mask) / torch.sum(non_zero_mask)).item(),
                f"l1/weight_min_{layer}": torch.min(weight).item(),
                f"l1/weight_max_{layer}": torch.max(weight).item(),
                f"l1/weight_above_0.9_{layer}": torch.sum(weight >= 0.9).item() / weight.numel(),
                f"l1/weight_below_0.1_{layer}": torch.sum(weight <= 0.1).item() / weight.numel(),
            }
            stats = stats | stats_
        return stats

    def get_and_reset_logs(self) -> Dict[str, float]:
        """
        Get logs for the current epoch.
        :return: Dictionary with logs.
        """
        weight_stats = self.get_model_weight_stats()
        metrics = {}
        for k, v in self.avg_value_meter.items():
            metrics[k] = v.mean
            v.reset()

        return weight_stats | metrics


class LateralNetwork(pl.LightningModule):
    """
    PyTorch Lightning module for a network with a single lateral layer.
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.model.forward(x)

    def new_sample(self):
        self.model.new_sample()

    def update_ts(self, ts):
        self.model.update_ts(ts)

    def get_and_reset_logs(self) -> Dict[str, float]:
        return self.model.get_and_reset_logs()

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
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(title)

        files = []
        for layer, weight in self.model.get_layer_weights().items():
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(axs[1], weight[:20, :20, ...].detach().cpu(), f"Example Weight matrix ({layer})")
            plt.tight_layout()

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None:
                fp = Path(fig_fp) / f'weights_{layer}.png'
                plt.savefig(fp)
                files.append(fp)

            if show_plot:
                plt.show()

            plt.close()
        return files

    def plot_samples(self,
                     img: List[Tensor],
                     features: List[Tensor],
                     input_features: List[Tensor],
                     lateral_features: List[Tensor],
                     lateral_features_f: List[Tensor],
                     plot_input_features: Optional[bool] = True,
                     show_plot: Optional[bool] = False,
                     ) -> List[Path]:
        """
        Plot the features extracted from a single sample.
        :param img: A list of samples, i.e. of batches of the original images (shape is (B, V, C, H, W) where V is
        the number of augmented views).
        :param features: A list of samples, i.e. batches of the features extracted from the image (shape is (B, V, C,
        H, W) where V is the number of augmented views and C is the number of output channels from the feature
        extractor).
        :param input_features: A list of samples, i.e. binary version of the features that were actually fed into the
        lateral network (shape is (B, V, C, H, W) where V is the number of augmented views and C is the number of
        output channels from the feature extractor).
        :param lateral_features: A list of samples, i.e. batches of the features extracted from the lateral network
        over time (shape is (B, V, T, C, H, W) where V is the number of augmented views, T is the number of
        timesteps, and C is the number of output channels from the lateral layer).
        lateral_features_f: Same as lateral_features but the intermediate float value before binarization.
        :param plot_input_features: Whether to plot the input features or not.
        :param show_plot: Whether to show the plot or not.
        :return: List of paths to the generated plots.
        """

        def _plot_input_features(img,
                                 features,
                                 input_features,
                                 fig_fp: Optional[str] = None,
                                 show_plot: Optional[bool] = False):
            plt_images, plt_titles = [], []
            for view_idx in range(img.shape[0]):
                plt_images.append(img[view_idx])
                plt_titles.append(f"Input view {view_idx}")
                for feature_idx in range(features.shape[1]):
                    plt_images.append(features[view_idx, feature_idx])
                    plt_titles.append(f"Features V={view_idx} C={feature_idx}")
                    plt_images.append(input_features[view_idx, feature_idx])
                    plt_titles.append(f"Lateral Input V={view_idx} C={feature_idx}")
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=2 * features.shape[1] + 1, plot_colorbar=True,
                        vmin=0, vmax=1, fig_fp=fig_fp, show_plot=show_plot)

        def _plot_lateral_activation_map(lateral_features,
                                         fig_fp: Optional[str] = None,
                                         show_plot: Optional[bool] = False):
            max_views = 3
            plt_images, plt_titles = [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                for time_idx in range(lateral_features.shape[1]):
                    for feature_idx in range(lateral_features.shape[2]):
                        plt_images.append(lateral_features[view_idx, time_idx, feature_idx])
                        plt_titles.append(
                            f"Lat. L={1 if time_idx == 0 else 2} V={view_idx} T={time_idx} C={feature_idx}")
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=lateral_features.shape[2], plot_colorbar=True,
                        vmin=0, vmax=1, fig_fp=fig_fp, show_plot=show_plot)

        def _plot_lateral_heat_map(lateral_features_f, fig_fp: Optional[str] = None, show_plot: Optional[bool] = False):
            max_views = min(3, lateral_features_f.shape[0])
            v_min, v_max = lateral_features_f[:max_views].min(), lateral_features_f[:max_views].max()
            plt_images, plt_titles = [], []
            for view_idx in range(max_views):
                for time_idx in range(lateral_features_f.shape[1]):
                    for feature_idx in range(lateral_features_f.shape[2]):
                        plt_images.append(lateral_features_f[view_idx, time_idx, feature_idx])
                        plt_titles.append(
                            f"L={1 if time_idx == 0 else 2} V={view_idx} T={time_idx} C={feature_idx}")
            plot_images(images=plt_images, titles=plt_titles, max_cols=lateral_features_f.shape[2], plot_colorbar=True,
                        fig_fp=fig_fp, cmap='hot', interpolation='nearest', vmin=v_min, vmax=v_max, show_plot=show_plot)

        def _plot_lateral_output(img,
                                 lateral_features,
                                 fig_fp: Optional[str] = None,
                                 show_plot: Optional[bool] = False):
            max_views = 10
            plt_images, plt_titles, plt_masks = [], [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                plt_images.extend([img[view_idx], img[view_idx]])
                plt_titles.extend([f"Input view {view_idx}", f"Extracted Features {view_idx}"])
                background = torch.all((lateral_features[view_idx, -1] == 0), dim=0)
                foreground = torch.argmax(lateral_features[view_idx, -1], dim=0)
                calc_mask = torch.where(~background, foreground + 1, 0.)
                plt_masks.extend([None, calc_mask])
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, masks=plt_masks, max_cols=2, plot_colorbar=True,
                        vmin=0, vmax=1, mask_vmin=0, mask_vmax=lateral_features.shape[2] + 1, fig_fp=fig_fp,
                        show_plot=show_plot)

        fig_fp = self.conf['run']['plots'].get('store_path', None)
        files = []
        for i, (img_i, features_i, input_features_i, lateral_features_i, lateral_features_f_i) in enumerate(
                zip(img, features, input_features, lateral_features, lateral_features_f)):
            for batch_idx in range(img_i.shape[0]):
                if fig_fp is not None:
                    fig_fp = Path(fig_fp)
                    base_name = f"sample_{i}_batch_idx_{batch_idx}"
                    if_fp = fig_fp / f'{base_name}_input_features.png'
                    am_fp = fig_fp / f'{base_name}_lateral_act_maps.png'
                    hm_fp = fig_fp / f'{base_name}_lateral_heat_maps.png'
                    lo_fp = fig_fp / f'{base_name}_lateral_output.png'
                    files.extend([if_fp, am_fp, hm_fp, lo_fp])
                else:
                    if_fp, am_fp, hm_fp, lo_fp = None, None, None, None
                if plot_input_features:
                    _plot_input_features(img_i[batch_idx], features_i[batch_idx], input_features_i[batch_idx],
                                         fig_fp=if_fp, show_plot=show_plot)
                elif if_fp is not None:
                    files.remove(if_fp)
                _plot_lateral_activation_map(lateral_features_i[batch_idx],
                                             fig_fp=am_fp, show_plot=show_plot)
                _plot_lateral_heat_map(lateral_features_f_i[batch_idx],
                                       fig_fp=hm_fp, show_plot=show_plot)
                _plot_lateral_output(img_i[batch_idx], lateral_features_i[batch_idx],
                                     fig_fp=lo_fp, show_plot=show_plot)

        return files

    def create_activations_video(self,
                                 images: List[Tensor],
                                 features: List[List[Tensor]],
                                 activations: List[List[List[Tensor]]]) -> List[str]:
        """
        Create a video of the activations.
        :param images: The original images.
        :param features: The features extracted from the images.
        :param activations: The activations after the lateral connections.
        :return: A list of paths to the created videos.
        """
        folder = Path("../tmp") / "video_toys"
        if not folder.exists():
            folder.mkdir(parents=True)

        c = 0

        videos_fp = []
        for img_idx in range(len(images)):
            img = images[img_idx]
            features_img = features[img_idx]
            activations_img = activations[img_idx]

            for batch_idx in range(img.shape[0]):

                for view_idx in range(features_img.shape[2]):
                    img_view = img[batch_idx]
                    features_view = features_img[batch_idx, :, view_idx]
                    activations_view = activations_img[batch_idx, :, :, view_idx]

                    for time_idx in range(len(activations_view)):
                        activations_view_time = activations_view[time_idx]
                        features_view_dec = bin2dec(features_view.permute(1, 2, 0))
                        activations_view_time_dec = bin2dec(activations_view_time.permute(1, 2, 0))

                        plot_images(images=[img_view] * 3,
                                    show_plot=False,
                                    fig_fp=str(folder / f"{c:04d}.png"),
                                    titles=["Image", "Inp. Features", "Activations"],
                                    suptitle=f"Image {img_idx}.{batch_idx}, View {view_idx}, Time {time_idx}",
                                    masks=[None, features_view_dec, activations_view_time_dec],
                                    max_cols=3, plot_colorbar=True, vmin=0, vmax=1, mask_vmin=0, mask_vmax=2 ** 4)
                        c += 1
                fig_fp = self.conf['run']['plots'].get('store_path', None)
                fig_fp = Path(fig_fp) if fig_fp is not None else folder
                # video_fp = f"{fig_fp / datetime.now().strftime('%Y-%d-%m_%H-%M-%S')}_{img_idx}_{batch_idx}.mp4"
                video_fp = f"{fig_fp / 'activations'}_img_{img_idx}_batch_{batch_idx}.mp4"
                create_video_from_images_ffmpeg(folder, video_fp)
                videos_fp.append(Path(video_fp))

                for f in folder.glob("*.png"):
                    f.unlink()
        return videos_fp

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        return LateralLayerEfficientNetwork1L(self.conf, self.fabric)

    def on_epoch_end(self):
        logs = self.get_and_reset_logs()
        self.log_dict(logs)
        print_logs(logs)

    def _normalize_image_list(self, img_list):
        img_list = torch.stack([i.squeeze() for i in img_list])
        img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min() + 1e-9)
        img_list = [img_list[i] for i in range(img_list.shape[0])]
        return img_list
