from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
import lightning.pytorch as pl
import numpy as np
from torch.autograd import Variable
from torchvision import utils

from data import plot_images
from tools import AverageMeter
from utils import print_logs


# TODO: In jedem timestep muss ursprüngliche Features berücksichtigt werden!
# TODO: änhliche Linien über mehrere Timesteps
# TODO: In jedem timestep geht die Sparsity herunter bis neues Bild ingelesen wird


class LateralLayerEfficient(nn.Module):

    def __init__(self,
                 fabric: Fabric,
                 in_channels: int,
                 out_channels: int,
                 locality_size: Optional[int] = 5,
                 lr: Optional[float] = 0.01,
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
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.locality_size = locality_size
        self.neib_size = 2 * self.locality_size + 1
        self.lr = lr
        self.k = 10
        self.thr_rate = 10.
        self.target_rate = self.k / self.out_channels
        self.train_ = True
        self.step = 0
        self.ts = None
        self.prev_activations = {}

        self.W = torch.randn((self.out_channels, self.in_channels, self.neib_size, self.neib_size), requires_grad=True,
                             device=fabric.device)
        self.W.data = self.W.data / (1e-10 + torch.sqrt(torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))
        self.b = torch.zeros((1, self.out_channels, 1, 1), requires_grad=False).to(fabric.device)

        self.optimizer = torch.optim.Adam([self.W, ], lr=self.lr)

    def new_sample(self):
        """
        To be called when a new sample is fed into the network.
        """
        self.prev_activations = {}

    def update_ts(self, ts):
        """
        Set the current timestep (relevant for sparsity rate)
        :param ts: The current timestep
        """
        self.ts = ts
        self.update_k(5 - ts)  # TODO: replace 5 with max. timestep

    def update_k(self, k):
        self.k = k
        self.target_rate = self.k / self.out_channels

    def forward(self, x: Tensor) -> Tensor:
        self.optimizer.zero_grad()

        prelimy = F.conv2d(x, self.W, padding="same")

        # Then we compute the "real" output (y) of each cell, with winner-take-all competition
        with torch.no_grad():
            realy = (prelimy - self.b)

            # TODO: How to limit the activations locally?
            mask = torch.any(x > 0, dim=(1), keepdim=True)  # where something in the input is active -> limit to this range -> BAD?
            realy = realy * mask  # remove all activations where input was not active

            if self.ts in self.prev_activations:
                realy = .6 * realy + .4 * self.prev_activations[self.ts]
                self.prev_activations[self.ts] = realy
            else:
                self.prev_activations[self.ts] = realy

            tk = torch.topk(realy.data, self.k, dim=1, largest=True)[0]
            realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
            realy.data = (realy.data > 0).float()

        # Then we compute the surrogate output yforgrad, whose gradient computations produce the desired Hebbian output
        # Note: We must not include thresholds here, as this would not produce the expected gradient expressions. The
        # actual values will come from realy, which does include thresholding.
        yforgrad = prelimy - 1 / 2 * torch.sum(self.W * self.W, dim=(1, 2, 3))[None, :, None, None]
        yforgrad.data = realy.data  # We force the value of yforgrad to be the "correct" y

        loss = torch.sum(-1 / 2 * yforgrad * yforgrad)
        if self.train_:
            loss.backward()
        if self.step > 100 and self.train_:  # No weight modifications before batch 100 (burn-in) or after learning
            # epochs (during data accumulation for training / testing)
            self.optimizer.step()
            self.W.data = self.W.data / (1e-10 + torch.sqrt(
                torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))  # Weight normalization

        with torch.no_grad():
            # Threshold adaptation is based on realy, i.e. the one used for plasticity. Always binarized (firing vs.
            # not firing).
            self.b += self.thr_rate * (
                    torch.mean((realy.data > 0).float(), dim=(0, 2, 3))[None, :, None, None] - self.target_rate)

        self.step += 1
        return realy.detach()


class LateralLayerToy(nn.Module):

    def __init__(self,
                 fabric: Fabric,
                 n_channels: int,
                 locality_size: Optional[int] = 5,
                 alpha: Optional[float] = 0.9,
                 lr: Optional[float] = 0.01,
                 ):
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.
        :param fabric: Fabric instance.
        :param n_channels: Number of channels in the input and output.
        :param locality_size: Size of the locality, i.e. how many neurons are connected to each other. For 
        example, if locality_size = 2, then each neuron is connected to 5 neurons on each side of it.
        :param alpha: Factor by which the activation probability is normalized (lower values lead to higher
        activation probabilities).
        :param lr: Learning rate.
        """
        super().__init__()
        self.n_channels = n_channels
        self.locality_size = locality_size
        self.alpha = alpha
        self.lr = lr
        # weight has the shape (output_channels, input_channels, kernel_height, kernel_width)
        self.W = torch.rand((self.n_channels, self.n_channels, 2 * self.locality_size + 1, 2 * self.locality_size + 1),
                            dtype=torch.float32, requires_grad=False).to(fabric.device)
        self.train_ = True
        self.step = 0

    def update_weights(self, input: Tensor, output: Tensor, probabilities: Tensor):
        """
        Updates the weights of the layer (i.e. a training step).
        :param input: Layer input.
        :param output: Layer output.
        :param probabilities: Activation probabilities of the output.
        """
        self.step += 1
        patch_size = self.locality_size * 2 + 1

        prob_shaped = probabilities.reshape(probabilities.shape[:2] + (-1,))
        winner_val, winner_idx = torch.topk(prob_shaped, int(32 * 32 * 0.01), dim=(-1))
        winner_mask = torch.zeros_like(prob_shaped).scatter(-1, winner_idx, torch.ones_like(winner_val)).reshape(
            probabilities.shape)

        input_masked = input * winner_mask
        output_masked = output * winner_mask

        input_patches = F.pad(input_masked,
                              (self.locality_size, self.locality_size, self.locality_size, self.locality_size)) \
            .unfold(2, patch_size, 1).unfold(3, patch_size, 1).unsqueeze(2)
        output_reshaped = output_masked.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, patch_size,
                                                                           patch_size).unsqueeze(1)
        pos_co_activation = torch.matmul(input_patches, output_reshaped)
        neg_co_activation = torch.matmul(input_patches, output_reshaped - 1) + torch.matmul(input_patches - 1,
                                                                                            output_reshaped)

        update = torch.mean(pos_co_activation + neg_co_activation, dim=(0, 3, 4))
        self.W += self.lr * (
            update)  # - 10 * self.W * torch.mean(output_reshaped, dim=(0, 3, 4)).permute(1, 0, 2, 3) ** 2)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        :param input: Layer input.
        :return: Layer output.
        """
        weight = F.tanh(self.W)
        a = F.conv2d(input, weight, groups=4, padding="same")  # TODO: check group parameter
        z = torch.sum(weight, dim=(1, 2, 3))
        p = a / (self.alpha * z.view(1, self.n_channels, 1, 1))
        # output = torch.bernoulli(p.clip(0., 1.))
        output = torch.where(p > 0.5, 1., 0.)

        if self.train_:
            self.update_weights(input, output, p)

        return output


class LateralLayerEfficientNetwork(nn.Module):
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
        lm_conf = self.conf["lateral_model"]

        self.l1 = LateralLayerEfficient(
            self.fabric,
            in_channels=self.conf["feature_extractor"]["out_channels"],
            out_channels=lm_conf["channels"],
            locality_size=lm_conf['locality_size'],
            lr=lm_conf['lr']
        )
        self.l2 = LateralLayerEfficient(
            self.fabric,
            in_channels=lm_conf["channels"],
            out_channels=lm_conf["channels"],
            locality_size=lm_conf['locality_size'],
            lr=lm_conf['lr']
        )

    def new_sample(self):
        self.l1.new_sample()
        self.l2.new_sample()

    def get_layer_weights(self):
        return {"L1": self.l1.W, "L2": self.l2.W}

    def set_train(self, train: bool):
        self.l1.train_ = train
        self.l2.train_ = train


    def forward(self, x: Tensor) -> Tuple[List[int], List[Tensor], int]:
        """
        Forward pass over multiple timesteps
        :param x: The input tensor (extracted features of the image)
        :return: List of changes, List of features extracted by the lateral layers, number of timesteps
        """
        t = 0

        self.l1.update_ts(0)

        changes, features = [], []
        x = self.l1(x)
        features.append(x)
        for t in range(self.conf["lateral_model"]["max_timesteps"]):
            self.l2.update_ts(t)
            x_old = x
            x = self.l2(x)
            change = F.l1_loss(x_old, x).item()
            changes.append(change)
            features.append(x)
            # if change < self.conf["lateral_model"]["change_threshold"]:
            #     break
        return changes, features, t


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
        self.avg_meter_t = AverageMeter()

    def forward(self, x: Tensor) -> Tuple[List[int], List[Tensor], int]:
        """
        Forward pass through the model.
        :param x: Input image.
        :return:  List of changes, List of features extracted by the lateral layers, number of timesteps
        """
        return self.model(x)

    def forward_steps_multiple_views_through_time(self, x: Tensor) -> Tuple[
        List[Tensor], List[List[Tensor]], List[float]]:
        """
        Forward pass through the model over multiple timesteps. The number of timesteps depends on the change
        threshold and the maximum number of timesteps (i.e. the forward pass through time is cancel if the activations'
        change is below a given threshold).
        :param x: Features extracted from the image.
        :return: List of features extracted by the feature extractor, List of features build by lateral connections,
        list of changes
        """
        # if x.unique().shape[0] > 2 or x.unique(sorted=True) != torch.tensor([0, 1]):
        # x = torch.bernoulli(x.clip(0, 1))
        # x = torch.where(x > 0.5, 1., 0.)

        self.model.new_sample()

        changes, input_features, lateral_features = [], [], []
        for view_idx in range(x.shape[1]):
            x_view = x[:, view_idx, ...]
            x_view = torch.where(x_view > 0., 1., 0.)
            input_features.append(x_view)
            changes_, features_, t = self.forward(x_view)
            changes.append(1.)
            changes.append(changes_)
            lateral_features.append(features_)
            self.avg_meter_t(t)

        return input_features, lateral_features, changes

    def get_logs(self) -> Dict[str, float]:
        """
        Get logs for the current epoch.
        :return: Dictionary with logs.
        """
        logs = {
            "avg_t": self.avg_meter_t.mean,
        }
        self.avg_meter_t.reset()
        return logs | self.get_model_weight_stats()

    def get_model_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics of the model weights.
        :return: Dictionary with statistics.
        """
        stats = {}
        for layer, weight in self.model.get_layer_weights().items():
            stats_ = {
                f"weight_mean_{layer}": torch.mean(weight).item(),
                f"weight_std_{layer}": torch.std(weight).item(),
                f"weight_min_{layer}": torch.min(weight).item(),
                f"weight_max_{layer}": torch.max(weight).item(),
                f"weight_above_0.9_{layer}": torch.sum(weight >= 0.9).item() / weight.numel(),
                f"weight_below_-0.9_{layer}": torch.sum(weight <= -0.9).item() / weight.numel(),
            }
            stats = stats | stats_
        return stats

    def plot_model_weights(self):
        """
        Plot a histogram of the model weights.
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
            weight_img_list = [weight[i, j].unsqueeze(0) for i in range(weight.shape[0]) for j in
                               range(weight.shape[1])]
            # Order is [(0, 0), (0, 1), ..., (3, 2), (3, 3)]
            # Each row shows the input weights per output channel
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True)
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(title)

        # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        # _hist_plot(axs[0], self.model.W.detach().cpu(), "Weight distribution (without tanh)")
        # _hist_plot(axs[1], F.tanh(self.model.W).detach().cpu(), "Weight distribution (with tanh)")
        # _plot_weights(axs[2], self.model.W.detach().cpu(), "Weight matrix")

        for layer, weight in {"L1": self.model.l1.W, "L2": self.model.l2.W}.items():
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(axs[1], weight[:20, :20, ...].detach().cpu(), f"Example Weight matrix ({layer})")
            plt.tight_layout()
            plt.show()

    def plot_features_single_sample(self, img: Tensor, features: Tensor):
        """
        Plot the features extracted from a single sample.
        :param img: The original image
        :param features: The features extracted from the image.
        """

        def _normalize_image_list(img_list):
            img_list = torch.stack([i.squeeze() for i in img_list])
            img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min())
            img_list = [img_list[i] for i in range(img_list.shape[0])]
            return img_list

        def _plot_input_features(img, features, input_features):
            plt_images, plt_titles = [], []
            for view_idx in range(img.shape[0]):
                plt_images.append(img[view_idx])
                plt_titles.append(f"Input view {view_idx}")
                for feature_idx in range(features.shape[1]):
                    plt_images.append(features[view_idx, feature_idx])
                    plt_titles.append(f"Features V={view_idx} C={feature_idx}")
                    plt_images.append(input_features[view_idx, feature_idx])
                    plt_titles.append(f"Lateral Input V={view_idx} C={feature_idx}")
            plt_images = _normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=2 * features.shape[1] + 1, plot_colorbar=True,
                        vmin=0, vmax=1)

        def _plot_lateral_activation_map(lateral_features):
            max_views = 1
            plt_images, plt_titles = [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                for time_idx in range(lateral_features.shape[1]):
                    for feature_idx in range(lateral_features.shape[2]):
                        plt_images.append(lateral_features[view_idx, time_idx, feature_idx])
                        plt_titles.append(
                            f"Lat. L={1 if time_idx == 0 else 2} V={view_idx} T={time_idx} C={feature_idx}")
            plt_images = _normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=lateral_features.shape[2], plot_colorbar=True,
                        vmin=0, vmax=1)

        def _plot_lateral_output(img, lateral_features):
            max_views = 10
            plt_images, plt_titles, plt_masks = [], [], []
            for view_idx in range(min(max_views, lateral_features.shape[0])):
                plt_images.extend([img[view_idx], img[view_idx]])
                plt_titles.extend([f"Input view {view_idx}", f"Extracted Features {view_idx}"])
                torch.argmax(lateral_features[view_idx, -1], dim=0)
                plt_masks.extend([None, torch.argmax(lateral_features[view_idx, -1], dim=0)])
            plot_images(images=plt_images, titles=plt_titles, masks=plt_masks, max_cols=2, plot_colorbar=True,
                        vmin=0, vmax=1)


        self.model.set_train(False)
        with torch.no_grad():
            input_features, lateral_features, changes = self.forward_steps_multiple_views_through_time(features)

        input_features = torch.stack(input_features, dim=1)
        lateral_features = torch.stack([torch.stack(f, dim=1) for f in lateral_features], dim=1)

        # select first sample of the batch
        img = img[0]
        features = features[0]
        input_features = input_features[0]
        lateral_features = lateral_features[0]

        _plot_input_features(img, features, input_features)
        _plot_lateral_activation_map(lateral_features)
        _plot_lateral_output(img, lateral_features)

        # n_images = min(view_idxes[-1] + 1, 3)
        # max_features_per_time = 10

    #
    # images, titles = [], []
    #
    # for img_idx in range(n_images):
    #     img_i = img[0, img_idx, ...]  # only use batch idx 0
    #     features_i = [f[0] for f, idx in zip(features, view_idxes) if idx == img_idx]
    #     images.append(img_i)
    #     titles.append(f"Input {img_idx + 1}")
    #
    #     for i in range(features_i[0].shape[0]):
    #         images.append(features_i[0][i])
    #         titles.append(f"Feature {i + 1} (t=0)")
    #
    #     for t in np.linspace(0, len(features_i) - 1, min(len(features_i), max_features_per_time), dtype=int):
    #         z_t = features_i[t]
    #         images.append(img_i)
    #         titles.append(f"Input {img_idx + 1}")
    #         for i in range(z_t.shape[0]):
    #             images.append(z_t[i])
    #             titles.append(f"Feature {i + 1} (t={t})")
    #
    # images2 = torch.stack([i.squeeze() for i in images])
    # images2 = (images2 - images2.min()) / (images2.max() - images2.min())
    # images2 = [images2[i] for i in range(images2.shape[0])]
    # plot_images(images=images2, titles=titles, max_cols=features[0].shape[1] + 1, plot_colorbar=True)

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        return LateralLayerEfficientNetwork(self.conf, self.fabric)

    def on_epoch_end(self):
        print_logs(self.get_logs())
