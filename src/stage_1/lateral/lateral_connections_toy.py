from datetime import datetime
from pathlib import Path
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
from tools import AverageMeter, bin2dec
from utils import create_video_from_images_ffmpeg, print_logs


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
        self.k = 1  # out_channels  # TODO: Currently no winner-take-all, so k is not used
        self.thr_rate = 0.05  # 10.
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

    def update_k(self, k):
        self.k = k
        self.target_rate = self.k / self.out_channels
        assert False, "Should this be called?"

    def forward(self, x: Tensor) -> Tensor:
        self.optimizer.zero_grad()

        prelimy = F.conv2d(x, self.W, padding="same")

        # Then we compute the "real" output (y) of each cell, with winner-take-all competition
        with torch.no_grad():
            realy = (prelimy - self.b)

            # if self.ts in self.prev_activations:
            #     realy = .4 * realy + (1 - .4) * self.prev_activations[self.ts]
            if self.ts-1 in self.prev_activations:
                realy = .5 * realy + (1-.5) * self.prev_activations[self.ts-1]
            self.prev_activations[self.ts] = realy

            # TODO: In order that the channels have more distinct features, we could limit the number of activations per channel to 1.5x the input activations of the same channel?
            # TODO: We could use different Filters, e.g. some layers can access only some input filters (number of combinations: 2^in_channels)

            # k winner take all
            tk = torch.topk(realy.data, self.k, dim=1, largest=True)[0]
            realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0

            # binary output
            realy.data = (realy.data > 0).float()

        # Then we compute the surrogate output yforgrad, whose gradient computations produce the desired Hebbian output
        # Note: We must not include thresholds here, as this would not produce the expected gradient expressions. The
        # actual values will come from realy, which does include thresholding.
        yforgrad = prelimy - 1 / 2 * torch.sum(self.W * self.W, dim=(1, 2, 3))[None, :, None, None]  # Instar rule, dw ~= y(x-w)
        # yforgrad = prelimy - 1/2 * torch.sum(self.W * self.W, dim=(1,2,3))[None,:, None, None] * realy.data # Oja's rule, dw ~= y(x-yw)
        # yforgrad = prelimy
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
        self.concat_input = True
        lm_conf = self.conf["lateral_model"]
        in_channels = self.conf["feature_extractor"]["out_channels"]

        self.l1 = LateralLayerEfficient(
            self.fabric,
            in_channels= in_channels + lm_conf["channels"] if self.concat_input else in_channels,
            out_channels=lm_conf["channels"],
            locality_size=lm_conf['locality_size'],
            lr=lm_conf['lr']
        )

    def new_sample(self):
        self.l1.new_sample()

    def get_layer_weights(self):
        return {"L1": self.l1.W}

    def set_train(self, train: bool):
        self.l1.train_ = train

    def forward(self, x: Tensor) -> Tuple[List[int], Tensor, int]:
        """
        Forward pass over multiple timesteps
        :param x_in: The input tensor (extracted features of the image)
        :return: List of changes, Features extracted by the lateral layers, number of timesteps
        """
        t = 0

        changes, features = [], []
        x_in = torch.cat([x, torch.zeros_like(x)], dim=1) if self.concat_input else x

        for t in range(self.conf["lateral_model"]["max_timesteps"]):
            self.l1.update_ts(t)
            x_old = x_in
            z = self.l1(x_in)
            x_in = torch.cat([x, z], dim=1) if self.concat_input else z
            change = F.l1_loss(x_old, x_in).item()
            changes.append(change)
            features.append(z)
            # if change < self.conf["lateral_model"]["change_threshold"]:
            #     break
        return changes, torch.stack(features, dim=1), t


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

    def forward(self, x: Tensor) -> Tuple[List[int], Tensor, int]:
        """
        Forward pass through the model.
        :param x: Input image.
        :return:  List of changes, Features extracted by the lateral layers, number of timesteps
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

        return torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), changes

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

        for layer, weight in self.model.get_layer_weights().items():
            fig, axs = plt.subplots(1, 2, figsize=(8, 5))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(axs[1], weight[:20, :20, ...].detach().cpu(), f"Example Weight matrix ({layer})")
            plt.tight_layout()
            plt.show()

    def plot_samples(self,
                     img: List[Tensor],
                     features: List[Tensor],
                     input_features: List[Tensor],
                     lateral_features: List[Tensor]):
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
        """

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
            plt_images = self._normalize_image_list(plt_images)
            plot_images(images=plt_images, titles=plt_titles, max_cols=2 * features.shape[1] + 1, plot_colorbar=True,
                        vmin=0, vmax=1)

        def _plot_lateral_activation_map(lateral_features):
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
                        vmin=0, vmax=1)

        def _plot_lateral_output(img, lateral_features):
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
                        vmin=0, vmax=1, mask_vmin=0, mask_vmax=lateral_features.shape[2] + 1)

        for img_i, features_i, input_features_i, lateral_features_i in zip(img, features, input_features,
                                                                           lateral_features):
            for batch_idx in range(img_i.shape[0]):
                _plot_input_features(img_i[batch_idx], features_i[batch_idx], input_features_i[batch_idx])
                _plot_lateral_activation_map(lateral_features_i[batch_idx])
                _plot_lateral_output(img_i[batch_idx], lateral_features_i[batch_idx])

    def create_activations_video(self,
                                 images: List[Tensor],
                                 features: List[List[Tensor]],
                                 activations: List[List[List[Tensor]]]):
        """
        Create a video of the activations.
        :param images: The original images.
        :param features: The features extracted from the images.
        :param activations: The activations after the lateral connections.
        """
        folder = Path("../tmp") / "video_toys"
        if not folder.exists():
            folder.mkdir(parents=True)

        c = 0

        for img_idx in range(len(images)):
            img = images[img_idx]
            features_img = features[img_idx]
            activations_img = activations[img_idx]

            for batch_idx in range(img.shape[0]):

                for view_idx in range(len(features_img)):
                    img_view = img[batch_idx, view_idx]
                    features_view = features_img[batch_idx, view_idx]
                    activations_view = activations_img[batch_idx, view_idx]

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

                create_video_from_images_ffmpeg(folder,
                                                f"{folder / datetime.now().strftime('%Y-%d-%m_%H-%M-%S')}_{img_idx}_"
                                                f"{batch_idx}.mp4")

                for f in folder.glob("*.png"):
                    f.unlink()

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        return LateralLayerEfficientNetwork1L(self.conf, self.fabric)

    def on_epoch_end(self):
        print_logs(self.get_logs())

    def _normalize_image_list(self, img_list):
        img_list = torch.stack([i.squeeze() for i in img_list])
        img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min() + 1e-9)
        img_list = [img_list[i] for i in range(img_list.shape[0])]
        return img_list
