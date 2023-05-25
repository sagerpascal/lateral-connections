from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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

HEBBIAN_ALGO = Literal['instar', 'oja', 'vanilla']
W_INIT = Literal['random', 'zeros']


# TODO: In jedem timestep muss ursprüngliche Features berücksichtigt werden!
# TODO: änhliche Linien über mehrere Timesteps
# TODO: In jedem timestep geht die Sparsity herunter bis neues Bild ingelesen wird


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
                 w_init: Optional[W_INIT] = 'random',
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

        self.train_ = True
        self.step = 0
        self.ts = None
        self.prev_activations = {}
        self.prev_activation = None

        assert self.k <= self.out_channels, "k must be smaller than out_channels"
        assert self.k > 0, "k must be greater than 0"

        print("Mask out background:", self.mask_out_bg, "Moving average:", self.moving_average)
        if self.mask_out_bg:
            self.target_rate = self.target_rate / 10

        weight_shape = (self.out_channels, self.in_channels, self.neib_size, self.neib_size)
        if w_init == "random":
            self.W = torch.randn(weight_shape, requires_grad=True, device=fabric.device)
        elif w_init == "zeros":
            self.W = torch.zeros(weight_shape, requires_grad=True, device=fabric.device)

        self.W.data = self.W.data / (1e-10 + torch.sqrt(torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))
        self.b = torch.zeros((1, self.out_channels, 1, 1), requires_grad=False).to(fabric.device)

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()

        prelimy = F.conv2d(x, self.W, padding="same")

        # Then we compute the "real" output (y) of each cell, with winner-take-all competition
        with torch.no_grad():
            realy = (prelimy - self.b)

            if self.moving_average:
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

            # k winner take all
            smallest_value_per_channel = torch.amin(realy, dim=(2, 3)) + 0.0001
            tk = torch.topk(realy.data, self.k, dim=1, largest=True)[0]
            realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
            if self.mask_out_bg:
                realy.data[realy.data <= smallest_value_per_channel.view(
                    smallest_value_per_channel.shape + (1, 1))] = 0  # mask out background...

            # binary output
            threshold = (torch.sort(realy.view(realy.shape[0], -1), dim=1, descending=True)[0][:,
                         int(realy.numel() / realy.shape[0] / realy.shape[1] * 0.2)])
            realy.data = (realy.data > threshold.view(realy.shape[0], 1, 1, 1)).float()

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
        if self.train_:
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
        self.avg_meter_t = AverageMeter()
        self.concat_input = True
        lm_conf = self.conf["lateral_model"]
        in_channels = self.conf["feature_extractor"]["out_channels"]

        self.l1 = LateralLayerEfficient(
            self.fabric,
            in_channels=in_channels + lm_conf["out_channels"] if self.concat_input else in_channels,
            **lm_conf["l1_params"],
        )

    def new_sample(self):
        self.l1.new_sample()

    def get_layer_weights(self):
        return {"L1": self.l1.W}

    def set_train(self, train: bool):
        self.l1.train_ = train

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass over multiple timesteps
        :param x: The input tensor (extracted features of the image)
        :return: Extracted features thar were fed into the lateral layer (binarized version of features from feature
        extractor), Features extracted by the lateral layers (binarized), Features extracted by the lateral layers (as
        float)
        """

        # TODO: Probabilistic neuron testen, zuerst mit WTA
        # TODO: Softere Art von WTA - wie kann Symmetrie gebrochen werden?

        self.l1.new_sample()
        z = None

        input_features, lateral_features, lateral_features_f = [], [], []
        for view_idx in range(x.shape[1]):
            # prepare input view
            x_view = x[:, view_idx, ...]
            x_view = torch.where(x_view > 0., 1., 0.)
            input_features.append(x_view)

            if z is None and self.concat_input:
                z = torch.zeros_like(x_view)

            t = 0
            features, features_float = [], []
            for t in range(self.conf["lateral_model"]["max_timesteps"]):
                self.l1.update_ts(t)
                # x_old = x_in
                x_in = torch.cat([x_view, z], dim=1) if self.concat_input else z
                z_float, z = self.l1(x_in)
                features.append(z)
                features_float.append(z_float)
                # if F.l1_loss(x_old, x_in).item() < self.conf["lateral_model"]["change_threshold"]:
                #     break

            lateral_features.append(torch.stack(features, dim=1))
            lateral_features_f.append(torch.stack(features_float, dim=1))
            self.avg_meter_t(t)

        return torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), torch.stack(lateral_features_f,
                                                                                                     dim=1)

    def get_model_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics of the model weights.
        :return: Dictionary with statistics.
        """
        stats = {}
        for layer, weight in self.get_layer_weights().items():
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

    def get_logs(self) -> Dict[str, float]:
        return self.model.get_logs()

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

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None:
                plt.savefig(Path(fig_fp) / f'weights_{layer}.png')

    def plot_samples(self,
                     img: List[Tensor],
                     features: List[Tensor],
                     input_features: List[Tensor],
                     lateral_features: List[Tensor],
                     lateral_features_f: List[Tensor]):
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
        """

        def _plot_input_features(img, features, input_features, fig_fp: Optional[str] = None):
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
                        vmin=0, vmax=1, fig_fp=fig_fp)

        def _plot_lateral_activation_map(lateral_features, fig_fp: Optional[str] = None):
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
                        vmin=0, vmax=1, fig_fp=fig_fp)

        def _plot_lateral_heat_map(lateral_features_f, fig_fp: Optional[str] = None):
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
                        fig_fp=fig_fp, cmap='hot', interpolation='nearest', vmin=v_min, vmax=v_max)

        def _plot_lateral_output(img, lateral_features, fig_fp: Optional[str] = None):
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
                        vmin=0, vmax=1, mask_vmin=0, mask_vmax=lateral_features.shape[2] + 1, fig_fp=fig_fp)

        fig_fp = self.conf['run']['plots'].get('store_path', None)
        for i, (img_i, features_i, input_features_i, lateral_features_i, lateral_features_f_i) in enumerate(
                zip(img, features, input_features, lateral_features, lateral_features_f)):
            for batch_idx in range(img_i.shape[0]):
                if fig_fp is not None:
                    fig_fp = Path(fig_fp)
                base_name = f"sample_{i}_batch_idx_{batch_idx}"
                _plot_input_features(img_i[batch_idx], features_i[batch_idx], input_features_i[batch_idx],
                                     fig_fp=fig_fp / f'{base_name}_input_features.png')
                _plot_lateral_activation_map(lateral_features_i[batch_idx],
                                             fig_fp=fig_fp / f'{base_name}_lateral_act_maps.png')
                _plot_lateral_heat_map(lateral_features_f_i[batch_idx],
                                       fig_fp=fig_fp / f'{base_name}_lateral_heat_maps.png')
                _plot_lateral_output(img_i[batch_idx], lateral_features_i[batch_idx],
                                     fig_fp=fig_fp / f'{base_name}_lateral_output.png')

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
                fig_fp = self.conf['run']['plots'].get('store_path', None)
                fig_fp = Path(fig_fp) if fig_fp is not None else folder
                create_video_from_images_ffmpeg(folder,
                                                f"{fig_fp / datetime.now().strftime('%Y-%d-%m_%H-%M-%S')}_{img_idx}_"
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
