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
                 fabric,
                 n_channels: int,
                 locality_size: Optional[int] = 5,
                 lr: Optional[float] = 0.01,
                 ):
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.
        Source: https://github.com/ThomasMiconi/HebbianCNNPyTorch/tree/main

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
        self.neib_size = 2 * self.locality_size + 1
        self.lr = lr
        self.k = 1
        self.thr_rate = 10.
        self.target_rate = self.k / self.n_channels
        self.train_ = True
        self.step = 0

        self.W = torch.randn((self.n_channels, self.n_channels, self.neib_size, self.neib_size), requires_grad=True,
                             device=fabric.device)
        self.W.data = self.W.data / (1e-10 + torch.sqrt(torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))
        self.b = torch.zeros((1, self.n_channels, 1, 1), requires_grad=False).to(fabric.device)

        self.optimizer = torch.optim.Adam([self.W,], lr=self.lr)

    def forward(self, x: Tensor) -> Tensor:
        self.optimizer.zero_grad()

        prelimy = F.conv2d(x, self.W, padding="same")

        # Then we compute the "real" output (y) of each cell, with winner-take-all competition
        with torch.no_grad():
            realy = (prelimy - self.b)
            tk = torch.topk(realy.data, self.k, dim=1, largest=True)[0]
            realy.data[realy.data < tk.data[:, -1, :, :][:, None, :, :]] = 0
            realy.data = (realy.data > 0).float()

        # Then we compute the surrogate output yforgrad, whose gradient computations produce the desired Hebbian output
        # Note: We must not include thresholds here, as this would not produce the expected gradient expressions. The actual values will come from realy, which does include thresholding.
        yforgrad = prelimy - 1 / 2 * torch.sum(self.W * self.W, dim=(1, 2, 3))[None, :, None, None]
        yforgrad.data = realy.data  # We force the value of yforgrad to be the "correct" y

        loss = torch.sum(-1 / 2 * yforgrad * yforgrad)
        loss.backward()
        if self.step > 100 and self.train_:  # No weight modifications before batch 100 (burn-in) or after learning epochs (during data accumulation for training / testing)
            self.optimizer.step()
            self.W.data = self.W.data / (1e-10 + torch.sqrt(torch.sum(self.W.data ** 2, dim=[1, 2, 3], keepdim=True)))  # Weight normalization

        with torch.no_grad():
            # Threshold adaptation is based on realy, i.e. the one used for plasticity. Always binarized (firing vs. not firing).
            self.b += self.thr_rate * (
                        torch.mean((realy.data > 0).float(), dim=(0, 2, 3))[None, :, None, None] - self.target_rate)

        self.step += 1
        return realy.detach()




class LateralLayerToy(nn.Module):

    def __init__(self,
                 fabric,
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input image.
        :return: Output of the model (updated features).
        """
        return self.model(x)

    def forward_steps_through_time(self, x: Tensor) -> Tuple[List[Tensor], List[float], List[int]]:
        """
        Forward pass through the model over multiple timesteps. The number of timesteps depends on the change
        threshold and the maximum number of timesteps (i.e. the forward pass through time is cancel if the activations'
        change is below a given threshold).
        :param x: Features extracted from the image.
        :return: List of features, list of changes, list ov view indexes.
        """
        # if x.unique().shape[0] > 2 or x.unique(sorted=True) != torch.tensor([0, 1]):
            # x = torch.bernoulli(x.clip(0, 1))
            # x = torch.where(x > 0.5, 1., 0.)

        with torch.no_grad():
            dd = (2, 3, 4) if len(x.shape) == 5 else (1, 2, 3)
            x = x - torch.mean(x, dim=dd, keepdim=True)
            x = x / (1e-10 + torch.std(x, dim=dd, keepdim=True))


        changes, features, view_idxes = [], [], []
        for view_idx in range(x.shape[1]):
            t = 0
            x_view = x[:, view_idx, ...]
            features.append(x_view)
            view_idxes.append(view_idx)
            for t in range(self.conf["lateral_model"]["max_timesteps"]):
                x_old = x_view
                x_view = self.forward(x_view)
                change = F.l1_loss(x_old, x_view).item()
                changes.append(change)
                features.append(x_view)
                view_idxes.append(view_idx)
                if change < self.conf["lateral_model"]["change_threshold"]:
                    break

            self.avg_meter_t(t)

        return features, changes, view_idxes

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
        stats = {
            "weight_mean": torch.mean(self.model.W).item(),
            "weight_std": torch.std(self.model.W).item(),
            "weight_min": torch.min(self.model.W).item(),
            "weight_max": torch.max(self.model.W).item(),
            "weight_above_0.9": torch.sum(self.model.W >= 0.9).item() / self.model.W.numel(),
            "weight_below_-0.9": torch.sum(self.model.W <= -0.9).item() / self.model.W.numel(),
        }
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
            weight_img_list = [weight[i, j].unsqueeze(0) for i in range(weight.shape[0]) for j in range(weight.shape[1])]
            # Order is [(0, 0), (0, 1), ..., (3, 2), (3, 3)]
            # Each row shows the input weights per output channel
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True)
            ax.imshow(grid.permute(1, 2, 0))
            ax.set_title(title)

        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        _hist_plot(axs[0], self.model.W.detach().cpu(), "Weight distribution (without tanh)")
        _hist_plot(axs[1], F.tanh(self.model.W).detach().cpu(), "Weight distribution (with tanh)")
        _plot_weights(axs[2], self.model.W.detach().cpu(), "Weight matrix")

        plt.tight_layout()
        plt.show()

    def plot_features_single_sample(self, img: Tensor, features: Tensor):
        """
        Plot the features extracted from a single sample.
        :param img: The original image
        :param features: The features extracted from the image.
        """

        features, changes, view_idxes = self.forward_steps_through_time(features)

        n_images = min(view_idxes[-1] + 1, 3)
        max_features_per_time = 10

        images, titles = [], []

        for img_idx in range(n_images):
            img_i = img[0, img_idx, ...]  # only use batch idx 0
            features_i = [f[0] for f, idx in zip(features, view_idxes) if idx == img_idx]
            images.append(img_i)
            titles.append(f"Input {img_idx + 1}")

            for i in range(features_i[0].shape[0]):
                images.append(features_i[0][i])
                titles.append(f"Feature {i + 1} (t=0)")

            for t in np.linspace(0, len(features_i) - 1, min(len(features_i), max_features_per_time), dtype=int):
                z_t = features_i[t]
                images.append(img_i)
                titles.append(f"Input {img_idx + 1}")
                for i in range(z_t.shape[0]):
                    images.append(z_t[i])
                    titles.append(f"Feature {i + 1} (t={t})")

        plot_images(images=images, titles=titles, max_cols=features[0].shape[1] + 1, vmin=0., vmax=1.)

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        lm_conf = self.conf["lateral_model"]
        # return LateralLayerToy(
        #     self.fabric,
        #     n_channels=self.conf["feature_extractor"]["out_channels"],
        #     locality_size=lm_conf['locality_size'],
        #     alpha=lm_conf['alpha'],
        #     lr=lm_conf['lr']
        # )

        return LateralLayerEfficient(
                self.fabric,
                n_channels=self.conf["feature_extractor"]["out_channels"],
                locality_size=lm_conf['locality_size'],
                lr=lm_conf['lr']
            )

    def on_epoch_end(self):
        print_logs(self.get_logs())
