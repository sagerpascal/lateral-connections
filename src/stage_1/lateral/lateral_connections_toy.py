from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
import lightning.pytorch as pl
import numpy as np


class LateralLayerToy(nn.Module):

    def __init__(self, fabric, n_channels: int, locality_size: int = 2):
        super().__init__()
        """
        Lateral Layer trained with Hebbian Learning. The input and output of this layer are binary.
        :param fabric: Fabric instance.
        :param n_channels: Number of channels in the input and output.
        :param locality_size: Size of the locality, i.e. how many neurons are connected to each other. For 
        example, if locality_size = 2, then each neuron is connected to 5 neurons on each side of it.
        """
        self.n_channels = n_channels
        self.locality_size = locality_size
        # weight has the shape (output_channels, input_channels, kernel_height, kernel_width)
        self.W = torch.rand((self.n_channels, self.n_channels, 2 * self.locality_size + 1, 2 * self.locality_size + 1),
                            dtype=torch.float32, requires_grad=False).to(fabric.device)
        self.alpha = 0.9
        self.lr = 0.01
        self.train_ = True
        self.step = 0

    def update_weights(self, input: Tensor, output: Tensor):
        """
        Updates the weights of the layer (i.e. a training step).
        :param input: Layer input.
        :param output: Layer output.
        """
        self.step += 1
        patch_size = self.locality_size * 2 + 1
        # input_patches = F.pad(input, (self.locality_size, self.locality_size, self.locality_size,
        # self.locality_size)) \
        #     .unfold(2, patch_size, 1).unfold(3, patch_size, 1).permute(0, 1, 4, 5, 2, 3) \
        #     .reshape(input.shape[0], 1, input.shape[1], patch_size, patch_size, -1)
        # output_flatten = output.reshape(output.shape[0], output.shape[1], 1, 1, 1, -1)
        # pos_co_activation = torch.matmul(input_patches, output_flatten.repeat(1, 1, 1, patch_size, patch_size, 1))
        # neg_co_activation = torch.matmul(input_patches, output_flatten - 1) + torch.matmul(input_patches - 1,
        #                                                                                    output_flatten)

        input_patches = F.pad(input, (self.locality_size, self.locality_size, self.locality_size, self.locality_size)) \
            .unfold(2, patch_size, 1).unfold(3, patch_size, 1).unsqueeze(2)
        output = output.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, patch_size, patch_size).unsqueeze(1)
        pos_co_activation = torch.matmul(input_patches, output)
        neg_co_activation = torch.matmul(input_patches, output - 1) + torch.matmul(input_patches - 1, output)

        update = torch.mean(pos_co_activation + neg_co_activation, dim=(0, 3, 4))
        self.W += self.lr * update

        if self.step % 1000 == 0:
            print("Weight values", self.W.unique())
            print("High weight values", torch.sum(self.W >= 0.9))
            print("Low weight values", torch.sum(self.W <= -0.9))
            print("Mean weight value", torch.mean(self.W))
            print("Weight shape", self.W.shape)
            bins = 20
            min, max = torch.min(self.W).item(), torch.max(self.W).item()
            hist = torch.histc(self.W.detach().cpu(), bins=bins, min=min, max=max)

            x = np.linspace(min, max, bins)
            plt.bar(x, hist, align='center')
            plt.xlabel(f'Bins form {min:.4f} to {max:.4f}')
            plt.title("Weight distribution (without tanh)")
            plt.show()

            min, max = torch.min(F.tanh(self.W)).item(), torch.max(F.tanh(self.W)).item()
            hist = torch.histc(F.tanh(self.W).detach().cpu(), bins=bins, min=min, max=max)

            x = np.linspace(min, max, bins)
            plt.bar(x, hist, align='center')
            plt.xlabel(f'Bins form {min:.4f} to {max:.4f}')
            plt.title("Weight distribution (with tanh)")
            plt.show()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        :param input: Layer input.
        :return: Layer output.
        """
        weight = F.tanh(self.W)
        a = F.conv2d(input, weight, groups=1, padding="same")  # TODO: check group parameter
        z = torch.sum(weight, dim=(1, 2, 3))
        p = a / (self.alpha * z.view(1, self.n_channels, 1, 1))
        output = torch.bernoulli(p.clip(0., 1.))

        if self.train_:
            self.update_weights(input, output)

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input image.
        :return: Output of the model (updated features).
        """
        return self.model(x)

    def configure_model(self) -> nn.Module:
        """
        Create the model.
        :return: Model with lateral connections.
        """
        return LateralLayerToy(self.fabric, n_channels=4, locality_size=5)
