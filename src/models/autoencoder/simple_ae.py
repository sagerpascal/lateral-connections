"""

Simple Autoencoder models

"""
from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Encoder(nn.Module):
    """
    Encoder of the Autoencoder
    """

    def __init__(self, channels: List[int], kernel_size: int):
        """
        :param channels: The number of channels in each layer.
        :param kernel_size: The kernel size of the convolutional layers.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.model = self.setup_model()

    def setup_model(self) -> nn.Sequential:
        """
        Setup the modules of the Encoder

        :return: The modules of the Encoder.
        """
        modules = []
        for i in range(len(self.channels) - 1):
            s = 1 if i == 0 else 2
            modules.append(
                nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=self.kernel_size, stride=s, padding=2))
            modules.append(nn.ReLU(True))

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.model(x)
        return x


class Decoder(nn.Module):
    """
    Decoder of the Autoencoder
    """

    def __init__(self, channels: List[int], kernel_size: int):
        """
        :param channels: The number of channels in each layer.
        :param kernel_size: The kernel size of the convolutional layers.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.model = self.setup_model()

    def setup_model(self) -> nn.Sequential:
        """
        Setup the modules of the Encoder

        :return: The modules of the Encoder.
        """
        modules = []
        for i in range(len(self.channels) - 2, -1, -1):
            s = 2 if i > 0 else 1
            op = 1 if i > 0 else 0
            modules.append(
                nn.ConvTranspose2d(self.channels[i + 1], self.channels[i], kernel_size=self.kernel_size, stride=s,
                                   padding=2, output_padding=op))
            if i >= 1:
                modules.append(nn.ReLU(True))

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.model(x)
        return x


class Autoencoder(nn.Module):
    """
    Simple Autoencoder
    """

    def __init__(self,
                 channels: List[int],
                 kernel_size: int,
                 ):
        """
        :param channels: The number of channels in each layer.
        :param kernel_size: The kernel size of the convolutional layers.
        """
        super().__init__()
        self.encoder = Encoder(channels, kernel_size)
        self.decoder = Decoder(channels, kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Autoencoder
        :param x: The input tensor.
        :return: The reconstructed input tensor, and the encodings.
        """
        x = F.pad(x, (2, 2, 2, 2), mode="constant", value=-0.1307 / 0.3081)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = x_recon[:, :, 2:-2, 2:-2]
        return x_recon, z
