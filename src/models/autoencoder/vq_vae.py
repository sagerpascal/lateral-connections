"""

Implementation of VQ-VAE models.

Sources:
    https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VectorQuantizer(nn.Module):
    """
    This layer takes a tensor to be quantized. The channel dimension will be used as the space in which to quantize.
    All other dimensions will be flattened and will be seen as different examples to quantize.

    The output tensor will have the same shape as the input.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        """
        Constructor.
        :param num_embeddings: Number of embedding vectors (codebook size).
        :param embedding_dim: Size of each embedding vector (length of each embedding vector).
        :param commitment_cost: Factor for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Quantize the input tensor.
        :param inputs: The tensor to quantize.
        :return: Quantization loss, quantized tensor, perplexity and encodings.
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """
    A slightly modified version of VectorQuantizer which will use exponential moving averages to update the embedding
    vectors instead of an auxilliary loss. This has the advantage that the embedding updates are independent of the
    choice of optimizer for the encoder, decoder and other parts of the architecture. For most experiments the EMA
    version trains faster than the non-EMA version.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 decay: float,
                 epsilon: Optional[float] = 1e-5):
        """
        Constructor.
        :param num_embeddings: Number of embedding vectors (codebook size).
        :param embedding_dim: Size of each embedding vector (length of each embedding vector).
        :param commitment_cost: Factor for the commitment loss.
        :param decay: Decay for the EMA.
        :param epsilon: Small value to avoid numerical instability.
        """
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Quantize the input tensor.
        :param inputs: The tensor to quantize.
        :return: Quantization loss, quantized tensor, perplexity and encodings.
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        """
        Constructor.
        :param in_channels: Block input channels.
        :param num_hiddens: Number of output (hidden) channels.
        :param num_residual_hiddens: Number of residual channels (channels inside residual block).
        """
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor.
        :return: Residual block output.
        """
        return x + self._block(x)


class ResidualStack(nn.Module):
    """
    Stack of residual blocks.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        """
        Constructor.
        :param in_channels: Number of input channels.
        :param num_hiddens: Number of output (hidden) channels.
        :param num_residual_layers:
        :param num_residual_hiddens: Number of residual channels (channels inside residual block).
        """
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input tensor.
        :return: Ouput from residual stack.
        """
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    """
    Encoder with conv layers and residual stack.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        """
        Constructor.
        :param in_channels: Number of input channels.
        :param num_hiddens: Size of the hidden channels.
        :param num_residual_layers: Number of residual blocks.
        :param num_residual_hiddens: Number of residual channels (channels inside residual block).
        """
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        :param inputs: Input tensor.
        :return: Output tensor.
        """
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_hiddens: int,
            num_residual_layers: int,
            num_residual_hiddens: int,
            out_channels: int
    ):
        """
        Constructor.
        :param in_channels: Number of input channels.
        :param num_hiddens: Size of the hidden channels.
        :param num_residual_layers: Number of residual blocks.
        :param num_residual_hiddens: Number of residual channels (channels inside residual block).
        :param out_channels: Number of output channels.
        """
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        :param inputs: Input tensor.
        :return: Output tensor.
        """
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Auto-Encoder with residual blocks.
    """

    def __init__(self,
                 in_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 ):
        """
        Constructor.
        :param in_channels: Number of input channels (number of channels of the image).
        :param num_hiddens: Number of hidden channels.
        :param num_residual_layers: Number of residual layers.
        :param num_residual_hiddens: Number of residual hidden channels.
        :param num_embeddings: Number of embeddings (number of latent vectors).
        :param embedding_dim: Size of the embedding vectors.
        :param commitment_cost: Commitment cost factor.
        :param decay: Decay factor for the exponential moving average update of the embedding vectors.
        """
        super().__init__()

        self._encoder = Encoder(in_channels,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings,
                                              embedding_dim,
                                              commitment_cost,
                                              decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings,
                                           embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        :param x: Input tensor (image).
        :return: Loss, reconstructed image, perplexity.
        """
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


class TinyVQVAE(nn.Module):
    """
    Tiny Vector Quantized Variational Auto-Encoder with only convolutional layers.
    """

    def __init__(self,
                 in_channels: int,
                 num_hiddens: int,
                 num_conv_layers: int,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 ):
        """
        Constructor.
        :param in_channels: Number of input channels (number of channels of the image).
        :param num_hiddens: Number of hidden channels.
        :param num_conv_layers: Number of convolutional layers.
        :param num_embeddings: Number of embeddings (number of latent vectors).
        :param embedding_dim: Size of the embedding vectors.
        :param commitment_cost: Commitment cost factor.
        :param decay: Decay factor for the exponential moving average update of the embedding vectors.
        """
        super().__init__()
        assert num_conv_layers > 0, "Number of convolutional layers must be greater than 0."

        self._encoder = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3,
                                                 stride=1, padding=1)] + [
                                          nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3,
                                                    stride=1, padding=1) for _ in range(num_conv_layers - 1)])

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings,
                                              embedding_dim,
                                              commitment_cost,
                                              decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings,
                                           embedding_dim,
                                           commitment_cost)

        self._decoder = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens,
                                                          kernel_size=3, stride=1, padding=1)
                                       for _ in range(num_conv_layers - 1)] + [
                                          nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=in_channels,
                                                             kernel_size=3, stride=1, padding=1)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        :param x: Input tensor (image).
        :return: Loss, reconstructed image, perplexity.
        """
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity