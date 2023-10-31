"""
PyTorch Lightning modules for training and validating an Autoencoder as feature extractor of stage 0.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import plot_images, undo_norm_from_conf
from data.utils.patches2d import Patches2D
from models.autoencoder.simple_ae import Autoencoder
from models.autoencoder.vq_vae import SmallVQVAE
from data import show_grid
from models import BaseLitModule, TinyVQVAE
from tools import torch_optim_from_conf
from tools.custom_math import bin2dec


class AEFeatureExtractor(BaseLitModule):
    """
    Extract features from images using an autoencoder.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__(conf, fabric, logging_prefixes=["train", "val"])
        self.model = self.configure_model()
        self.data_var = torch.mean(torch.Tensor(self.conf['dataset']['std'])).to(fabric.device) ** 2

    def preprocess_data_(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Preprocess the data batch.
        :param batch: Data batch, containing input data and labels.
        :return: Preprocessed data batch.
        """
        x, y = batch
        return x, y

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the model.
        :param x:
        :return: Loss, reconstructed image, perplexity, and encodings.
        """
        return self.model(x)

    def step(self, batch: Tensor, batch_idx: int, mode_prefix: str) -> Tensor:
        """
        Forward step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :param mode_prefix: Prefix for the mode (train, val, test).
        :return: Loss of the training step.
        """
        x, y = self.preprocess_data_(batch)
        x_recon, encodings = self.forward(x)
        loss = F.mse_loss(x_recon, x) / self.data_var
        self.log_step(
            processed_values={"loss": loss},
            metric_pairs=[(x_recon, x)],
            prefix=mode_prefix
        )

        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Forward training step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the training step.
        """
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Forward validation step: Forward pass, and logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the validation step.
        """
        return self.step(batch, batch_idx, "val")

    def visualize_samples(self, batch: Tensor, batch_idx: int):
        """
        Visualize a batch of samples.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        """
        x, y = self.preprocess_data_(batch)
        if x.shape[0] > 32:
            x = x[:32, ...]

        x_recon, encodings = self.forward(x)
        x_recon = undo_norm_from_conf(x_recon, self.conf)
        x_recon = torch.clip(x_recon, 0, 1)
        grid_orig = torchvision.utils.make_grid(undo_norm_from_conf(x, self.conf), nrow=8, normalize=True)
        grid_recon = torchvision.utils.make_grid(x_recon, nrow=8, normalize=True)
        show_grid([grid_orig, grid_recon])

    def visualize_filters(self, batch: Tensor, batch_idx: int):
        """
        Visualize the encodings of a batch of samples.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        """
        x, y = self.preprocess_data_(batch)
        if x.shape[0] > 20:
            x = x[:20, ...]

        features = self.model.encoder.model[0](x)
        features = self.model.encoder.model[1](features)
        features_max = features.max(dim=1)[0].unsqueeze(1)
        features_min = features.min(dim=1)[0].unsqueeze(1)
        features_norm = (features - features_min) / (features_max - features_min + 1e-9)

        features_max_glob = features.max()
        features_min_glob = features.min()
        features_norm_glob = (features - features_min_glob) / (features_max_glob - features_min_glob + 1e-9)

        nrows = features.shape[1]
        features_norm = features_norm.reshape(-1, features_norm.shape[2], features_norm.shape[3]).unsqueeze(1)
        features_norm_glob = features_norm_glob.reshape(-1, features_norm_glob.shape[2], features_norm_glob.shape[3]).unsqueeze(1)
        grid_features = torchvision.utils.make_grid(features_norm, nrow=nrows, normalize=True)
        grid_features_bin = torchvision.utils.make_grid((features_norm > 0.95).float(), nrow=nrows, normalize=True)
        grid_features_glob = torchvision.utils.make_grid(features_norm_glob, nrow=nrows, normalize=True)
        show_grid([grid_features, grid_features_glob, grid_features_bin])

    def configure_model(self):
        """
        Configure the model, i.e. create an VQ-VAE instance.
        :return:
        """
        model_conf = self.conf["model"]
        if model_conf["type"] == "AE":
            params = model_conf["params"]
            return Autoencoder(
                **params
            )
        else:
            raise NotImplementedError(f"Model {model_conf['type']} not implemented")

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[ReduceLROnPlateau]]:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer and scheduler.
        """
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)
