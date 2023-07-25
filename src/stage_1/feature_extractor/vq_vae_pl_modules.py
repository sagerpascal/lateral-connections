"""
PyTorch Lightning modules for training and validating a VQ-VAE as feature extractor of stage 1.
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
from models.autoencoder.vq_vae import SmallVQVAE
from data import show_grid
from models import BaseLitModule, TinyVQVAE
from tools import torch_optim_from_conf
from tools.custom_math import bin2dec


class VQVAEFeatureExtractorPatchMode(BaseLitModule):
    """
    Extract features from non-overlapping patches of an image using a VQ-VAE.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__(conf, fabric, logging_prefixes=["train", "val"])
        self.model = self.configure_model()
        self.p2d = self.configure_patch_handler()
        self.data_var = torch.mean(torch.Tensor(self.conf['dataset']['std'])).to(fabric.device) ** 2

    def prepare_data_(self, x: Tensor) -> Tensor:
        """
        Prepare data for training and validation by splitting the images into patches.
        :param x: Input data.
        :return: Data for training and validation.
        """
        return self.p2d.image_to_patches(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the model.
        :param x: Input image.
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
        x, y = batch
        x = self.prepare_data_(x)
        loss = []
        for i in range(x.shape[1]):
            q_loss, x_recon, perplexity, encodings = self.forward(x[:, i, ...])
            recon_loss = F.mse_loss(x_recon, x[:, i, ...]) / self.data_var
            self.log_step(
                processed_values={"q_loss": q_loss, "recon_loss": recon_loss, "perplexity": perplexity},
                metric_pairs=[(x_recon, x[:, i, ...])],
                prefix=mode_prefix
            )
            loss.append(q_loss + recon_loss)
        return torch.stack(loss).mean()

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
        x, y = batch
        if x.shape[0] > 32:
            x = x[:32, ...]
        x_patches = self.prepare_data_(x)
        x_recons = []
        for i in range(x_patches.shape[1]):
            loss_, x_recon, perplexity, encodings = self.forward(x_patches[:, i, ...])
            x_recons.append(x_recon)
        x_recons = undo_norm_from_conf(self.p2d.patches_to_image(torch.stack(x_recons, dim=1)), self.conf)
        x_recons = torch.clip(x_recons, 0, 1)
        grid_orig = torchvision.utils.make_grid(undo_norm_from_conf(self.p2d.pad_image(x), self.conf),
                                                nrow=8, normalize=True)
        grid_recon = torchvision.utils.make_grid(x_recons, nrow=8, normalize=True)
        show_grid([grid_orig, grid_recon])

    def configure_model(self):
        """
        Configure the model, i.e. create an VQ-VAE instance.
        :return:
        """
        model_conf = self.conf["model"]
        if model_conf["type"] == "TinyVQVAE":
            params = model_conf["params"]
            return TinyVQVAE(
                in_channels=self.conf["dataset"]["num_channels"],
                **params
            )
        else:
            raise NotImplementedError(f"Model {model_conf['type']} not implemented")

    def configure_patch_handler(self):
        """
        Configure the patch handler, i.e. create a Patches2D instance. This instance is used to split
        the images into patches.
        :return: Patches2D instance.
        """
        if self.conf["dataset"]["name"].lower() == "mnist":
            padding_mode = "constant"
            padding_value = 0 - self.conf["dataset"]["mean"][0] / self.conf["dataset"]["std"][0]
            padding_height = 32 - self.conf["dataset"]["img_height"]
            padding_width = 32 - self.conf["dataset"]["img_width"]
        elif self.conf["dataset"]["name"].lower().startswith("cifar"):
            padding_mode, padding_value = "constant", 0
            padding_height, padding_width = 0, 0
        else:
            raise NotImplementedError(f"Dataset {self.conf['dataset']['name']} not implemented")

        return Patches2D(
            image_height=self.conf["dataset"]["img_height"],
            image_width=self.conf["dataset"]["img_width"],
            patch_height=self.conf["dataset"]["patch_height"],
            patch_width=self.conf["dataset"]["patch_width"],
            padding_height=padding_height,
            padding_width=padding_width,
            padding_mode=padding_mode,
            padding_value=padding_value
        )

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[ReduceLROnPlateau]]:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer.
        """
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)


class VQVAEFeatureExtractorImageMode(BaseLitModule):
    """
    Extract features from non-overlapping patches of an image using a VQ-VAE.
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        q_loss, x_recon, perplexity, encodings = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x) / self.data_var
        loss = q_loss + recon_loss
        self.log_step(
            processed_values={"loss": loss, "q_loss": q_loss, "recon_loss": recon_loss, "perplexity": perplexity},
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

        loss_, x_recon, perplexity, encodings = self.forward(x)
        x_recon = undo_norm_from_conf(x_recon, self.conf)
        x_recon = torch.clip(x_recon, 0, 1)
        grid_orig = torchvision.utils.make_grid(undo_norm_from_conf(x, self.conf), nrow=8, normalize=True)
        grid_recon = torchvision.utils.make_grid(x_recon, nrow=8, normalize=True)
        show_grid([grid_orig, grid_recon])

    def visualize_encodings(self, batch: Tensor, batch_idx: int):
        """
        Visualize the encodings of a batch of samples.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        """
        x, y = self.preprocess_data_(batch)
        if x.shape[0] > 20:
            x = x[:20, ...]

        loss_, x_recon, perplexity, encodings = self.forward(x)
        x = undo_norm_from_conf(x, self.conf)
        x_recon = undo_norm_from_conf(x_recon, self.conf)
        x_recon = torch.clip(x_recon, 0, 1)
        encodings = bin2dec(encodings.permute(0, 2, 3, 1), encodings.shape[1]).unsqueeze(1)

        images, masks, titles = [], [], []
        for b in range(x.shape[0]):
            images.extend([x[b], x_recon[b], x_recon[b]])
            masks.extend([None, None, encodings[b]])
            titles.extend([f"Orig. {y[b]}", f"Recon. {y[b]}", f"Encodings {y[b]}"])
        plot_images(images=images, masks=masks, titles=titles, max_cols=3)

    def configure_model(self):
        """
        Configure the model, i.e. create an VQ-VAE instance.
        :return:
        """
        model_conf = self.conf["model"]
        if model_conf["type"] == "SmallVQVAE":
            params = model_conf["params"]
            return SmallVQVAE(
                in_channels=self.conf["dataset"]["num_channels"],
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
