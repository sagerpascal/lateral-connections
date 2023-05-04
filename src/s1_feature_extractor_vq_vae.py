import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytz
import torch
import torch.nn.functional as F
import torchvision
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import plot_images, undo_norm_from_conf
from data.utils.patches2d import Patches2D
from early_commitment import EarlyCommitmentModule
from models.autoencoder.vq_vae import SmallVQVAE
from src.data import loaders_from_config, show_grid
from src.models import BaseLitModule, TinyVQVAE
from src.tools import loggers_from_conf, torch_optim_from_conf
from src.tools.custom_math import bin2dec
from src.utils import get_config, print_start, print_warn
from tools.callbacks.save_model import SaveBestModelCallback
from tools.store_load_run import load_run


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Analysis of Early Commitment")
    parser.add_argument("config",
                        type=str,
                        help="Path to the config file",
                        )
    parser.add_argument("--batch-size",
                        type=int,
                        # default=64,
                        metavar="N",
                        dest="dataset:batch_size",
                        help="input batch size for training (default: 64)"
                        )
    parser.add_argument("--epochs",
                        type=int,
                        # default=20,
                        metavar="N",
                        dest="run:n_epochs",
                        help="number of epochs to train (default: 10)"
                        )
    parser.add_argument("--lr",
                        type=float,
                        # default=0.001,
                        metavar="LR",
                        dest="optimizers:opt1:params:lr",
                        help="learning rate (default: 0.001)"
                        )
    parser.add_argument('--wandb',
                        action='store_true',
                        default=False,
                        dest='logging:wandb:active',
                        help='Log to wandb'
                        )
    parser.add_argument('--store',
                        type=str,
                        dest='run:store_state_path',
                        help='Path where the model will be stored'
                        )
    parser.add_argument('--load',
                        type=str,
                        dest='run:load_state_path',
                        help='Path from where the model will be loaded'
                        )

    store_state_path: None
    load_state_path: None

    args = parser.parse_args()
    return args


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

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[LRScheduler]]:
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

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[LRScheduler]]:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer and scheduler.
        """
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)


def get_model(config: Dict[str, Optional[Any]], fabric: Fabric) -> BaseLitModule:
    """
    Get the model according to configuration.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: The model
    """
    if config["run"]["mode"] == "patches":
        return VQVAEFeatureExtractorPatchMode(config, fabric)
    elif config["run"]["mode"] == "full_image":
        return VQVAEFeatureExtractorImageMode(config, fabric)
    else:
        raise ValueError(f"Mode {config['run']['mode']} unknown / not implemented")


def setup_fabric(config: Dict[str, Optional[Any]]) -> Fabric:
    """
    Setup the Fabric instance.
    :param config: Configuration dict
    :return: Fabric instance
    """
    callbacks = []
    if "store_state_path" in config["run"] and config["run"]["store_state_path"] != 'None':
        callbacks.append(SaveBestModelCallback(metric_key="val/loss", mode="min"))
    loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=callbacks)
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_components(config: Dict[str, Optional[Any]], fabric: Fabric) -> (
        VQVAEFeatureExtractorPatchMode, Optimizer, Optional[LRScheduler]):
    """
    Setup components for training.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the model and the optimizer
    """
    model = get_model(config, fabric)
    optimizer, scheduler = model.configure_optimizers()
    model, optimizer = fabric.setup(model, optimizer)
    return model, optimizer, scheduler


def setup_dataloader(config: Dict[str, Optional[Any]], fabric: Fabric) -> (DataLoader, DataLoader):
    """
    Setup the dataloaders for training and testing.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the training and testing dataloader
    """
    train_loader, _, test_loader = loaders_from_config(config)
    if isinstance(train_loader, DataLoader):
        train_loader = fabric.setup_dataloaders(train_loader)
        test_loader = fabric.setup_dataloaders(test_loader)
    else:
        print_warn("Train and test loader not setup with fabric.", "Fabric Warning:")

    return train_loader, test_loader


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        model: EarlyCommitmentModule,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        epoch: int,
):
    """
    Train a single epoch.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param train_dataloader: Training dataloader
    :param epoch: Current epoch
    :return: Returns the training logs
    """
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader),
                         total=len(train_dataloader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch + 1}/{config['run']['n_epochs']}"):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        model: EarlyCommitmentModule,
        test_dataloader: DataLoader,
        epoch: int,
):
    """
    Evaluate a single epoch.
    :param config: Configuration dict
    :param model: The model to evaluate
    :param test_dataloader: Testing dataloader
    :param epoch: Current epoch
    :return: Returns the validation logs
    """
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader),
                             total=len(test_dataloader),
                             colour="GREEN",
                             desc=f"Validate Epoch {epoch + 1}/{config['run']['n_epochs']}"):
            model.validation_step(batch, i)
            if i == 0:
                model.visualize_samples(batch, i)
                if hasattr(model, "visualize_encodings"):
                    model.visualize_encodings(batch, i)


def train():
    """
    Run the model and store the models activations in the last epoch.
    """
    print_start("Starting python script 's1_feature_extractor_vq_vae.py'...",
                title="Training S1: VQ-VAE Feature Extractor")

    # -------------------- Setup Config -------------------- #
    args = parse_args()
    config = get_config(args.config, args)
    if config['run']['store_state_path'] != 'None' and Path(config['run']['store_state_path']).is_dir():
        f_name = f"s1_{datetime.now(pytz.timezone('Europe/Zurich')).strftime('%Y-%m-%d_%H-%M-%S')}.ckpt"
        config['run']['store_state_path'] = config['run']['store_state_path'] + f"/{f_name}"
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")

    # -------------------- Setup Modules -------------------- #
    fabric = setup_fabric(config)
    model, optimizer, scheduler = setup_components(config, fabric)
    train_dataloader, test_dataloader = setup_dataloader(config, fabric)
    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, components = load_run(config, fabric)
        model.load_state_dict(components['model'])
        optimizer.load_state_dict(components['optimizer'])
        scheduler.load_state_dict(components['scheduler'])

    # -------------------- Run Training -------------------- #
    start_epoch = config['run']['current_epoch']
    for epoch in range(start_epoch, config['run']['n_epochs']):
        config['run']['current_epoch'] = epoch
        single_train_epoch(config, fabric, model, optimizer, train_dataloader, epoch)
        single_eval_epoch(config, model, test_dataloader, epoch)
        logs = model.on_epoch_end()
        if scheduler is not None:
            scheduler.step(logs["val/loss"])
        fabric.call("on_epoch_end", config=config, logs=logs, fabric=fabric,
                    components={"model": model, "optimizer": optimizer, "scheduler": scheduler})
    fabric.call("on_train_end")


if __name__ == '__main__':
    train()
