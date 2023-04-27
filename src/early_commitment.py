"""

Script to investigate early commitment of models.

"""
import torchvision
from torch.utils.data import DataLoader

from utils import get_config, print_warn, print_start
from data import loaders_from_config
from lightning.fabric import Fabric
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from tools import torch_optim_from_conf, loggers_from_conf
import argparse
from typing import Optional, Dict, Any
from models.lightning_modules.lightning_base import BaseLitModule
from torchvision.datasets.vision import VisionDataset


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser:
    :return:
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Analysis of Early Commitment")
    parser.add_argument("config",
                        type=str,
                        help="Path to the config file"
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
                        help='Log to wandb')

    args = parser.parse_args()
    return args


class EarlyCommitmentModule(BaseLitModule):
    """
    Lightning Module for Early Commitment Analysis.
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__(conf, fabric, logging_prefixes=["train", "val"])
        self.model = self.configure_model()

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.
        :param x: Input tensor.
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Forward training step: Forward pass, loss computation, logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the training step.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log_step(loss, y_hat, y, prefix="train")
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Forward validation step: Forward pass, loss computation, logging.
        :param batch: Data batch, containing input data and labels.
        :param batch_idx: Index of the batch.
        :return: Loss of the validation step.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log_step(loss, y_hat, y, prefix="val")
        return loss

    def configure_model(self) -> nn.Module:
        """
        Configure (create instance) the model.
        :return: A torch model.
        """
        model = torchvision.models.resnet18(weights=None, num_classes=self.conf['dataset']['num_classes'])
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer.
        """
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)

    def on_epoch_end(self):
        """
        Callback at the end of an epoch.
        """
        super().on_epoch_end()


def setup_components(config: Dict[str, Optional[Any]]) -> (
Fabric, EarlyCommitmentModule, torch.optim.Optimizer, pl.LightningDataModule):
    """
    Setup components for training.
    :param config: Configuration dict
    :return: Returns the Fabric, the model, the optimizer, the train dataloader and the test dataloader.
    """
    train_loader, _, test_loader = loaders_from_config(config)
    loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers)
    fabric.launch()
    fabric.seed_everything(1)
    model = EarlyCommitmentModule(config, fabric)
    optimizer = model.configure_optimizers()
    model, optimizer = fabric.setup(model, optimizer)
    if isinstance(train_loader, DataLoader):
        train_loader = fabric.setup_dataloaders(train_loader)
        test_loader = fabric.setup_dataloaders(test_loader)
    else:
        print_warn("Train and test loader not setup with fabric.", "Fabric Warning:")

    return fabric, model, optimizer, train_loader, test_loader


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        model: EarlyCommitmentModule,
        optimizer: torch.optim.Optimizer,
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
    """
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader),
                             total=len(test_dataloader),
                             colour="GREEN",
                             desc=f"Validate Epoch {epoch + 1}/{config['run']['n_epochs']}"):
            model.validation_step(batch, i)


def main():
    """
    Main function.
    """
    print_start("Starting python script 'early_commitment.py'...", title="Launching Early Commitment Analysis")
    args = parse_args()
    config = get_config(args.config, args)
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")

    fabric, model, optimizer, train_dataloader, test_dataloader = setup_components(config)

    for epoch in range(config['run']['n_epochs']):
        single_train_epoch(config, fabric, model, optimizer, train_dataloader, epoch)
        single_eval_epoch(config, model, test_dataloader, epoch)
        model.on_epoch_end()


if __name__ == '__main__':
    main()
