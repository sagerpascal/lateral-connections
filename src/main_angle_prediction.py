import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import loaders_from_config
from lateral_connections.feature_extractor.straight_line_pl_modules import FixedFilterFeatureExtractor
from lateral_connections.s1_lateral_connections import LateralNetwork
from models import BaseLitModule
from tools import loggers_from_conf, torch_optim_from_conf
from tools.callbacks.save_model import SaveBestModelCallback
from tools.store_load_run import load_run
from utils import get_config, print_start, print_warn


class AnglePredictionTorch(nn.Module):

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.model = self.setup_model()

    def setup_model(self) -> nn.Sequential:
        return nn.Sequential(*[
            nn.Conv2d(4, 32, kernel_size=self.kernel_size, stride=1, padding="same"),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class AnglePredictor(BaseLitModule):
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
        angle_pred = self.forward(x).squeeze(1)
        angle_gt = y['angle'].float()
        loss = F.mse_loss(angle_pred, angle_gt)
        self.log_step(
            processed_values={"loss": loss},
            metric_pairs=[(angle_pred, angle_gt)],
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

    def configure_model(self):
        """
        Configure the model, i.e. create an VQ-VAE instance.
        :return:
        """
        model_conf = self.conf["model"]
        if model_conf["type"] == "angle-predictor":
            params = model_conf["params"]
            return AnglePredictionTorch(
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


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Feature Extractor Stage 1")
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
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        dest='run:visualize_plots',
                        help='Plot results'
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

    args = parser.parse_args()
    return args


def get_model(config: Dict[str, Optional[Any]], fabric: Fabric) -> BaseLitModule:
    """
    Get the model according to configuration.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: The model
    """
    return AnglePredictor(config, fabric)


def setup_fabric(config: Optional[Dict[str, Optional[Any]]] = None) -> Fabric:
    """
    Setup the Fabric instance.
    :param config: Configuration dict
    :return: Fabric instance
    """
    if config is None:
        callbacks, loggers = [], []
    else:
        callbacks = []
        if "store_state_path" in config["run"] and config["run"]["store_state_path"] != 'None':
            callbacks.append(SaveBestModelCallback(metric_key="val/loss", mode="min"))
        loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=callbacks)
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_components(config: Dict[str, Optional[Any]], fabric: Fabric) -> (
        BaseLitModule, Optimizer, Optional[ReduceLROnPlateau]):
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


def configure() -> Dict[str, Optional[Any]]:
    """
    Load the config based on the given console args.
    :return:
    """
    args = parse_args()
    config = get_config(args.config, args)
    if config['run']['store_state_path'] != 'None' and Path(config['run']['store_state_path']).is_dir():
        f_name = f"s1_{datetime.now(pytz.timezone('Europe/Zurich')).strftime('%Y-%m-%d_%H-%M-%S')}.ckpt"
        config['run']['store_state_path'] = config['run']['store_state_path'] + f"/{f_name}"
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")
    return config


def setup_feature_extractor(config: Dict[str, Optional[Any]], fabric: Fabric) -> pl.LightningModule:
    """
    Setup the feature extractor model that is used to extract features from images before they are fed into the model
    leveraging lateral connections.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Feature extractor model.
    """
    feature_extractor = FixedFilterFeatureExtractor(config, fabric)
    feature_extractor = fabric.setup(feature_extractor)
    return feature_extractor


def setup_modules(config: Dict[str, Optional[Any]]) -> Tuple[
    Fabric, pl.LightningModule, BaseLitModule, Optimizer, Optional[ReduceLROnPlateau],
    DataLoader, DataLoader]:
    """
    Setup the modules for training.
    :param config: Configuration dict
    :return: Returns the fabric, model, optimizer, scheduler, training dataloader and testing dataloader
    """
    fabric = setup_fabric(config)
    feature_extractor = setup_feature_extractor(config, fabric)
    model, optimizer, scheduler = setup_components(config, fabric)
    train_dataloader, test_dataloader = setup_dataloader(config, fabric)
    # if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
    #     config, components = load_run(config, fabric)
    #     model.load_state_dict(components['model'])
    #     optimizer.load_state_dict(components['optimizer'])
    #     scheduler.load_state_dict(components['scheduler'])
    return fabric, feature_extractor, model, optimizer, scheduler, train_dataloader, test_dataloader


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
lateral_network: LateralNetwork,
        feature_extractor: pl.LightningModule,
        model: BaseLitModule,
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
        with torch.no_grad():
            batch[0] = feature_extractor.binarize_features(feature_extractor(batch[0]).squeeze(1))

        if False:
            lateral_network.new_sample()
            z = torch.zeros((batch[0].shape[0], lateral_network.model.out_channels, batch[0].shape[2],
                             batch[0].shape[3]), device=batch[0].device)

            for t in range(config["lateral_model"]["max_timesteps"]):
                lateral_network.model.update_ts(t)
                x_in = torch.cat([batch[0], z], dim=1)
                z_float, z = lateral_network(x_in)

            batch[0] = z

        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        lateral_network: LateralNetwork,
        feature_extractor: pl.LightningModule,
        model: BaseLitModule,
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
            batch[0] = feature_extractor.binarize_features(feature_extractor(batch[0]).squeeze(1))

            if True:
                features_s = batch[0].shape
                num_elements = batch[0].numel()
                num_flips = int(0.03 * num_elements)
                random_mask = torch.randperm(num_elements)[:num_flips]
                random_mask = torch.zeros(num_elements, dtype=torch.bool).scatter(0, random_mask, 1)
                batch[0] = batch[0].view(-1)
                batch[0][random_mask] = 1.0 - batch[0][random_mask]
                batch[0] = batch[0].view(features_s)

            if False:
                lateral_network.new_sample()
                z = torch.zeros((batch[0].shape[0], lateral_network.model.out_channels, batch[0].shape[2],
                                 batch[0].shape[3]), device=batch[0].device)

                for t in range(config["lateral_model"]["max_timesteps"]):
                    lateral_network.model.update_ts(t)
                    x_in = torch.cat([batch[0], z], dim=1)
                    z_float, z = lateral_network(x_in)

                batch[0] = z
            model.validation_step(batch, i)



def train(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        lateral_network: LateralNetwork,
        feature_extractor: pl.LightningModule,
        model: BaseLitModule,
        optimizer: Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
):
    """
    Train the feature extractor for multiple epochs.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :param model: Model to train
    :param optimizer: Optimizer to use
    :param scheduler: LR scheduler to use
    :param train_dataloader: Training dataloader
    :param test_dataloader: Testing dataloader
    :return:
    """
    start_epoch = config['run']['current_epoch']
    for epoch in range(start_epoch, config['run']['n_epochs']):
        config['run']['current_epoch'] = epoch
        single_train_epoch(config, fabric, lateral_network, feature_extractor, model, optimizer, train_dataloader, epoch)
        single_eval_epoch(config, lateral_network, feature_extractor, model, test_dataloader, epoch)
        logs = model.on_epoch_end()
        if scheduler is not None:
            scheduler.step(logs["val/loss"])
        fabric.call("on_epoch_end", config=config, logs=logs, fabric=fabric,
                    components={"model": model, "optimizer": optimizer, "scheduler": scheduler.state_dict()})
    fabric.call("on_train_end")


def main():
    """
    Run the model and store the model with the lowest loss.
    """
    print_start("Starting python script 'main_autoencoder.py'...",
                title="Training S0: Autoencoder Feature Extractor")
    config = configure()
    fabric, feature_extractor, model, optimizer, scheduler, train_dataloader, test_dataloader = setup_modules(config)
    lateral_network = fabric.setup(LateralNetwork(config, fabric))
    config2, state = load_run(config, fabric)
    lateral_network.load_state_dict(state['lateral_network'])
    train(config, fabric, lateral_network, feature_extractor, model, optimizer, scheduler, train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
