import argparse
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Fabric
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import loaders_from_config, plot_images
from stage_1.feature_extractor.straight_line_pl_modules import FixedFilterFeatureExtractor
from stage_1.lateral.lateral_connections_toy import LateralNetwork
from utils import get_config, print_start, print_warn


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse arguments from command line.
    :param parser: Optional ArgumentParser instance.
    :return: Parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Lateral Connections Stage 1")
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


def configure() -> Dict[str, Optional[Any]]:
    """
    Load the config based on the given console args.
    :return: Configuration dict.
    """
    args = parse_args()
    config = get_config(args.config, args)
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")
    return config


def setup_fabric() -> Fabric:
    """
    Setup the fabric instance.
    :return: Fabric instance.
    """
    fabric = Fabric(accelerator="auto", devices=1, loggers=[], callbacks=[])
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_dataloader(config: Dict[str, Optional[Any]], fabric: Fabric) -> (DataLoader, DataLoader):
    """
    Setup the dataloaders for training and testing.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Returns the training and testing dataloader
    """
    train_loader, _, test_loader = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)
    return train_loader, test_loader


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





def single_train_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: pl.LightningModule,
        train_loader: DataLoader,
        epoch: int,
):
    """
        Train the model for a single epoch.
        :param config: Configuration dict.
        :param feature_extractor: Feature extractor model.
        :param lateral_network: Laternal network model.
        :param train_loader: Test set dataloader.
        :param epoch: Current epoch.
        """
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch + 1}/{config['run']['n_epochs']}"):
        with torch.no_grad():  # No gradients updates (Hebbian learning)
            features = feature_extractor(batch)
        lateral_network.model.train_ = True
        lateral_network.forward_steps_through_time(features)
        lateral_network.model.train_ = False

        # Plot sample
        if i == (len(train_loader) - 1) and config['run']['visualize_plots']:
            lateral_network.plot_features_single_sample(batch, features)
            lateral_network.plot_model_weights()


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: pl.LightningModule,
        test_loader: DataLoader,
        epoch: int,
):
    """
    Evaluate the model for a single epoch.
    :param config: Configuration dict.
    :param feature_extractor: Feature extractor model.
    :param lateral_network: Laternal network model.
    :param test_loader: Test set dataloader.
    :param epoch: Current epoch.
    """
    pass


def train(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: pl.LightningModule,
        train_loader: DataLoader,
        test_loader: DataLoader):
    """
    Train the model.
    :param config: Configuration dict
    :param feature_extractor: Feature extractor module
    :param train_loader: Training dataloader
    :param test_loader: Testing dataloader
    """
    start_epoch = config['run']['current_epoch']
    for epoch in range(start_epoch, config['run']['n_epochs']):
        config['run']['current_epoch'] = epoch
        single_train_epoch(config, feature_extractor, lateral_network, train_loader, epoch)
        lateral_network.on_epoch_end()


def setup_lateral_network(config, fabric) -> pl.LightningModule:
    """
    Setup the model using lateral connections.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Model using lateral connections.
    """
    return fabric.setup(LateralNetwork(config, fabric))


def main():
    """
    Run the model: Create modules, extract features from images and run the model leveraging lateral connections.
    """
    print_start("Starting python script 's1_lateral_connections.py'...",
                title="Training S1: Lateral Connections")
    config = configure()
    fabric = setup_fabric()
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    feature_extractor.eval()  # does not have to be trained
    lateral_network = setup_lateral_network(config, fabric)
    lateral_network.train()
    train(config, feature_extractor, lateral_network, train_loader, test_loader)


if __name__ == '__main__':
    main()
