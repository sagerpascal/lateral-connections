import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytz
import torch
from lightning.fabric import Fabric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import loaders_from_config
from models import BaseLitModule
from lateral_connections.feature_extractor import VQVAEFeatureExtractorImageMode, VQVAEFeatureExtractorPatchMode
from tools import loggers_from_conf
from utils import get_config, print_start, print_warn
from tools.callbacks.save_model import SaveBestModelCallback
from tools.store_load_run import load_run


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
    if config["run"]["mode"] == "patches":
        return VQVAEFeatureExtractorPatchMode(config, fabric)
    elif config["run"]["mode"] == "full_image":
        return VQVAEFeatureExtractorImageMode(config, fabric)
    else:
        raise ValueError(f"Mode {config['run']['mode']} unknown / not implemented")


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


def setup_modules(config: Dict[str, Optional[Any]]) -> Tuple[Fabric, BaseLitModule, Optimizer, Optional[ReduceLROnPlateau],
DataLoader, DataLoader]:
    """
    Setup the modules for training.
    :param config: Configuration dict
    :return: Returns the fabric, model, optimizer, scheduler, training dataloader and testing dataloader
    """
    fabric = setup_fabric(config)
    model, optimizer, scheduler = setup_components(config, fabric)
    train_dataloader, test_dataloader = setup_dataloader(config, fabric)
    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, components = load_run(config, fabric)
        model.load_state_dict(components['model'])
        optimizer.load_state_dict(components['optimizer'])
        scheduler.load_state_dict(components['scheduler'])
    return fabric, model, optimizer, scheduler, train_dataloader, test_dataloader


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
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
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
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
            model.validation_step(batch, i)
            if config['run']['visualize_plots'] and i == 0:
                model.visualize_samples(batch, i)
                if hasattr(model, "visualize_encodings"):
                    model.visualize_encodings(batch, i)


def train_feature_extractor(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
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
        single_train_epoch(config, fabric, model, optimizer, train_dataloader, epoch)
        single_eval_epoch(config, model, test_dataloader, epoch)
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
    print_start("Starting python script 's1_feature_extractor_vq_vae.py'...",
                title="Training S1: VQ-VAE Feature Extractor")
    config = configure()
    fabric, model, optimizer, scheduler, train_dataloader, test_dataloader = setup_modules(config)
    train_feature_extractor(config, fabric, model, optimizer, scheduler, train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
