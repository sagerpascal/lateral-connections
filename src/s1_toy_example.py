import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import wandb
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import loaders_from_config
from stage_1.feature_extractor.straight_line_pl_modules import FixedFilterFeatureExtractor
from stage_1.lateral.lateral_connections_toy import LateralNetwork
from tools import loggers_from_conf
from tools.store_load_run import load_run, save_run
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
                        dest='run:plots:enable',
                        help='Plot results'
                        )
    parser.add_argument('--plot_dir',
                        type=str,
                        dest='run:plots:store_path',
                        help='Store the plotted results in the given path'
                        )
    parser.add_argument("--train_noise",
                        type=float,
                        # default=0.,
                        dest="dataset:train_dataset_params:noise",
                        help="The noise added to the training data (default: 0.)"
                        )
    parser.add_argument("--valid_noise",
                        type=float,
                        # default=0.005,
                        dest="dataset:valid_dataset_params:noise",
                        help="The noise added to the validation data (default: 0.005)"
                        )
    parser.add_argument("--test_noise",
                        type=float,
                        # default=0.005,
                        dest="dataset:test_dataset_params:noise",
                        help="The noise added to the test data (default: 0.005)"
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


def setup_fabric(config: Dict[str, Optional[Any]]) -> Fabric:
    """
    Setup the fabric instance.
    :param config: Configuration dict
    :return: Fabric instance.
    """
    loggers = loggers_from_conf(config)
    #torch.backends.cudnn.deterministic = True
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=[])
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
    feature_extractor.eval()
    lateral_network.train()
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            features = feature_extractor(batch)
        lateral_network(features)


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
    feature_extractor.eval()
    lateral_network.eval()
    plt_img, plt_features, plt_input_features, plt_activations, plt_activations_f = [], [], [], [], []
    for i, batch in tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         colour="GREEN",
                         desc=f"Testing Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            features = feature_extractor(batch)
            input_features, lateral_features, lateral_features_f = lateral_network(features)
            plt_img.append(batch)
            plt_features.append(features)
            plt_input_features.append(input_features)
            plt_activations.append(lateral_features)
            plt_activations_f.append(lateral_features_f)

    plot = config['run']['plots']['enable'] and \
           (not config['run']['plots']['only_last_epoch'] or epoch == config['run']['n_epochs'])
    wandb_b = config['logging']['wandb']['active']
    store_plots = config['run']['plots'].get('store_path', False)

    assert not wandb_b or wandb_b and store_plots, "Wandb logging requires storing the plots."

    if plot or wandb_b or store_plots:
        plots_fp = lateral_network.plot_samples(plt_img,
                                                plt_features,
                                                plt_input_features,
                                                plt_activations,
                                                plt_activations_f,
                                                plot_input_features=True, #epoch == 0,
                                                show_plot=plot)
        weights_fp = lateral_network.plot_model_weights(show_plot=plot)
        if epoch == config['run']['n_epochs']:
            videos_fp = lateral_network.create_activations_video(plt_img, plt_input_features, plt_activations)

        if wandb_b:
            logs = {str(pfp.name[:-4]): wandb.Image(str(pfp)) for pfp in plots_fp}
            logs |= {str(wfp.name[:-4]): wandb.Image(str(wfp)) for wfp in weights_fp}
            if epoch == config['run']['n_epochs']:
                logs |= {str(vfp.name[:-4]): wandb.Video(str(vfp)) for vfp in videos_fp}
            wandb.log(logs | {"epoch": epoch}, step=epoch)


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

    # if config['logging']['wandb']['active'] or config['run']['plots']['enable']:
    #     single_eval_epoch(config, feature_extractor, lateral_network, test_loader, 0)

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch(config, feature_extractor, lateral_network, train_loader, epoch+1)
        single_eval_epoch(config, feature_extractor, lateral_network, test_loader, epoch+1)
        lateral_network.on_epoch_end()
        config['run']['current_epoch'] = epoch + 1


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
    print_start("Starting python script 's1_toy_example.py'...",
                title="Training S1: Lateral Connections Toy Example")
    config = configure()
    fabric = setup_fabric(config)
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)

    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, state = load_run(config, fabric)
        feature_extractor.load_state_dict(state['feature_extractor'])
        lateral_network.load_state_dict(state['lateral_network'])

    feature_extractor.eval()  # does not have to be trained
    if 'store_path' in config['run']['plots'] and config['run']['plots']['store_path'] != 'None':
        fp = Path(config['run']['plots']['store_path'])
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)
    train(config, feature_extractor, lateral_network, train_loader, test_loader)

    if 'store_state_path' in config['run'] and config['run']['store_state_path'] != 'None':
        save_run(config, fabric,
                 components={'feature_extractor': feature_extractor, 'lateral_network': lateral_network})


if __name__ == '__main__':
    main()
