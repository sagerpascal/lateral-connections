import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import loaders_from_config
from lateral_connections.feature_extractor.straight_line_pl_modules import FixedFilterFeatureExtractor
from lateral_connections.s2_rbm import L2RBM
from lateral_connections.s1_lateral_connections import LateralNetwork
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
    torch.backends.cudnn.deterministic = True
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
    # torch.backends.cudnn.deterministic = True
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


def cycle(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        l2: L2RBM,
        batch: Tensor,
        batch_idx: int,
        epoch: int,
        store_tensors: Optional[bool] = False,
        mode: Optional[str] = "train",
        fabric: Optional[Fabric] = None,
        l2_opt: Optional[Optimizer] = None,
):
    """
    Perform a single cycle of the model.
    :param config: Configuration dict
    :param feature_extractor: The feature extractor model to extract features from a given image.
    :param lateral_network: The network building sub-networks by using lateral connections
    :param l2: The L2RBM model
    :param batch: The images to process.
    :param batch_idx: The index of the batch.
    :param epoch: Current epoch
    :param store_tensors: Whether to store the tensors and return them.
    :param mode: The mode of the cycle, either train or eval.
    :param fabric: The fabric instance.
    :param l2_opt: The optimizer for the L2-model.
    :return: The features extracted from the image, the binarized features fed into the network with lateral
    connections, the features after lateral connections (binary) and the features after lateral connections as float
    """
    assert mode in ["train", "eval"], "Mode must be either train or eval"
    assert mode == "train" and fabric is not None or mode == "eval", "Fabric must be given in train mode"
    assert mode == "train" and fabric is not None or mode == "eval", "Optimizer must be given in train mode"

    with torch.no_grad():
        features = feature_extractor(batch)

    lateral_network.new_sample()
    z = None

    input_features, lateral_features, lateral_features_f, l2_features, l2h_features = [], [], [], [], []
    for view_idx in range(features.shape[1]):
        x_view_features = features[:, view_idx, ...]
        x_view_features = feature_extractor.binarize_features(x_view_features)

        # Add noise to the input features -> should be removed by net fragments
        # x_view_features = np.array(x_view_features.detach().cpu())
        # x_view_features = x_view_features + np.random.choice(2, x_view_features.shape, p=[1 - 0.005, 0.005])
        # x_view_features = torch.from_numpy(x_view_features).cuda().float()
        # x_view_features = feature_extractor.binarize_features(x_view_features)

        if store_tensors:
            input_features.append(x_view_features)

        if z is None:
            z = torch.zeros((x_view_features.shape[0], lateral_network.model.out_channels, x_view_features.shape[2],
                             x_view_features.shape[3]), device=batch.device)

        features_lat, features_lat_float, features_l2, features_l2_h = [], [], [], []
        for t in range(config["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([x_view_features, z], dim=1)
            z_float, z = lateral_network(x_in)

            z2, z2_feedback, h, loss = l2.eval_step(z)

            if epoch > 10:
                mask_active = (z > 0) | (z2_feedback > 0)
                if F.mse_loss(z[mask_active], z2_feedback[mask_active]) < .1:
                    z = z2_feedback

            features_lat.append(z)
            if store_tensors:
                features_lat_float.append(z_float)
                features_l2.append(z2_feedback)
                features_l2_h.append(h)

        features_lat = torch.stack(features_lat, dim=1)
        features_lat_median = torch.median(features_lat, dim=1)[0]
        if store_tensors:
            features_lat_float = torch.stack(features_lat_float, dim=1)
            features_l2 = torch.stack(features_l2, dim=1)
            features_l2_h = torch.stack(features_l2_h, dim=1)

        if mode == "train":  # TODO: Train at the end after all timesteps (use median activation per cell),
            # also update L1 after training
            # Train L1
            x_rearranged = lateral_network.model.l1.rearrange_input(
                torch.cat([x_view_features, features_lat_median], dim=1))
            lateral_network.model.l1.hebbian_update(x_rearranged, features_lat_median)

            # Train L2
            l2_opt.zero_grad()
            z2, z2_feedback, h, loss = l2.train_step(features_lat_median)
            fabric.backward(loss)
            l2_opt.step()

        if store_tensors:
            features_lat_float_median = torch.median(features_lat_float, dim=1)[0]
            features_l2_median = torch.median(features_l2, dim=1)[0]
            l2h_features_median = torch.median(features_l2_h, dim=1)[0]
            features_lat = torch.cat([features_lat, features_lat_median.unsqueeze(1)], dim=1)
            features_lat_float = torch.cat([features_lat_float, features_lat_float_median.unsqueeze(1)], dim=1)
            features_l2 = torch.cat([features_l2, features_l2_median.unsqueeze(1)], dim=1)
            features_l2_h = torch.cat([features_l2_h, l2h_features_median.unsqueeze(1)], dim=1)
            lateral_features.append(features_lat)
            lateral_features_f.append(features_lat_float)
            l2_features.append(features_l2)
            l2h_features.append(features_l2_h)

    if store_tensors:
        return features, torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), torch.stack(
            lateral_features_f, dim=1), torch.stack(l2_features, dim=1), torch.stack(l2h_features, dim=1)


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        l2: L2RBM,
        train_loader: DataLoader,
        epoch: int,
        fabric: Fabric,
        l2_opt: Optimizer,
):
    """
        Train the model for a single epoch.
        :param config: Configuration dict.
        :param feature_extractor: Feature extractor model.
        :param lateral_network: Laternal network model.
        :param l2: The L2RBM model
        :param train_loader: Test set dataloader.
        :param epoch: Current epoch.
        :param fabric: The fabric instance.
        :param l2_opt: The optimizer for the L2-model.
        """
    feature_extractor.eval()
    lateral_network.eval()
    l2.train()
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch}/{config['run']['n_epochs']}"):
        cycle(config, feature_extractor, lateral_network, l2, batch[0], i, epoch=epoch, store_tensors=False,
              mode="train",
              fabric=fabric, l2_opt=l2_opt)


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        l2: L2RBM,
        test_loader: DataLoader,
        epoch: int,
):
    """
    Evaluate the model for a single epoch.
    :param config: Configuration dict.
    :param feature_extractor: Feature extractor model.
    :param lateral_network: Laternal network model.
    :param l2: The L2RBM model
    :param test_loader: Test set dataloader.
    :param epoch: Current epoch.
    """
    feature_extractor.eval()
    lateral_network.eval()
    l2.eval()
    plt_img, plt_features, plt_input_features, plt_activations, plt_activations_f, plt_activations_l2 = [], [], [], \
        [], [], []
    for i, batch in tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         colour="GREEN",
                         desc=f"Testing Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            features, input_features, lateral_features, lateral_features_f, l2_features, l2_h_features = cycle(config,
                                                                                                               feature_extractor,
                                                                                                               lateral_network,
                                                                                                               l2,
                                                                                                               batch[0],
                                                                                                               i,
                                                                                                               epoch=epoch,
                                                                                                               store_tensors=True,
                                                                                                               mode="eval")
            plt_img.append(batch[0])
            plt_features.append(features)
            plt_input_features.append(input_features)
            plt_activations.append(lateral_features)
            plt_activations_f.append(lateral_features_f)
            plt_activations_l2.append(l2_features)

    plot = config['run']['plots']['enable'] and \
           (not config['run']['plots']['only_last_epoch'] or epoch == config['run']['n_epochs'])
    wandb_b = config['logging']['wandb']['active']
    store_plots = config['run']['plots'].get('store_path', False)

    assert not wandb_b or wandb_b and store_plots, "Wandb logging requires storing the plots."

    if plot or wandb_b or store_plots:
        if epoch == 0:
            feature_extractor.plot_model_weights(show_plot=plot)
#
        plots_fp = lateral_network.plot_samples(plt_img,
                                                plt_features,
                                                plt_input_features,
                                                plt_activations,
                                                plt_activations_f,
                                                plot_input_features=epoch == 0,
                                                show_plot=plot)
        # weights_fp = lateral_network.plot_model_weights(show_plot=plot)
        # plots_l2_fp = l2.plot_samples(plt_img, plt_activations_l2, show_plot=plot)
        if epoch == config['run']['n_epochs']:
            videos_fp = lateral_network.create_activations_video(plt_img, plt_input_features, plt_activations)

        if wandb_b:
            logs = {str(pfp.name[:-4]): wandb.Image(str(pfp)) for pfp in plots_fp}
            logs |= {str(wfp.name[:-4]): wandb.Image(str(wfp)) for wfp in weights_fp}
            logs |= {str(wfp.name[:-4]): wandb.Image(str(wfp)) for wfp in plots_l2_fp}
            if epoch == config['run']['n_epochs']:
                logs |= {str(vfp.name[:-4]): wandb.Video(str(vfp)) for vfp in videos_fp}
            wandb.log(logs | {"epoch": epoch, "trainer/global_step": epoch})


def train(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        l2: L2RBM,
        train_loader: DataLoader,
        test_loader: DataLoader,
        fabric: Fabric,
        l2_opt: Optimizer,
        l2_sched: Optional[ReduceLROnPlateau] = None
):
    """
    Train the model.
    :param config: Configuration dict
    :param feature_extractor: Feature extractor module
    :param lateral_network: Lateral network module
    :param l2: L2RBM module
    :param train_loader: Training dataloader
    :param test_loader: Testing dataloader
    :param fabric: The fabric instance.
    :param l2_opt: The optimizer for the L2-model.
    :param l2_sched: The lr scheduler for the L2-model.
    """
    start_epoch = config['run']['current_epoch']

    if config['logging']['wandb']['active'] or config['run']['plots']['enable']:
        single_eval_epoch(config, feature_extractor, lateral_network, l2, test_loader, 0)
        lateral_network.on_epoch_end()  # print logs

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch(config, feature_extractor, lateral_network, l2, train_loader, epoch + 1, fabric, l2_opt)
        single_eval_epoch(config, feature_extractor, lateral_network, l2, test_loader, epoch + 1)
        lateral_network.on_epoch_end()
        l2_logs = l2.on_epoch_end()
        if l2_sched is not None:
            l2_sched.step(l2_logs["l2/val/loss"])
        config['run']['current_epoch'] = epoch + 1


def setup_lateral_network(config, fabric) -> LateralNetwork:
    """
    Setup the model using lateral connections.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: Model using lateral connections.
    """
    return fabric.setup(LateralNetwork(config, fabric))


def setup_l2(config, fabric) -> L2RBM:
    """
    Setup the model L2.
    :param config: Configuration dict
    :param fabric: Fabric instance
    :return: L2 model.
    """
    return fabric.setup(L2RBM(config, fabric))


def main():
    """
    Run the model: Create modules, extract features from images and run the model leveraging lateral connections.
    """
    print_start("Starting python script 'main_lateral_connections.py'...",
                title="Training S1: Lateral Connections Toy Example")
    config = configure()
    fabric = setup_fabric(config)
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)
    l2 = setup_l2(config, fabric)
    l2_opt, l2_sched = l2.configure_optimizers()

    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, state = load_run(config, fabric)
        feature_extractor.load_state_dict(state['feature_extractor'])
        lateral_network.load_state_dict(state['lateral_network'])
        l2.load_state_dict(state['l2'])
        l2_opt.load_state_dict(state['l2_opt'])
        l2_sched.load_state_dict(state['l2_sched'])

    feature_extractor.eval()  # does not have to be trained
    if 'store_path' in config['run']['plots'] and config['run']['plots']['store_path'] is not None and \
            config['run']['plots']['store_path'] != 'None':
        fp = Path(config['run']['plots']['store_path'])
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)
    train(config, feature_extractor, lateral_network, l2, train_loader, test_loader, fabric, l2_opt, l2_sched)

    if 'store_state_path' in config['run'] and config['run']['store_state_path'] is not None and config['run'][
        'store_state_path'] != 'None':
        save_run(config, fabric,
                 components={'feature_extractor': feature_extractor, 'lateral_network': lateral_network, 'l2': l2,
                             'l2_opt': l2_opt, 'l2_sched': l2_sched.state_dict()})


if __name__ == '__main__':
    main()
