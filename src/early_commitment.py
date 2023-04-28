"""

Script to investigate early commitment of models.

"""
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader

from utils import get_config, print_warn, print_start
from data import loaders_from_config
from lightning.fabric import Fabric
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from tools import torch_optim_from_conf, loggers_from_conf
import argparse
from typing import Optional, Dict, Any, Callable
from models.lightning_modules.lightning_base import BaseLitModule
from models.classification import BlockCallbackResNet18
from tools.callbacks import LogTensorsCallback
from pathlib import Path
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

PICKLE_FP = Path("../tmp/early_commitment.pickle")


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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor.
        :param y: Optional labels (needed for logging during validation).
        """
        return self.model(x, y)

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
        y_hat = self.forward(x, y)
        loss = F.cross_entropy(y_hat, y)
        self.log_step(loss, y_hat, y, prefix="val")
        return loss

    def configure_model(self) -> nn.Module:
        """
        Configure (create instance) the model.
        :return: A torch model.
        """
        return BlockCallbackResNet18(self.conf)

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
    log_tensor_callback = LogTensorsCallback(PICKLE_FP)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=[log_tensor_callback])
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

    return fabric, model, optimizer, train_loader, test_loader, log_tensor_callback


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


def train():
    """
    Run the model and store the models activations in the last epoch.
    """
    print_start("Starting python script 'early_commitment.py'...", title="Launching Early Commitment Analysis")
    args = parse_args()
    config = get_config(args.config, args)
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")

    fabric, model, optimizer, train_dataloader, test_dataloader, log_tensor_callback = setup_components(config)

    for epoch in range(config['run']['n_epochs']):
       single_train_epoch(config, fabric, model, optimizer, train_dataloader, epoch)
       if epoch + 1 == config['run']['n_epochs']:
           model.model.register_after_block_callback("tc_1", log_tensor_callback)
       single_eval_epoch(config, model, test_dataloader, epoch)
       model.on_epoch_end()
    fabric.call("on_train_end")

    return fabric


def analyze_activations(fabric: Fabric):
    """
    Analyze the activations of the model (to identify early commitment).
    :param fabric: Fabric instance.

    """
    print_start("Starting analysis in script 'early_commitment.py'...", title="Analysing Early Commitment")
    kmeans = KMeans(n_clusters=300, mode='euclidean', verbose=1)
    mi_scores = {}
    for fp in PICKLE_FP.parent.glob("early_commitment*.pickle"):
        layer = fp.stem.split("_")[-1]
        with open(str(fp), 'rb') as handle:
            activations = pickle.load(handle)
            X, Y = [], []
            for y, x in activations.items():
                X = X + x
                Y = Y + ([y] * len(x))
            X = torch.from_numpy(np.vstack(X)).to(fabric.device)
            Y = np.stack(Y)
            print(X.shape, Y.shape)
            labels = kmeans.fit_predict(X)
            mi = normalized_mutual_info_score(labels.detach().cpu().numpy(), Y)
            mi_scores[layer] = mi
            print(mi)
    print(mi_scores)

    mi_scores = dict(sorted(mi_scores.items()))

    x, y = [], []
    for layer, mi in mi_scores.items():
        x.append(layer)
        y.append(mi)

    plt.plot(x, y)
    plt.xlabel("ResNet Block")
    plt.ylabel("Mutual Information")
    plt.tight_layout()
    plt.show()

    print("The plot shows a linear increase, indicating that early layers do not commit to an output class."
          "However, the Gestalt principle of similarity suggests that the early layers should commit to a class."
          "I the human brain, we look at some tiny features an can already decide that these fetures most likely"
          "belong to a very limited subset of classes.")


if __name__ == '__main__':
    fabric = train()
    analyze_activations(fabric)
