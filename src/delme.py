from utils import get_config
from data import loaders_from_config, undo_norm, undo_norm_from_conf, plot_images, show_grid
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import wandb
from tqdm import tqdm
from tools import torch_optim_from_conf, metrics_from_conf, loggers_from_conf, AverageMeter
import argparse

# TODO: https://github.com/Lightning-Universe/lightning-bolts




def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
                        # default=10,
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

    args = parser.parse_args()
    return args


args = parse_args()
config = get_config(args.config, args)
train_loader, _, test_loader = loaders_from_config(config)


# img, lbl = next(iter(train_loader))
# img = undo_norm_from_conf(img, config)

# plot_images(img, lbl)


class LitModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.l1 = nn.Linear(28 * 28, 10)
        self.loss_meters = {"train/loss": AverageMeter(), "val/loss": AverageMeter()}
        self.metrics = self.configure_metrics()
        self.configure_wandb_metrics()
        self.current_epoch_ = 0
        self.wandb_logs = {}

    # TODO: Instead of forward, an entire Model could be used
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log_step(loss, y_hat, y, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log_step(loss, y_hat, y, prefix="val")

    def log_step(self, loss, y_hat, y, prefix=""):
        for m_name, metric in self.loss_meters.items():
            if m_name.startswith(prefix):
                metric.add(loss)
        for m_name, metric in self.metrics.items():
            if m_name.startswith(prefix):
                metric(y_hat, y)

    def configure_optimizers(self):
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)

    def configure_metrics(self):
        train_metrics = {f"train/{k}": v for k, v in metrics_from_conf(self.conf).items()}
        val_metrics = {f"val/{k}": v for k, v in metrics_from_conf(self.conf).items()}
        return train_metrics | val_metrics

    def configure_wandb_metrics(self):
        for prefix in ["train", "val"]:
            wandb.define_metric(f'{prefix}/loss', summary='min')
            for m_name, in self.metrics.keys():
                wandb.define_metric(f'{prefix}/{m_name}', summary='max')

    def log_(self):
        logs = {}
        for prefix in ["train", "val"]:
            logs[f"{prefix}/loss"] = self.loss_meters[f"{prefix}/loss"].mean()
            self.loss_meters[f"{prefix}/loss"].reset()
            for m_name, m in self.metrics.items():
                logs[f"{prefix}/{m_name}"] = m.mean()
                m.reset()
        self.logs.update(logs)
        self.log(self.logs, step=self.current_epoch_)
        self.logs = {}

    def on_epoch_end(self):
        self.log_()
        self.current_epoch_ += 1


# loggers = loggers_from_conf(config)
loggers = []
fabric = Fabric(accelerator="auto", devices=1, loggers=loggers)
print(fabric)
fabric.launch()
fabric.seed_everything(1)
model = LitModel(config)
optimizer = model.configure_optimizers()
model, optimizer = fabric.setup(model, optimizer)
train_dataloader = fabric.setup_dataloaders(train_loader)
test_dataloader = fabric.setup_dataloaders(test_loader)

for epoch in range(config['run']['n_epochs']):
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader),
                         total=len(train_dataloader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch + 1}/{config['run']['n_epochs']}"):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader),
                             total=len(test_dataloader),
                             colour="GREEN",
                             desc=f"Validate Epoch {epoch + 1}/{config['run']['n_epochs']}"):
            model.validation_step(batch, i)

    model.on_epoch_end()
