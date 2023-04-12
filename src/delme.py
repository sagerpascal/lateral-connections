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
from tools import torch_optim_from_conf, metrics_from_conf, loggers_from_conf

# TODO: https://github.com/Lightning-Universe/lightning-bolts

config = get_config("base_config")
train_loader, _, test_loader = loaders_from_config(config)


# img, lbl = next(iter(train_loader))
# img = undo_norm_from_conf(img, config)

# plot_images(img, lbl)


class LitModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.l1 = nn.Linear(28 * 28, 10)
        self.metrics = self.configure_metrics()
        self.configure_wandb_metrics()
        self.current_epoch_ = 0

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
        self.log_step(loss, y_hat, y, prefix="val")  # TODO: Implement log at epoch end

    def log_step(self, loss, y_hat, y, prefix=""):
        logs = {f"{prefix}/loss": loss, "epoch": self.current_epoch_}
        for m_name, m in self.metrics.items():
            logs[f"{prefix}/{m_name}"] = m(y_hat, y)
        self.log_dict(logs)  # TODO: Implement log at epoch end

    def configure_optimizers(self):
        return torch_optim_from_conf(self.parameters(), 'opt1', self.conf)

    def configure_metrics(self):
        return metrics_from_conf(self.conf)

    def configure_wandb_metrics(self):
        for prefix in ["train", "val"]:
            wandb.define_metric(f'{prefix}/loss', summary='min')
            for m_name, in self.metrics.keys():
                wandb.define_metric(f'{prefix}/{m_name}', summary='max')

    def on_epoch_end(self):
        self.current_epoch_ += 1


# loggers = loggers_from_conf(config)
loggers = []
fabric = Fabric(accelerator="auto", devices=1, loggers=loggers)
print(fabric)
fabric.launch()
fabric.seed_everything(1)
model = LitModel(config)
optimizer = model.configure_optimizers()
num_epochs = 10
model, optimizer = fabric.setup(model, optimizer)
train_dataloader = fabric.setup_dataloaders(train_loader)
test_dataloader = fabric.setup_dataloaders(test_loader)

for epoch in range(num_epochs):
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader),
                         total=len(train_dataloader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        loss = model.training_step(batch, i)
        fabric.backward(loss)
        optimizer.step()

    model.eval()
    for i, batch in tqdm(enumerate(test_dataloader),
                         total=len(test_dataloader),
                         colour="GREEN",
                         desc=f"Validate Epoch {epoch + 1}/{num_epochs}"):
        model.validation_step(batch, i)

    model.on_epoch_end()
