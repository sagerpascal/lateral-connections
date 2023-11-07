from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import plot_images
from data.custom_datasets.arc import get_colormap
from models.lightning_modules.lightning_base import BaseLitModule
from tools import torch_optim_from_conf


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k):
        """Create a RBM."""
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_visible))
        self.h = nn.Parameter(torch.randn(1, n_hidden))
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.k = k

    def visible_to_hidden(self, v):
        """
         sampling a hidden variable given a visible variable.
        """
        p = torch.sigmoid(F.linear(v, self.W, self.h))
        return p.bernoulli(), p

    def hidden_to_visible(self, h):
        """
        Conditional sampling a visible variable given a hidden variable.
        """
        p = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        return p.bernoulli(), p

    def free_energy(self, v):
        """
        Free energy function.

        .. math::
            \begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
            \end{align}
        """
        v_term = torch.matmul(v, self.v.t())
        w_x_h = F.linear(v, self.W, self.h)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, v):
        """
        Compute the real and generated examples.
        """
        v = v.flatten(1)
        h, ph = self.visible_to_hidden(v)
        for _ in range(self.k):
            v_gibb, pv = self.hidden_to_visible(h)
            h, ph = self.visible_to_hidden(v_gibb)
        return v, v_gibb, torch.cat([h, ph], dim=1).view(-1, 1, 1, 32).repeat(1, 1, 32, 1)


class L2RBM(BaseLitModule):
    """
    Lightning Module the L2 layer implemented in a RBM-Fashion
    """

    def __init__(self, conf: Dict[str, Optional[Any]], fabric: Fabric):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param fabric: Fabric instance.
        """
        super().__init__(conf, fabric, logging_prefixes=["l2/train", "l2/val"])
        self.conf = conf
        self.model = self.configure_model(conf)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor.
        """
        return self.model(x)

    def step(self, x: Tensor, log_prefix: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # pt, pt2, h = self.forward(x, y)
        # loss = self.model.free_energy_hidden(pt) - self.model.free_energy_hidden(pt2)
        v, v_gibb, h = self.forward(x)
        loss = self.model.free_energy(v) - self.model.free_energy(v_gibb)
        v = v.reshape(-1, self.conf['lateral_model']['channels'] * self.conf['n_alternative_cells'], self.conf['dataset']['img_width'], self.conf['dataset']['img_height'])
        v_gibb = v_gibb.reshape(-1, self.conf['lateral_model']['channels'] * self.conf['n_alternative_cells'], self.conf['dataset']['img_width'], self.conf['dataset']['img_height'])
        self.log_step(processed_values={"loss": loss}, metric_pairs=[(v, v_gibb)], prefix=log_prefix)
        return v, v_gibb, h, loss

    def train_step(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.step(x, "l2/train")

    def eval_step(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.step(x, "l2/val")

    def configure_model(self, conf: Dict[str, Optional[Any]]) -> nn.Module:
        """
        Configure (create instance) the model.
        :param conf: Configuration dictionary.
        :return: A torch model.
        """
        n_visible = conf['lateral_model']['channels'] * conf['dataset']['img_width'] * conf['dataset']['img_height'] * conf['n_alternative_cells']
        return RBM(n_visible=n_visible, n_hidden=conf['l2']['n_hidden'], k=conf['l2']['k'])

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[ReduceLROnPlateau]]:
        """
        Configure (create instance) the optimizer.
        :return: A torch optimizer.
        """
        return torch_optim_from_conf(self.parameters(), 'l2_opt', self.conf)

    def on_epoch_end(self):
        """
        Callback at the end of an epoch.
        """
        return super().on_epoch_end()

    def _normalize_image_list(self, img_list):
        try:
            img_list = torch.stack([i.squeeze() for i in img_list])
            img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min() + 1e-9)
            img_list = [img_list[i] for i in range(img_list.shape[0])]
        except RuntimeError:
            min_ = min([torch.min(i) for i in img_list])
            max_ = max([torch.max(i) for i in img_list])
            img_list = [(i - min_) / (max_ - min_ + 1e-9) for i in img_list]

        return img_list

    def plot_samples(self, img, activations_l2, show_plot):
        fig_fps = []
        for img_i, act_i in zip(img, activations_l2):
            for batch_idx in range(img_i.shape[0]):
                plt_images, plt_titles = [], []
                for view_idx in range(act_i.shape[2]):
                    plt_images.append(torch.argmax(img_i[batch_idx], dim=0))
                    plt_titles.append(f"B={batch_idx}")
                    for feature_idx in range(act_i.shape[3]):
                        plt_images.append(act_i[batch_idx, :, view_idx, feature_idx])
                        # plt_images.append(F.pad(act_i[batch_idx, view_idx], (0, 0, 13, 14), "constant", 0.5))
                        plt_titles.append(f"V={view_idx} F={feature_idx} L2")

                fig_fp = self.conf['run']['plots'].get('store_path', None)
                if fig_fp is not None:
                    fig_fp = Path(fig_fp) / f"l2_B={batch_idx}.png"
                plt_images = self._normalize_image_list(plt_images)
                plot_images(images=plt_images, titles=plt_titles, max_cols=act_i.shape[3] + 1, plot_colorbar=True,
                            vmin=0, vmax=9, fig_fp=fig_fp, show_plot=show_plot, interpolation='none', cmap=get_colormap())
                fig_fps.append(fig_fp)
        return fig_fps
