from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data import plot_images
from models.lightning_modules.lightning_base import BaseLitModule
from tools import torch_optim_from_conf


class RBM(nn.Module):
    def __init__(self, n_visible=4 * 32 * 32, n_hidden=16, k=1):
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

    def forward(self, v, prototype):
        """
        Compute the real and generated examples.
        """
        v = v.flatten(1)
        prototype = prototype.flatten(1)
        h, ph = self.visible_to_hidden(v)
        for _ in range(self.k):
            v_gibb, pv = self.hidden_to_visible(h)
            h, ph = self.visible_to_hidden(v_gibb)
        return v, v_gibb, torch.cat([h, ph], dim=1)

class RBM2(nn.Module):
    def __init__(self, n_visible=4 * 32 * 32, n_hidden=1 * 32 * 32, k=1):
        """Create a RBM."""
        super(RBM2, self).__init__()
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

    def free_energy_visible(self, v):
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

    def free_energy_hidden(self, h):
        h_term = torch.matmul(h, self.h.t())
        w_x_h = F.linear(h, self.W.t(), self.v)
        v_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-v_term - h_term)

    def forward(self, z, prototype):
        z = z.flatten(1)  # Visible Target
        pt = prototype.flatten(1)  # Hidden Target
        pt2, ppt2 = self.visible_to_hidden(z)  # Visible prediction
        z2, pz2 = self.hidden_to_visible(pt2) # Hidden prediction

        return pt, pt2, z, z2, ppt2

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
        self.model = self.configure_model()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor.
        :param y: Target tensor / prototype tensor.
        """
        return self.model(x, y)

    def step(self, x: Tensor, y: Tensor, batch_idx: int, log_prefix: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pt, pt2, z, z2, h = self.forward(x, y)
        loss = self.model.free_energy_hidden(pt) - self.model.free_energy_hidden(pt2)
        loss += self.model.free_energy_visible(z) - self.model.free_energy_visible(z2)
        self.log_step(processed_values={"loss": loss}, prefix=log_prefix)
        return pt, pt2, h, loss

    def train_step(self, x: Tensor, y: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.step(x, y, batch_idx, "l2/train")

    def eval_step(self, x: Tensor, y: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.step(x, y, batch_idx, "l2/val")

    def configure_model(self) -> nn.Module:
        """
        Configure (create instance) the model.
        :return: A torch model.
        """
        return RBM2()

    def configure_optimizers(self) -> Tuple[Optimizer, Optional[LRScheduler]]:
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
        img_list = torch.stack([i.squeeze() for i in img_list])
        img_list = (img_list - img_list.min()) / (img_list.max() - img_list.min() + 1e-9)
        img_list = [img_list[i] for i in range(img_list.shape[0])]
        return img_list

    def plot_samples(self, img, activations_l2, show_plot):
        print(len(activations_l2))
        print(activations_l2[0].shape)
        plt_images, plt_titles = [], []
        for img_i, act_i in zip(img, activations_l2):
            for batch_idx in range(img_i.shape[0]):
                for view_idx in range(img_i.shape[1]):
                    plt_images.append(img_i[batch_idx, view_idx])
                    plt_titles.append(f"B={batch_idx} V={view_idx}")
                    plt_images.append(act_i[batch_idx, 0, view_idx].reshape(-1, 32, 32))
                    # plt_images.append(F.pad(act_i[batch_idx, view_idx], (0, 0, 13, 14), "constant", 0.5))
                    plt_titles.append(f"B={batch_idx} V={view_idx} L2")

        fig_fp = Path(self.conf['run']['plots'].get('store_path', None))
        if fig_fp is not None:
            fig_fp = fig_fp / f"l2.png"
        plt_images = self._normalize_image_list(plt_images)
        plot_images(images=plt_images, titles=plt_titles, max_cols=2, plot_colorbar=True,
                    vmin=0, vmax=1, fig_fp=fig_fp, show_plot=show_plot)
