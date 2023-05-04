import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from data import fig_to_img, plot_images, undo_norm_from_conf

_path_t = Union[str, os.PathLike, Path]


class VisualizeImageCallback:

    def __init__(
            self,
            conf: Dict[str, Optional[Any]],
            plot: bool = True,
            store_fp: Optional[_path_t] = None,
            return_images: bool = True
    ):
        """
        Callback to visualize images.
        :param conf: Config dict
        :param plot: Plot images (matplotlib.pyplot.show()
        :param store_fp: File path to store images (None for not storing images, only plotting)
        :param return_images: Return a dict of images
        """
        self.conf = conf
        self.plot = plot
        self.store_fp = store_fp
        self.return_images = return_images

    def on_log_samples(
            self,
            samples: torch.Tensor,
            preds: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None,
            epoch: Optional[int] = None
    ) -> Optional:
        """
        Store the samples as images
        :param samples: Samples
        :param preds: Predicted values
        :param targets: Target values
        :param epoch: Current epoch
        :return:
        """
        samples_no_norm = undo_norm_from_conf(samples.detach(), self.conf)
        titles = [f"Pred: {str(p.item())}, Target: {str(t.item())}" for p, t in
                  zip(preds, targets)] if preds is not None and targets is not None else None
        fp = f"{self.store_fp[:-4]}_ep_{epoch}{self.store_fp[-4:]}" if self.store_fp is not None else self.store_fp
        fig = plot_images(samples_no_norm, titles=titles, show_plot=self.plot, fig_fp=fp)
        if self.return_images:
            return fig_to_img(fig)
