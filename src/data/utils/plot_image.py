from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

def undo_norm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Undo normalization.
    :param img: Image.
    :param mean: Mean.
    :param std: Standard deviation.
    """
    if img.device != mean.device:
        mean = mean.to(img.device)
    if img.device != std.device:
        std = std.to(img.device)
    if len(img.shape) == 4:
        assert img.shape[1] == 3 or img.shape[1] == 1, "Image must be Tensor of shape 3xMxN or 1xMxN."
        assert img.shape[1] == mean.shape[0] and img.shape[1] == std.shape[0], "Mean and std must have same number of" \
                                                                               " channels as image."
        return img * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    elif len(img.shape) == 3:
        assert img.shape[0] == 3 or img.shape[0] == 1, "Image must be Tensor of shape 3xMxN or 1xMxN."
        assert img.shape[0] == mean.shape[0] and img.shape[0] == std.shape[0], "Mean and std must have same number of" \
                                                                               " channels as image."
        return img * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    else:
        raise ValueError("Image must be Tensor of shape Bx3xMxN, Bx1xMxN, 3xMxN or 1xMxN..")


def undo_norm_from_conf(img: torch.Tensor, config: Dict[str, Any]):
    """
    Undo normalization.
    :param img: Image.
    :param config: Configuration.
    """
    return undo_norm(img, torch.Tensor([config['dataset']['mean']]), torch.Tensor(config['dataset']['std']))


def plot_images(
        images: List[Any],
        masks: Optional[List[Any]] = None,
        titles: Optional[List[Any]] = None,
        suptitle: Optional[str] = None,
        show_plot: bool = True,
        fig_fp: Optional[str] = None,
        max_cols: Optional[int] = 5,
        vmin: Optional[Union[float, int]] = None,
        vmax: Optional[Union[float, int]] = None,
        mask_vmin: Optional[Union[float, int]] = None,
        mask_vmax: Optional[Union[float, int]] = None,
        plot_colorbar: Optional[bool] = False,
        cmap: Optional[Union[List[str] | str]] = None,
        interpolation: Optional[Union[List[str] | str]] = 'none',
) -> plt.Figure:
    """
    Plot images.
    :param images: A list of images.
    :param masks: A list of masks that are overlaid on the images.
    :param titles: A list of titles or labels.
    :param suptitle: Overall title.
    :param show_plot: Show plot.
    :param fig_fp: File path to save figure.
    :param max_cols: Maximum number of columns.
    :param vmin: Minimum value of the image.
    :param vmax: Maximum value of the image.
    :param mask_vmin: Minimum value of the mask.
    :param mask_vmax: Maximum value of the mask.
    :param plot_colorbar: Plot colorbar.
    :param cmap: Colormap, either a `str` or a `List[str]`.
    :param interpolation: Interpolation, either a `str` or a `List[str]`.
    :return: matplotlib Figure.
    """
    if isinstance(images, torch.Tensor) and len(images.shape) == 4:
        images = [images[i, ...] for i in range(images.shape[0])]
    if not isinstance(images, list):
        images = [images]

    if masks is not None:
        if isinstance(masks, torch.Tensor) and len(masks.shape) == 4:
            masks = [masks[i, ...] for i in range(masks.shape[0])]
        if not isinstance(masks, list):
            masks = [masks]
        assert len(images) == len(masks), "Number of images and masks must be equal."

    if titles is not None:
        if isinstance(titles, torch.Tensor) and titles.shape[0] > 1:
            titles = [titles[i] for i in range(titles.shape[0])]
        if not isinstance(titles, list):
            titles = [titles]

    cmaps = cmap if isinstance(cmap, list) else [cmap] * len(images)
    interpolations = interpolation if isinstance(interpolation, list) else [interpolation] * len(images)

    cols = min(max_cols, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    transform = T.ToPILImage()

    for i, (img, lbl, cmap_, interpolation_) in enumerate(zip(images, titles, cmaps, interpolations)):
        ax = axes[i // cols, i % cols] if cols > 1 and rows > 1 else axes[i % cols] if cols > 1 else axes
        if isinstance(img, torch.Tensor):
            img = img.detach().squeeze().cpu().numpy()  # transform(img.squeeze())
        if isinstance(img, Image):
            img = np.array(img)

        if masks is not None:
            mask = masks[i]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().squeeze().cpu().numpy()   # transform(mask)
            if isinstance(mask, Image):
                mask = np.array(mask)

        if (vmax is not None and img.max() > vmax) or (vmin is not None and img.min() < vmin):
            raise ValueError("Image values must be between vmin and vmax.")

        if img.shape[-1] == 1 or len(img.shape) == 2:
            cmap_ = 'gray' if cmap_ is None else cmap_
            im = ax.imshow(img, cmap=cmap_, vmin=vmin, vmax=vmax, interpolation=interpolation_)
        else:
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap_, interpolation=interpolation_)

        if plot_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        if masks is not None and mask is not None:
            ax.imshow(mask, alpha=0.6, cmap='jet', interpolation=interpolation_, vmin=mask_vmin, vmax=mask_vmax)

        if lbl is not None:
            lbl = str(lbl.item()) if isinstance(lbl, torch.Tensor) else lbl
            ax.set_title(lbl)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if suptitle is not None:
        fig.suptitle(suptitle)
    if fig_fp is not None:
        plt.savefig(fig_fp)
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close()
    return fig


def show_grid(
        img_grid: Union[torch.Tensor, List[torch.Tensor]],
        show_plot: bool = True,
        fig_fp: Optional[str] = None
) -> plt.Figure:
    """
    Show a grid of images.
    :param img_grid: Image grid.
    :param show_plot: Show plot.
    :param fig_fp: File path to save figure.
    :return: Figure.
    """
    if not isinstance(img_grid, list):
        img_grid = [img_grid]
    transform = T.ToPILImage()
    fig, axs = plt.subplots(ncols=len(img_grid), squeeze=False, figsize=(15, 10))
    for i, img_grid in enumerate(img_grid):
        img_grid = img_grid.detach()
        img_grid = transform(img_grid)
        axs[0, i].imshow(np.asarray(img_grid))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if fig_fp is not None:
        plt.savefig(fig_fp)
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def fig_to_img(fig: plt.Figure) -> Image:
    """
    Convert a matplotlib figure to a PIL image.
    :param fig: matplotlib figure.
    :return: PIL image.
    """
    assert isinstance(fig, plt.Figure), "Input must be a matplotlib figure."
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return Image.fromarray(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 'RGB')
