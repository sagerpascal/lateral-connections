import matplotlib.pyplot as plt
from typing import List, Any, Optional, Union, Dict
import PIL
import numpy as np
import torch
import torchvision.transforms as T


def undo_norm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Undo normalization.
    :param img: Image.
    :param mean: Mean.
    :param std: Standard deviation.
    """
    if len(img.shape) == 4:
        assert img.shape[1] == 3 or img.shape[1] == 1, "Image must be Tensor of shape 3xMxN or 1xMxN."
        assert img.shape[1] == mean.shape[0] and img.shape[1] == std.shape[0], "Mean and std must have same number of channels as image."
        return img * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    elif len(img.shape) == 3:
        assert img.shape[0] == 3 or img.shape[0] == 1, "Image must be Tensor of shape 3xMxN or 1xMxN."
        assert img.shape[0] == mean.shape[0] and img.shape[0] == std.shape[0], "Mean and std must have same number of channels as image."
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


def plot_images(images: List[Any], labels: Optional[List[Any]] = None):
    """
    Plot images.
    :param images: A list of images.
    :param labels: A list of labels.
    """
    if isinstance(images, torch.Tensor) and len(images.shape) == 4:
        images = [images[i, ...] for i in range(images.shape[0])]
    if isinstance(labels, torch.Tensor) and labels.shape[0] > 1:
        labels = [labels[i] for i in range(labels.shape[0])]
    if not isinstance(images, list):
        images = [images]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    cols = min(5, len(images))
    rows = len(images) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    transform = T.ToPILImage()

    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i // cols, i % cols] if cols > 1 and rows > 1 else axes[i % cols] if cols > 1 else axes
        if isinstance(img, torch.Tensor):
            img = transform(img)
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)

        if img.shape[-1] == 1:
            ax.imshow(img, cmap='binary')
        else:
            ax.imshow(img)
        if lbl is not None:
            ax.set_title(str(lbl.item()))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()
    plt.tight_layout()


def show_grid(img_grid: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Show a grid of images.
    :param img_grid: Image grid.
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
    plt.show()
    plt.tight_layout()
