from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as T

figsize = 32
color = 255


def _get_pdraw() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    """
    Get a PIL image and a PIL draw object.
    :return: The PIL image and the PIL draw object.
    """
    img = Image.new('L', (figsize, figsize))
    return img, ImageDraw.Draw(img)


def create_0() -> Image.Image:
    """
    Create a 0.
    :return: Image of a 0.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (2, 30), fill=color)  # left line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    return img


def create_1() -> Image.Image:
    """
    Create a 1.
    :return: Image of a 1.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (16, 2), fill=color)  # half top line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((16, 2) + (16, 30), fill=color)  # center vertical line
    return img


def create_2() -> Image.Image:
    """
    Create a 2.
    :return: Image of a 2.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((30, 2) + (30, 16), fill=color)  # half top right line
    draw.line((2, 16) + (2, 30), fill=color)  # half bottom left line
    return img


def create_3() -> Image.Image:
    """
    Create a 3.
    :return: Image of a 3.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    return img


def create_4() -> Image.Image:
    """
    Create a 4.
    :return: Image of a 4.
    """
    img, draw = _get_pdraw()
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    draw.line((2, 2) + (2, 16), fill=color)  # half top left line
    return img


def create_5() -> Image.Image:
    """
    Create a 5.
    :return: Image of a 5.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((2, 2) + (2, 16), fill=color)  # half top left line
    draw.line((30, 16) + (30, 30), fill=color)  # half bottom right line
    return img


def create_6() -> Image.Image:
    """
    Create a 6.
    :return: Image of a 6.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((2, 2) + (2, 30), fill=color)  # left line
    draw.line((30, 16) + (30, 30), fill=color)  # half bottom right line
    return img


def create_7() -> Image.Image:
    """
    Create a 7.
    :return: Image of a 7.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    return img


def create_8() -> Image.Image:
    """
    Create a 8.
    :return: Image of a 8.
    """
    img, draw = _get_pdraw()
    draw.line((2, 2) + (2, 30), fill=color)  # left line
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    return img


def create_9() -> Image.Image:
    """
    Create a 9.
    :return: Image of a 9.
    """
    img, draw = _get_pdraw()
    draw.line((2, 30) + (30, 30), fill=color)  # bottom line
    draw.line((30, 2) + (30, 30), fill=color)  # right line
    draw.line((2, 2) + (30, 2), fill=color)  # top line
    draw.line((2, 16) + (30, 16), fill=color)  # center horizontal line
    draw.line((2, 2) + (2, 16), fill=color)  # half top left line
    return img


def create_noise() -> Image.Image:
    """
    Create a random image.
    :return: Image with random noise.
    """
    return Image.fromarray(np.random.randint(low=0, high=2, size=(figsize, figsize), dtype=np.uint8) * 255)


class EightBitDataset(Dataset):
    """
    Dataset with 8-bit numbers.
    """

    def __init__(self,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 samples_per_class: int = 1000,
                 include_noise: bool = False):
        """
        Initialize the dataset.
        :param transform: Transform to apply to the images.
        :param target_transform: Transform to apply to the targets.
        :param samples_per_class: Number of samples per class.
        :param include_noise: Add a class with random noise.
        """
        super(EightBitDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.include_noise = include_noise
        self.data = self._get_data() * samples_per_class

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def _get_data(self) -> list[Tuple[int, Any]]:
        """
        Get the data as list of tuples containing the label and the corresponding image.
        :return: Data as a list.
        """
        data = [
            (0, create_0()),
            (1, create_1()),
            (2, create_2()),
            (3, create_3()),
            (4, create_4()),
            (5, create_5()),
            (6, create_6()),
            (7, create_7()),
            (8, create_8()),
            (9, create_9()),
        ]
        if self.include_noise:
            data.append((10, create_noise))

        return data

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target, img = self.data[index]

        if callable(img):
            img = img()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    dataset = EightBitDataset(samples_per_class=1, include_noise=True)
    fig, axs = plt.subplots(nrows=4, ncols=3)
    for i in range(len(dataset)):
        axs[i // 3, i % 3].imshow(dataset[i][0], cmap='gray')
        axs[i // 3, i % 3].grid()
        axs[i // 3, i % 3].axis("off")
    plt.tight_layout()
    plt.show()
