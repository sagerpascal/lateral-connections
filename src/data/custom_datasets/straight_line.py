import random
from typing import Callable, Literal, Optional, Tuple

from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class StraightLine(Dataset):

    def __init__(self,
                 split: Literal['train', 'val', 'test'],
                 img_h: Optional[int] = 32,
                 img_w: Optional[int] = 32,
                 num_images: Optional[int] = 10_000,
                 num_channels: Literal[1, 3] = 1,
                 transform: Optional[Callable] = None):
        """
        Dataset that generates images with a straight line.
        :param img_h: Height of the images.
        :param img_w: Width of the images.
        :param num_images: Number of images to generate.
        :param num_channels: Number of channels of the images (1 for grayscale, 3 for RGB).
        :param transform: Optional transform to be applied on a sample.
        """
        super().__init__()
        assert num_channels == 1 or num_channels == 3, "num_channels must be 1 or 3"

        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.num_images = num_images
        self.num_channels = num_channels
        self.transform = transform

    def __len__(self):
        """
        Returns the number of images in the dataset.
        :return: Number of images in the dataset.
        """
        return self.num_images

    def _get_random_line_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns the coordinates of a random straight line (create two random x,y coordinates).
        :return: a Tuple of two Tuples of x,y coordinates.
        """
        x1 = random.randint(0, self.img_w)
        y1 = random.randint(0, self.img_h)
        x2 = random.randint(0, self.img_w)
        y2 = random.randint(0, self.img_h)

        if self.split == 'train':
            # vertical or horizontal line
            if random.random() < 0.5:
                y1 = y2
            else:
                x1 = x2
        return (x1, y1), (x2, y2)

    def _create_l_image(self):
        """
        Creates a black grayscale image with a random straight line in withe drawn on it.
        :return: The image.
        """
        img = Image.new('L', (self.img_w, self.img_h), color=0)
        draw = ImageDraw.Draw(img)
        draw.line(self._get_random_line_coords(), fill=255, width=1)
        return img

    def _create_rgb_image(self):
        """
        Creates a black RGB image with a random straight line in withe drawn on it.
        :return: The image.
        """
        img = Image.new('RGB', (self.img_w, self.img_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line(self._get_random_line_coords(), fill=(255, 255, 255), width=1)
        return img

    def __getitem__(self, idx):
        """
        Returns an image with a random straight line drawn on it.
        :param idx: Index of the image to return (has no effect)
        :return: The image
        """
        if self.num_channels == 1:
            img = self._create_l_image()
        elif self.num_channels == 3:
            img = self._create_rgb_image()

        if self.transform:
            img = self.transform(img)

        return img


def _plot_some_samples():
    """
    Plots some samples of the straight line dataset.
    """
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = StraightLine(split="test", img_h=32, img_w=32, num_images=10, num_channels=1, transform=transform)

    fig, axs = plt.subplots(2, 5, figsize=(10, 3))
    for i in range(10):
        img = dataset[i]
        axs[i // 5, i % 5].imshow(img.squeeze(), vmin=0, vmax=1, cmap='gray')
        axs[i // 5, i % 5].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _plot_some_samples()
