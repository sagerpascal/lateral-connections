import random
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as T


class StraightLine(Dataset):

    def __init__(self,
                 split: Literal['train', 'val', 'test'],
                 img_h: Optional[int] = 32,
                 img_w: Optional[int] = 32,
                 num_images: Optional[int] = 50,
                 num_aug_versions: Optional[int] = 0,
                 num_channels: Literal[1, 3] = 1,
                 aug_range: Optional[int] = 2,
                 vertical_horizontal_only: Optional[bool] = False,
                 fixed_lines_eval_test: Optional[bool] = False,
                 noise: Optional[float] = 0.,
                 aug_strategy: Optional[str] = 'random',
                 transform: Optional[Callable] = None):
        """
        Dataset that generates images with a straight line.
        :param img_h: Height of the images.
        :param img_w: Width of the images.
        :param num_images: Number of images to generate.
        :param num_aug_versions: Number of similar versions of each image to generate.
        :param num_channels: Number of channels of the images (1 for grayscale, 3 for RGB).
        :param vertical_horizontal_only: Whether to only generate vertical and horizontal lines.
        :param fixed_lines_eval_test: Whether to use fixed lines for the evaluation and test sets.
        :param noise: The amount of noise to add to the image (i.e. probability to set some pixels to 1).
        :param aug_strategy: The strategy to use for creating different image views. Can be 'random' or 'trajectory'.
        :param transform: Optional transform to be applied on a sample.
        """
        super().__init__()
        assert num_channels == 1 or num_channels == 3, "num_channels must be 1 or 3"
        assert num_aug_versions >= 0, "num_aug_versions must be >= 0"
        assert 0 <= noise <= 1, "noise must be between 0 and 1"

        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.num_images = num_images
        self.num_aug_versions = num_aug_versions
        self.num_channels = num_channels
        self.vertical_horizontal_only = vertical_horizontal_only
        self.fixed_lines_eval_test = fixed_lines_eval_test
        self.noise = noise
        self.aug_strategy = aug_strategy
        self.transform = transform
        self.aug_range = aug_range
        self.n_black_pixels = 5

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        """
        Returns the number of images in the dataset.
        :return: Number of images in the dataset.
        """
        return self.num_images

    def _get_random_line_coords(self, idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns the coordinates of a random straight line (create two random x,y coordinates).
        :param idx: Index of the image.
        :return: a Tuple of two Tuples of x,y coordinates.
        """
        # TODO: Delete this (always the same 2 lines)
        if idx % 2 == 0:
            return (2, 15), (30, 15)
        elif idx % 2 == 1:
            return (15, 2), (15, 30)

        # encourage longer lines
        x1 = random.randint(0, self.img_w // 2)
        y1 = random.randint(0, self.img_h // 2)
        x2 = random.randint(self.img_w // 2, self.img_w)
        y2 = random.randint(self.img_h // 2, self.img_h)
        if random.random() < 0.5:
            x1, x2 = x2, x1
        if random.random() < 0.5:
            y1, y2 = y2, y1

        if self.vertical_horizontal_only:
            if random.random() < 0.5:
                y1 = y2
            else:
                x1 = x2

        if self.fixed_lines_eval_test and (self.split == 'val' or self.split == 'test'):
            if idx == 0:
                x1, x2, y1, y2 = self.img_w // 2, self.img_w // 2, 5, self.img_h - 5
            elif idx == 1:
                x1, x2, y1, y2 = 5, self.img_w - 5, self.img_h // 2, self.img_h // 2
            elif idx == 2:
                x1, x2, y1, y2 = 5, self.img_w - 5, 5, self.img_h - 5
            elif idx == 3:
                x1, x2, y1, y2 = 5, self.img_w - 5, self.img_h - 5, 5

        return (x1, y1), (x2, y2)

    def _slightly_change_line_coords_random(self, coords: Tuple[Tuple[int, int], Tuple[int, int]]
                                            ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Slightly changes the coordinates of a straight line.
        :param coords: The coordinates of the straight line.
        :return: The new coordinates.
        """
        (x1, y1), (x2, y2) = coords

        if self.aug_range > 0:
            x1 += random.randint(-self.aug_range, self.aug_range)
            y1 += random.randint(-self.aug_range, self.aug_range)
            x2 += random.randint(-self.aug_range, self.aug_range)
            y2 += random.randint(-self.aug_range, self.aug_range)

        return (x1, y1), (x2, y2)

    def _change_line_coords_trajectory(self, coords: Tuple[Tuple[int, int], Tuple[int, int]], num_trajectory: int,
                                       ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Smoothly move the line from a starting point to a new point.
        :param coords: The coordinates of the initial straight line.
        :return: The new coordinates.
        """
        (x1, y1), (x2, y2) = coords
        x1_target = x1 + random.randint(-self.aug_range, self.aug_range)
        y1_target = y1 + random.randint(-self.aug_range, self.aug_range)
        x2_target = x2 + random.randint(-self.aug_range, self.aug_range)
        y2_target = y2 + random.randint(-self.aug_range, self.aug_range)

        coords_list = []
        for start, end in [(x1, x1_target), (y1, y1_target), (x2, x2_target), (y2, y2_target)]:
            step_size = abs(end - start) / num_trajectory
            trajectories = []
            point = start
            for i in range(num_trajectory):
                if start < end:
                    point += step_size
                else:
                    point -= step_size
                trajectories.append(int(round(point, 0)))
            coords_list.append(trajectories)

        result = []
        for j in range(num_trajectory):
            result.append(((coords_list[0][j], coords_list[1][j]), (coords_list[2][j], coords_list[3][j])))

        return result

    def _create_l_image(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Image:
        """
        Creates a black grayscale image with a random straight line in withe drawn on it.
        :param line_coords: The coordinates of the line to draw.
        :return: The image.
        """
        img = Image.new('L', (self.img_w, self.img_h), color=0)
        draw = ImageDraw.Draw(img)
        draw.line(line_coords, fill=255, width=1)
        return img

    def _create_rgb_image(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Image:
        """
        Creates a black RGB image with a random straight line in withe drawn on it.
        :param line_coords: The coordinates of the line to draw.
        :return: The image.
        """
        img = Image.new('RGB', (self.img_w, self.img_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line(line_coords, fill=(255, 255, 255), width=1)
        return img

    def _create_image(self, idx: int, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Image:
        """
        Creates either a RBG or a grayscale image with a random straight line in withe drawn on it.
        :param line_coords: The coordinates of the line to draw.
        :return: The image.
        """
        if self.num_channels == 1:
            img = self._create_l_image(line_coords)
        elif self.num_channels == 3:
            img = self._create_rgb_image(line_coords)
        else:
            raise ValueError('num_channels must be 1 or 3')

        # add a black pixel in the middle (discontinous line)
        if (self.split == 'val' or self.split == 'test') and (idx == 4 or idx == 5):
            img = np.array(img)
            line_center = (line_coords[0][0] + line_coords[1][0]) // 2, (line_coords[0][1] + line_coords[1][1]) // 2
            all_line_coords = np.argwhere(img > 128)
            center_point_idx = np.sum(np.abs(all_line_coords - np.array([line_center[1], line_center[0]]).reshape(1,2).repeat(all_line_coords.shape[0], axis=0)), axis=1).argmin()
            n_black = min(self.n_black_pixels, all_line_coords.shape[0] - 2)
            lower_idx = center_point_idx - n_black // 2
            upper_idx = center_point_idx + (n_black - (center_point_idx - lower_idx))
            idxs = np.array([list(all_line_coords[i]) for i in range(lower_idx, upper_idx)])
            img[idxs[:, 0], idxs[:, 1]] = 0
            img = Image.fromarray(img.astype(np.uint8))

        # add noise
        if self.noise > 0. or (self.split == 'val' or self.split == 'test') and (idx == 2 or idx == 3):
            noise = self.noise if self.noise > 0. else 0.005
            img = np.array(img)
            img = img + np.random.choice(2, img.shape, p=[1 - noise, noise]) * 255
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, idx: int):
        """
        Returns an image with a random straight line drawn on it.
        :param idx: Index of the image to return (has no effect)
        :return: The image
        """
        line_coords = self._get_random_line_coords(idx)

        if self.aug_strategy == 'random':
            aug_line_coords = [self._slightly_change_line_coords_random(line_coords) for _ in range(self.num_aug_versions)]

        elif self.aug_strategy == 'trajectory':
            aug_line_coords = self._change_line_coords_trajectory(line_coords, self.num_aug_versions)

        else:
            raise ValueError('aug_strategy must be "random" or "trajectory"')

        images = [self._create_image(idx, line_coords)]
        for aug_coords in aug_line_coords:
            images.append(self._create_image(idx, aug_coords))

        images = torch.stack(images, dim=0) if self.num_aug_versions > 0 else images[0]
        return images


def _plot_some_samples():
    """
    Plots some samples of the straight line dataset.
    """
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = StraightLine(split="test", img_h=32, img_w=32, num_images=10, num_aug_versions=9, num_channels=1,
                           transform=transform, vertical_horizontal_only=True, fixed_lines_eval_test=True, noise=0.005,
                           aug_strategy='trajectory', aug_range=15)

    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        img = dataset[i]
        for idx in range(img.shape[0]):
            j = i * 10 + idx
            axs[j // 10, j % 10].imshow(img[idx].squeeze(), vmin=0, vmax=1, cmap='gray', interpolation='none')
            axs[j // 10, j % 10].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _plot_some_samples()
