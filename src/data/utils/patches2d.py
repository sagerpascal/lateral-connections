"""
Utility class to convert images to patches and vice versa.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Patches2D:
    """
    Convert image into non-overlapping patches and vice versa.
    """

    def __init__(
            self,
            image_height: int,
            image_width: int,
            patch_height: int,
            patch_width: int,
            padding_height: int,
            padding_width: int,
            padding_mode: Optional[str] = "constant",
            padding_value: Optional[int] = 0
    ):
        """
        Constructor.
        :param image_height: Height of the image.
        :param image_width: Width of the image.
        :param patch_height: Height of the patch.
        :param patch_width: Width of the patch.
        :param padding_height: Height of the padding.
        :param padding_width: Width of the padding.
        :param padding_mode: Padding mode ('constant', 'reflect', 'replicate' or 'circular'. Default: 'constant').
        :param padding_value: fill value for 'constant' padding. Default: 0
        """
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.padding_height = padding_height
        self.padding_width = padding_width
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        assert self.padding_height % 2 == 0 or self.padding_height < 0, 'Padding height must be even and >0.'
        assert self.padding_width % 2 == 0 or self.patch_width < 0, 'Padding width must be even and >0.'
        assert self.image_height + self.padding_height >= self.patch_height, \
            'Image height + padding height must be greater than patch height.'
        assert self.image_width + self.padding_width >= self.patch_width, \
            'Image width + padding width must be greater than patch width.'
        assert (self.image_height + self.padding_height) % self.patch_height == 0, \
            'Image height + padding height must be divisible by patch height.'
        assert (self.image_width + self.padding_width) % self.patch_width == 0, \
            'Image width + padding width must be divisible by patch width.'

    def image_to_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches. The second channel contains the patches. The patches are sorted from top left to
        bottom right (like a CNN filter).

        Note: If overlapping padding is needed, use different value for the second argument in the unfold argument.
        However, it is unclear how to convert the patches back to the image.

        :param image: Image tensor (BxCxHxW).
        :return: Patches tensor (BxNPxCxPHxPW) where NP is nuber of patches, PH patch height and PW patch width.
        """
        if self.padding_height > 0 or self.padding_width > 0:
            image = self.pad_image(image)
        n_channels = image.shape[1]
        return image \
            .unfold(2, self.patch_height, self.patch_height) \
            .unfold(3, self.patch_width, self.patch_width) \
            .permute(0, 2, 3, 1, 4, 5) \
            .reshape(image.shape[0], -1, n_channels, self.patch_height, self.patch_width)

    def pad_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Pad image.
        :param image: Image tensor.
        :return: Padded image tensor.
        """
        return F.pad(
            image,
            pad=(self.padding_width // 2, self.padding_width // 2, self.padding_height // 2, self.padding_height // 2),
            mode=self.padding_mode,
            value=self.padding_value
        )

    def _get_n_patches_per_side(self) -> Tuple[int, int]:
        """
        Get number of patches per side.
        :return: Number of patches per side (height, width).
        """
        n_patches_height = (self.image_height + self.padding_height) // self.patch_height
        n_patches_width = (self.image_width + self.padding_width) // self.patch_width
        return n_patches_height, n_patches_width

    def patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches to image.
        :param patches: Patches tensor.
        :return: Image tensor.
        """
        n_patches_height, n_patches_width = self._get_n_patches_per_side()
        n_channels = patches.shape[2]

        return patches \
            .reshape(patches.shape[0], n_patches_height, n_patches_width, n_channels, self.patch_height,
                     self.patch_width) \
            .permute(0, 3, 1, 4, 2, 5) \
            .reshape(patches.shape[0], n_channels, self.image_height + self.padding_height,
                     self.image_width + self.padding_width)
