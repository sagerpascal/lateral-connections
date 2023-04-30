import unittest
from src.data.utils.patches2d import Patches2D
import torch


class TestPatches2D(unittest.TestCase):

    def test_padding_valid(self):
        invalid_paddings = [-1, 1, 3]
        for padding in invalid_paddings:
            self.assertRaises(AssertionError, Patches2D, 256, 256, 64, 64, padding, 0)
            self.assertRaises(AssertionError, Patches2D, 256, 256, 64, 64, 0, padding)

    def test_patch_size(self):
        # ok
        Patches2D(10, 10, 2, 2, 0, 0)

        # Patch width too big
        self.assertRaises(AssertionError, Patches2D, 10, 10, 12, 2, 0, 0)
        self.assertRaises(AssertionError, Patches2D, 10, 10, 8, 2, 4, 0)

        # Patch height too big
        self.assertRaises(AssertionError, Patches2D, 10, 10, 2, 12, 0, 0)
        self.assertRaises(AssertionError, Patches2D, 10, 10, 2, 8, 0, 4)

    def test_patch_size_valid(self):
        # ok
        Patches2D(10, 10, 2, 2, 0, 0)
        Patches2D(10, 10, 2, 2, 2, 0)
        Patches2D(10, 10, 2, 2, 0, 2)
        Patches2D(10, 10, 4, 4, 2, 2)
        Patches2D(12, 12, 4, 4, 0, 0)

        # invalid height
        self.assertRaises(AssertionError, Patches2D, 10, 10, 4, 2, 4, 0)
        self.assertRaises(AssertionError, Patches2D, 10, 10, 4, 2, 0, 0)

        # invalid width
        self.assertRaises(AssertionError, Patches2D, 10, 10, 2, 4, 0, 4)
        self.assertRaises(AssertionError, Patches2D, 10, 10, 2, 4, 0, 0)

    def test_images_to_patches_no_padding(self):
        batch_size = 6
        num_channels = 5
        img_size_h, img_size_w = 32, 24
        patch_size_h, patch_size_w = 16, 8
        n_patches = int((img_size_h / patch_size_h) * (img_size_w / patch_size_w))
        img = torch.rand((batch_size, num_channels, img_size_h, img_size_w))
        p2d = Patches2D(img_size_h, img_size_w, patch_size_h, patch_size_w, 0, 0)

        # check shape
        self.assertEqual(p2d.image_to_patches(img).shape,
                         torch.Size((batch_size, n_patches, num_channels, patch_size_h, patch_size_w)))

        # check if patches_to_image and image_to_patches are inverse
        self.assertTrue(torch.all(img == p2d.patches_to_image(p2d.image_to_patches(img))))

        # check if patches are correct
        img_patches = p2d.image_to_patches(img)
        for i in range(n_patches):
            hi = (i // (img_size_w // patch_size_w)) * patch_size_h
            wi = (i % (img_size_w // patch_size_w)) * patch_size_w
            img_slice = img[:, :, hi: hi+patch_size_h, wi: wi+patch_size_w]
            self.assertTrue(torch.all(img_slice == img_patches[:, i, ...]))


    def test_images_to_patches_padding(self):
        batch_size = 6
        num_channels = 5
        img_size_h, img_size_w = 28, 22
        patch_size_h, patch_size_w = 16, 8
        padding_h, padding_w = 4, 2
        n_patches = int(((img_size_h + padding_h) / patch_size_h) * ((img_size_w + padding_w) / patch_size_w))
        img = torch.rand((batch_size, num_channels, img_size_h, img_size_w))
        img_result = torch.zeros((batch_size, num_channels, img_size_h + padding_h, img_size_w + padding_w))
        img_result[:, :, padding_h // 2: padding_h // 2 + img_size_h, padding_w // 2: padding_w // 2 + img_size_w] = img
        p2d = Patches2D(img_size_h, img_size_w, patch_size_h, patch_size_w, padding_h, padding_w)

        # check shape
        self.assertEqual(p2d.image_to_patches(img).shape,
                         torch.Size((batch_size, n_patches, num_channels, patch_size_h, patch_size_w)))

        # check if patches_to_image and image_to_patches are inverse
        self.assertTrue(torch.all(img_result == p2d.patches_to_image(p2d.image_to_patches(img))))

        # check if patches are correct
        img_patches = p2d.image_to_patches(img)
        for i in range(n_patches):
            hi = (i // ((img_size_w + padding_w) // patch_size_w)) * patch_size_h
            wi = (i % ((img_size_w + padding_w) // patch_size_w)) * patch_size_w
            img_slice = img_result[:, :, hi: hi+patch_size_h, wi: wi+patch_size_w]
            self.assertTrue(torch.all(img_slice == img_patches[:, i, ...]))
