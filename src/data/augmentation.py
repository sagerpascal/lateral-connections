from typing import Dict
from torchvision import transforms
from torchvision.transforms import RandomApply
import torch.nn as nn
from typing import List, Any


def get_image_augmentation(config: Dict) -> List[Any]:
    """
    Get image augmentation based on the config.
    :param config: Config.
    :return: A torchvision transform object.
    """
    augmentations = config["dataset"]["augmentation"]
    transforms_list = []

    for aug in augmentations.values():
        t = []
        for transf in aug[1]['transformations']:
            class_ = getattr(transforms, transf.pop('transformation'))
            t.append(class_(**transf))
        transforms_list.append(RandomApply(nn.ModuleList(t), aug[0]['probability']))

    return transforms_list


def _get_image_normalization(config: Dict) -> transforms.Normalize:
    """
    Get image normalization based on the config.
    :param config: Config.
    :return: A torchvision transform object.
    """
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]
    return transforms.Normalize(mean, std)


def get_transform_pipeline(config: Dict) -> transforms.Compose:
    """
    Get the image pipeline.
    :param config: Config.
    :return: Image pipeline.
    """
    image_pipeline = [transforms.ToTensor()]
    image_pipeline.extend(get_image_augmentation(config))
    image_pipeline.append(_get_image_normalization(config))
    return transforms.Compose(image_pipeline)
