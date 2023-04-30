from typing import Dict, Union, TypeVar, Optional, Tuple, Any
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
import os
from pathlib import Path
from src.data.augmentation import get_transform_pipeline, get_image_augmentation
from src.data.loader import get_torch_data_loaders, get_ffcv_data_loaders, get_ffcv_image_pipeline, get_ffcv_label_pipeline

T_co = TypeVar('T_co', covariant=True)
_path_t = Union[str, os.PathLike, Path]


def _get_dataset(
        dataset_name: str,
        dataset_path: _path_t,
        transform: Optional[transforms.Compose] = None,
) -> Tuple[Any, Optional[Any], Any]:
    """
    Get a dataset based on its name.
    :param dataset_name: Name of the dataset.
    :param transform: Transform to apply to the dataset.
    :return: A dataset.
    """
    if dataset_name == "mnist":
        train_set = MNIST(root=dataset_path, train=True, download=True, transform=transform)
        valid_set = None
        test_set = MNIST(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        train_set = CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        valid_set = None
        test_set = CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset_name == "cifar100":
        train_set = CIFAR100(root=dataset_path, train=True, download=True, transform=transform)
        valid_set = None
        test_set = CIFAR100(root=dataset_path, train=False, download=True, transform=transform)
    elif dataset_name == "imagenet":
        train_set = ImageNet(root=dataset_path, split="train", download=True, transform=transform)
        valid_set = ImageNet(root=dataset_path, split="val", download=True, transform=transform)
        test_set = ImageNet(root=dataset_path, split="test", download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))

    return train_set, valid_set, test_set


def loaders_from_config(config: Dict) -> Union[Any, Any, Any]:
    """
    Get a data loader from a config.
    :param config: Config.
    :return: A data loader.
    """
    data_config = config["dataset"]
    if data_config["loader"] == "torch":
        transform = get_transform_pipeline(config)
        train_set, valid_set, test_set = _get_dataset(data_config["name"], data_config["dir"], transform)
        return get_torch_data_loaders(
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )
    elif data_config["loader"] == "ffcv":
        mean, std = config["dataset"]["mean"], config["dataset"]["std"]
        augmentations = get_image_augmentation(config)
        image_pipeline = get_ffcv_image_pipeline(mean, std, augmentations)
        label_pipeline = get_ffcv_label_pipeline()
        train_set, valid_set, test_set = _get_dataset(data_config["name"], data_config["dir"])
        return get_ffcv_data_loaders(
            image_pipeline=image_pipeline,
            label_pipeline=label_pipeline,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            beton_train_filepath=data_config["beton_dir"] / "train.beton",
            beton_valid_filepath=data_config["beton_dir"] / "valid.beton",
            beton_test_filepath=data_config["beton_dir"] / "test.beton",
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )
    else:
        raise ValueError("Unknown data loader: {}".format(config["loader"]))
