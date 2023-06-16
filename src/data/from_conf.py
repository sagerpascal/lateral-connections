import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST

from data.augmentation import get_image_augmentation, get_transform_pipeline
from data.custom_datasets.straight_line import StraightLine
from data.loader import get_ffcv_data_loaders, get_ffcv_image_pipeline, get_ffcv_label_pipeline, \
    get_torch_data_loaders

T_co = TypeVar('T_co', covariant=True)
_path_t = Union[str, os.PathLike, Path]


def _get_dataset(
        dataset_name: str,
        dataset_path: Optional[_path_t] = None,
        transform: Optional[transforms.Compose] = None,
        dataset_config: Optional[Dict] = None,
) -> Tuple[Any, Optional[Any], Any]:
    """
    Get a dataset based on its name.
    :param dataset_name: Name of the dataset.
    :param dataset_path: Path to the dataset.
    :param transform: Transform to apply to the dataset.
    :param dataset_config: Config of the dataset.
    :return: A dataset.
    """
    if dataset_name == "mnist":
        train_set = MNIST(root=dataset_path, transform=transform, **dataset_config['train_dataset_params'])
        valid_set = None
        test_set = MNIST(root=dataset_path, transform=transform, **dataset_config['test_dataset_params'])
    elif dataset_name == "cifar10":
        train_set = CIFAR10(root=dataset_path, transform=transform, **dataset_config['train_dataset_params'])
        valid_set = None
        test_set = CIFAR10(root=dataset_path, transform=transform, **dataset_config['test_dataset_params'])
    elif dataset_name == "cifar100":
        train_set = CIFAR100(root=dataset_path, transform=transform, **dataset_config['train_dataset_params'])
        valid_set = None
        test_set = CIFAR100(root=dataset_path, transform=transform, **dataset_config['test_dataset_params'])
    elif dataset_name == "imagenet":
        train_set = ImageNet(root=dataset_path, transform=transform, **dataset_config['train_dataset_params'])
        valid_set = ImageNet(root=dataset_path, transform=transform, **dataset_config['valid_dataset_params'])
        test_set = ImageNet(root=dataset_path, transform=transform, **dataset_config['test_dataset_params'])
    elif dataset_name == "straightline":
        train_set = StraightLine(transform=transform, **dataset_config['train_dataset_params'])
        valid_set = StraightLine(transform=transform, **dataset_config['valid_dataset_params'])
        test_set = StraightLine(transform=transform, **dataset_config['test_dataset_params'])
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
        dir_ = data_config["dir"] if "dir" in data_config else None
        train_set, valid_set, test_set = _get_dataset(data_config["name"], dir_, transform, data_config)
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
        dir_ = data_config["dir"] if "dir" in data_config else None
        train_set, valid_set, test_set = _get_dataset(data_config["name"], dataset_path=dir_, dataset_config=data_config)
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
