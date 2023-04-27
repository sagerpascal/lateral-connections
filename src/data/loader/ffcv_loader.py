import torchvision
from torch.utils.data import Dataset
from typing import Optional, TypeVar, Callable, Any, List, Union
import torch
import os
from pathlib import Path
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from utils import print_info_data, print_warn

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]
_path_t = Union[str, os.PathLike, Path]


def _create_beton_file(dataset: Dataset[T_co], filepath: _path_t):
    """
    Create a beton file from a dataset.
    :param dataset: Dataset.
    :param filepath: Where to store the beton file.
    """
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
        print_info_data("Creating folder {}...".format(filepath.parent))

    print_info_data("Creating beton file at {}...".format(filepath))

    writer = DatasetWriter(filepath, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(dataset)


def get_image_pipeline(mean: List, std: List, augmentations: Optional[List[Operation]] = None) -> List[Operation]:
    """
    Get the image pipeline.
    :param mean: Mean of the dataset.
    :param std: Standard deviation of the dataset.
    :param augmentations: Optional augmentations.
    :return: Image pipeline.
    """

    if mean[0] < 1 and std[0] < 1:
        for i in range(len(mean)):
            mean[i] = mean[i] * 255
            std[i] = std[i] * 255
        print_warn(f"Mean and std are in [0, 1] range. Multiplying by 255...\nNew mean: {mean}\nNew std: {std}")

    image_pipeline = [
        SimpleRGBImageDecoder(),
    ]
    if augmentations is not None:
        image_pipeline.extend(augmentations)
    image_pipeline.extend([
        ToTensor(),
        ToDevice(torch.device('cuda:0'), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float32),
        torchvision.transforms.Normalize(mean, std),
    ])
    return image_pipeline


def get_label_pipeline() -> List[Operation]:
    """
    Get the label pipeline.
    :return: Label pipeline.
    """
    return [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]


def get_loader(
        dataset: Dataset[T_co],
        beton_filepath: _path_t,
        image_pipeline: List[Operation],
        label_pipeline: List[Operation],
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
) -> Optional[Loader]:
    """
    Get a data loader.
    :param dataset: Dataset.
    :param beton_filepath: Path to the beton file.
    :param batch_size: Batch size.
    :param num_workers: Number of workers.
    :param image_pipeline: Image pipeline.
    :param label_pipeline: Label pipeline.
    :param shuffle: Whether to shuffle the dataset.
    :param drop_last: Whether to drop the last mini-batch.
    :return: Data loader.
    """

    if dataset is None or beton_filepath is None:
        return None

    else:
        if not os.path.exists(beton_filepath):
            _create_beton_file(dataset, beton_filepath)

        return Loader(
            beton_filepath,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
            drop_last=drop_last,
            pipelines={'image': image_pipeline,
                       'label': label_pipeline},
            seed=0)


def get_ffcv_data_loaders(
        image_pipeline: List[Operation],
        label_pipeline: List[Operation],
        train_dataset: Optional[Dataset[T_co]] = None,
        valid_dataset: Optional[Dataset[T_co]] = None,
        test_dataset: Optional[Dataset[T_co]] = None,
        beton_train_filepath: Optional[_path_t] = None,
        beton_valid_filepath: Optional[_path_t] = None,
        beton_test_filepath: Optional[_path_t] = None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
) -> (Optional[Loader], Optional[Loader]):
    """
    Get data loaders for training and testing.
    :param image_pipeline: Image pipeline.
    :param label_pipeline: Label pipeline.
    :param train_dataset: Training dataset.
    :param test_dataset: Testing dataset.
    :param beton_train_filepath: Path to the beton file for the training dataset.
    :param beton_test_filepath: Path to the beton file for the testing dataset.
    :param batch_size: Batch size.
    :param num_workers: Number of workers.
    :param shuffle: Whether to shuffle the dataset.
    :param drop_last: Whether to drop the last mini-batch.
    :return: Training and testing data loaders.
    """
    train_loader = get_loader(train_dataset, beton_train_filepath, image_pipeline, label_pipeline,
                              batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)

    valid_loader = get_loader(valid_dataset, beton_valid_filepath, image_pipeline, label_pipeline,
                              batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)

    test_loader = get_loader(test_dataset, beton_test_filepath, image_pipeline, label_pipeline, batch_size=batch_size,
                             num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)

    return train_loader, valid_loader, test_loader
