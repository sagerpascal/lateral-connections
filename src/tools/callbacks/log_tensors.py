import torch
from typing import Optional, Union
import pickle
import os
from pathlib import Path

_path_t = Union[str, os.PathLike, Path]


class LogTensorsCallback:
    """
    Callback to log tensors to a pickle file. During each call, the tensors are added to a dictionary, which is
    pickled at the end of the training.
    """

    def __init__(self, pickle_fp: _path_t):
        """
        Constructor.
        :param pickle_fp: File path to the pickle file.
        """
        self.pickle_fp = pickle_fp
        self.data_dict = {}

    def __call__(self, block_idx: int, x: torch.Tensor, y: Optional[torch.Tensor]):
        """
        Callback function to log the tensors.
        :param block_idx: Index of the block.
        :param x: Input tensor.
        :param y: Optional target tensor.
        """
        if block_idx not in self.data_dict:
            self.data_dict[block_idx] = {}
        if y is not None:
            yi = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            for i in range(yi.shape[0]):
                if yi[i] not in self.data_dict[block_idx]:
                    self.data_dict[block_idx][yi[i]] = []
                self.data_dict[block_idx][yi[i]].append(x[i].flatten())
        else:
            yi = -1
            x = x.detach().cpu().numpy()
            if yi not in self.data_dict[block_idx]:
                self.data_dict[block_idx][yi] = []
            for i in range(x.shape[0]):
                self.data_dict[block_idx][yi].append(x[i])

    def on_train_end(self):
        """
        Save the data dict to a pickle file.
        """
        if len(self.data_dict) > 0:
            for k, v in self.data_dict.items():
                path = self.pickle_fp.parent / f"{self.pickle_fp.stem}_{k}{self.pickle_fp.suffix}"
                with open(str(path), 'wb') as handle:
                    pickle.dump(v, handle)
