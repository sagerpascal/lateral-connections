import json
import random
from pathlib import Path
from typing import List, Literal, Tuple
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset


class ArcDataset(Dataset):

    def __init__(self, add_noise: bool = False):
        self.add_noise = add_noise
        self.path = Path('../data/arc_subset.json')
        self.tasks = self.load_tasks()
        self.tasks = self.load_tasks()

    def load_tasks(self) -> List[dict]:
        with open(str(self.path.absolute())) as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.tasks)

    def get_item(self,
                 idx: int,
                 one_hot: bool = True,
                 pad: bool = True,
                 ) -> Tuple[torch.Tensor, dict]:
        task = self.tasks[idx]
        data = torch.Tensor(task['data'])
        metadata = task['metadata']
        metadata['img_size'] = data.shape

        if pad:
            final_size = 32
            pad_top = (final_size - data.shape[0]) // 2
            pad_bottom = final_size - data.shape[0] - pad_top
            pad_left = (final_size - data.shape[1]) // 2
            pad_right = final_size - data.shape[1] - pad_left
            metadata['pad'] = (pad_left, pad_right, pad_top, pad_bottom)
            data = torch.nn.functional.pad(data, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        if self.add_noise:
            # data = data.argmax(dim=2).float()

            noise_probability = 0.01
            noise = torch.rand_like(data)
            noise_indices = random.sample(range(noise.numel()), int((1 - noise_probability) * noise.numel()))
            noise_s = noise.shape
            noise = noise.view(-1)
            noise[noise_indices] = 0
            noise = noise.view(noise_s)

            data = (data + (noise * 10).round()) % 10
            # data = F.one_hot(data.long(), num_classes=10)
            # data = data.permute(0, 1, 4, 2, 3).float()

        if one_hot:
            data = torch.nn.functional.one_hot(data.to(torch.int64), num_classes=10).permute(2, 0, 1).float()

        return data, metadata

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        return self.get_item(idx)


class ArcDatasetOld(Dataset):

    def __init__(self,
                 split: Literal['train', 'test'],

                 ):
        self.split = split
        self.path = Path(f'../../../data/ARC-800-tasks/{split}/')
        self.tasks = self.load_tasks()

    def load_tasks(self) -> List[dict]:
        tasks = []
        for item in self.path.glob("*"):
            with open(str(item.absolute())) as f:
                data = json.load(f)
            data['metadata'] = {'task': item.name}
            tasks.append(data)
        return tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def to_tensor_and_pad(self, x: List[List[int]]) -> torch.Tensor:
        final_size = 32
        t = torch.Tensor(x)
        pad_top = (final_size - t.shape[0]) // 2
        pad_bottom = final_size - t.shape[0] - pad_top
        pad_left = (final_size - t.shape[1]) // 2
        pad_right = final_size - t.shape[1] - pad_left
        return torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        task = self.tasks[idx]
        train_data = task['train']
        test_data = task['test']
        metadata = task['metadata']

        train_input = torch.stack([self.to_tensor_and_pad(train_data[i]['input']) for i in range(len(train_data))])
        train_output = torch.stack([self.to_tensor_and_pad(train_data[i]['output']) for i in range(len(train_data))])
        test_input = torch.stack([self.to_tensor_and_pad(test_data[i]['input']) for i in range(len(test_data))])
        test_output = torch.stack([self.to_tensor_and_pad(test_data[i]['output']) for i in range(len(test_data))])

        return train_input, train_output, test_input, test_output, metadata


def get_colormap():
    # Define colors that resemble the Arc Challenge
    colors = [(0.0, 0.0, 0.0),  # 0 = black
              (0.0, 116 / 255, 217 / 255),  # 1 = blue
              (1.0, 65 / 255, 54 / 255),  # 2 = red
              (46 / 255, 204 / 255, 64 / 255),  # 3 = green
              (1.0, 220 / 255, 0.0),  # 4 = yellow
              (170 / 255, 170 / 255, 170 / 255),  # 5 = gray
              (240 / 255, 18 / 255, 190 / 255),  # 6 = pink
              (255 / 255, 133 / 255, 27 / 255),  # 7 = orange
              (127 / 255, 219 / 255, 1 / 255),  # 8 = light blue
              (135 / 255, 12 / 255, 37 / 255),  # 9 = wine red
              ]

    # Create a list of positions for the colors
    positions = np.linspace(0, 1, len(colors))

    # Create a custom colormap using LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ArcChallenge", list(zip(positions, colors)))

    return cmap





def _plot_some_samples():
    import matplotlib.pyplot as plt

    dataset = ArcDataset()

    fig, axs = plt.subplots(len(dataset) // 5, 5, figsize=(15, 8))
    for i in range(len(dataset)):
        data, metadata = dataset.get_item(i, False, False)
        axs[i // 5, i % 5].set_title(f'{metadata["task"]}')
        axs[i // 5, i % 5].imshow(data, cmap=get_colormap(), vmin=0, vmax=9, interpolation='none')

        # Major ticks
        axs[i // 5, i % 5].set_xticks([])
        axs[i // 5, i % 5].set_yticks([])

        # Minor ticks
        axs[i // 5, i % 5].set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
        axs[i // 5, i % 5].set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)

        # Gridlines based on minor ticks
        axs[i // 5, i % 5].grid(which='minor', color='gray', linestyle='-', linewidth=1)

        for lbl in axs[i // 5, i % 5].axes.get_xticklabels():
            lbl.set_fontsize(0.0)
        for lbl in axs[i // 5, i % 5].axes.get_yticklabels():
            lbl.set_fontsize(0.0)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _plot_some_samples()
