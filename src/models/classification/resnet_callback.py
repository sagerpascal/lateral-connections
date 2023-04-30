"""
ResNet Models with callbacks that are called after each block.
"""

import torch
from typing import Dict, Any, Optional, Callable
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class BlockCallbackResNet18(ResNet):
    """
    ResNet18 with callbacks that are called after each block.
    """

    def __init__(
            self,
            conf: Dict[str, Any],
            after_block_callbacks: Optional[
                Dict[str, Callable[[int, torch.Tensor, Optional[torch.Tensor]], None]]] = None):
        """
        Constructor.
        :param conf: Configuration dictionary.
        :param after_block_callbacks: Callbacks that are called after each block.
        """
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=conf['dataset']['num_classes'])
        self.conf = conf
        if after_block_callbacks is None:
            after_block_callbacks = {}
        self.after_block_callbacks = after_block_callbacks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.maxpool = nn.Identity()


    def register_after_block_callback(
            self, key: str,
            callback: Callable[[int, torch.Tensor, Optional[torch.Tensor]], None]):
        """
        Register a callback that is called after each block.
        :param key: Key of the callback.
        :param callback: Callback function.
        """
        self.after_block_callbacks[key] = callback


    def unregister_after_block_callback(self, key: str):
        """
        Unregister a callback.
        :param key: Key of the callback to remove.
        """
        self.after_block_callbacks.pop(key)


    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor.
        :param y: Optional labels.
        :return: Prediction of the model.
        """
        return self._forward_impl(x, y)


    def notify(self, block: int, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Notify all callbacks that are registered for the given block.
        :param block: Block number (from which block the data is).
        :param x: Input tensor.
        :param y: Optional labels.
        """
        for callback in self.after_block_callbacks.values():
            callback(block, x, y)


    def _forward_impl(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor.
        :param y: Optional labels.
        :return: Prediction of the model.
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.notify(1, x, y)

        x = self.layer1(x)
        self.notify(2, x, y)
        x = self.layer2(x)
        self.notify(3, x, y)
        x = self.layer3(x)
        self.notify(4, x, y)
        x = self.layer4(x)
        self.notify(5, x, y)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        self.notify(6, x, y)

        return x
