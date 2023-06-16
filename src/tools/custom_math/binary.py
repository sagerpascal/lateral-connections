"""
Custom function to deal with binary numbers
"""
from typing import Optional

import torch
from torch import Tensor


def bin2dec(b: Tensor, bits: Optional[int] = None) -> Tensor:
    """
    Convert binary number to decimal number.
    :param b: Tensor of binary numbers, the last dimension is converted.
    :param bits: Number of bits.
    :return: Tensor of decimal numbers.
    """
    if bits is None:
        bits = b.shape[-1]
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def dec2bin(x: Tensor, bits: Optional[int] = None) -> Tensor:
    """
    Convert decimal number to binary number.
    :param x: Tensor of decimal numbers, the last dimension is converted.
    :param bits: Number of bits.
    :return: Tensor of binary numbers.
    """
    if bits is None:
        bits = b.shape[-1]
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
