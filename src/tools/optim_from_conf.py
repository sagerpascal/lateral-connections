from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def torch_optim_from_conf(
        params: Union[List[Iterator[Parameter]] | Iterator[Parameter]],
        optim_key: Union[List[str] | str],
        config: Dict,
) -> Tuple[Union[Optimizer | List[Optimizer]], Union[Optional[LRScheduler] | List[Optional[LRScheduler]]]]:
    """
    Create a torch optimizer from a config dict. The optimizer key specifies which optimizer to use from the config
    dict.
    Either, a single parameter object and a single optimizer key must be provided or a list of parameter objects and
    a list of optimizer keys with the same length.

    :param params: Parameter object or list of parameter objects
    :param optim_key: Optimizer key or list of optimizer keys
    :param config: Config dict
    :return: Optimizer(s), scheduler(s)
    """
    return_list = False
    if isinstance(params, list):
        assert isinstance(optim_key, list), 'If multiple parameter objects are specified, optim_key must be a list.'
        assert len(params) == len(optim_key), 'Number of optimizer keys and parameter objects must match.'
        return_list = True
    else:
        params = [params]
        if isinstance(optim_key, str):
            optim_key = [optim_key]

    optimizers, schedulers = [], []
    for param, optimizer_conf in zip(params, optim_key):
        opt_conf = config['optimizers'][optimizer_conf]
        class_ = getattr(torch.optim, opt_conf['type'])
        optimizer = class_(param, **opt_conf['params'])
        optimizers.append(optimizer)
        if 'scheduler' in opt_conf:
            class_ = getattr(torch.optim.lr_scheduler, opt_conf['scheduler']['type'])
            schedulers.append(class_(optimizer, **opt_conf['scheduler']['params']))
        else:
            schedulers.append(None)

    if return_list:
        return optimizers, schedulers
    else:
        assert len(optimizers) == 1, 'If only one parameter object is specified, only one optimizer must be returned.'
        return optimizers[0], schedulers[0]
