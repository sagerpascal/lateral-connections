import torch
from typing import Dict, Iterator, Union, List
from torch.nn import Parameter


def torch_optim_from_conf(
        params: Union[List[Iterator[Parameter]] | Iterator[Parameter]],
        optim_key: Union[List[str] | str],
        config: Dict,
) -> Union[torch.optim.Optimizer | List[torch.optim.Optimizer]]:
    """
    Create a torch optimizer from a config dict. The optimizer key specifies which optimizer to use from the config
    dict.
    Either, a single parameter object and a single optimizer key must be provided or a list of parameter objects and
    a list of optimizer keys with the same length.

    :param params: Parameter object or list of parameter objects
    :param optim_key: Optimizer key or list of optimizer keys
    :param config: Config dict
    :return: Optimizer
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

    optimizers = []
    for param, optimizer_conf in zip(params, optim_key):
        opt_conf = config['optimizers'][optimizer_conf]
        class_ = getattr(torch.optim, opt_conf['type'])
        optimizers.append(class_(param, **opt_conf['params']))

    if return_list:
        return optimizers
    else:
        assert len(optimizers) == 1, 'If only one parameter object is specified, only one optimizer must be returned.'
        return optimizers[0]
