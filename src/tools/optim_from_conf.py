import torch
from typing import Dict, Iterator, Union, List
from torch.nn import Parameter


def torch_optim_from_conf(
        params: Union[List[Iterator[Parameter]] | Iterator[Parameter]],
        config: Dict
) -> Union[torch.optim.Optimizer | List[torch.optim.Optimizer]]:
    """
    Create a torch optimizer from a config dict. If one optimizer is specified in the config file, one parameter object
    is expected. If multiple optimizers are specified, a list of parameter objects is expected.
    :param params: The parameters to optimize
    :param config: Config dict
    :return: Optimizer
    """
    return_list = False
    if isinstance(params, list):
        assert len(params) == len(config['optimizers']), 'Number of optimizers and parameter objects must match.'
        return_list = True
    else:
        assert len(config['optimizers']) == 1, 'If only one optimizer is specified, only one parameter object is expected.'

    optimizers = []
    for param, optimizer_conf in zip(params, config['optimizers'].values()):
        class_ = getattr(torch.optim, optimizer_conf['type'])
        optimizers.append(class_(param, **optimizer_conf['params']))

    if return_list:
        return optimizers
    else:
        assert len(optimizers) == 1, 'If only one optimizer is specified, only one optimizer object is returned.'
        return optimizers[0]
