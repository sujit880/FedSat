import numpy as np


def list_overlapping(overlap_ratio, list_dict):
    # Calculate the desired overlap size for each list based on the ratio
    overlap_sizes = {
        k: max(int(len(v) * overlap_ratio), 1) for k, v in list_dict.items()
    }

    # print(f'Overlap Sizes: {overlap_sizes}')

    # Select overlapping elements for each list using numpy's random.choice
    overlap_elements_set = []
    overlap_elements_set.extend(
        np.random.choice(lst, size, replace=False)
        for lst, size in zip(list(list_dict.values()), list(overlap_sizes.values()))
    )
    overlap_elements = [item for sublist in overlap_elements_set for item in sublist]

    # print(f'Overlap Elements: {overlap_elements}')

    # Add the overlapping elements to the respective lists
    for i, key in enumerate(list_dict):
        # print(i,key)
        lst = list_dict[key]
        options = [x for x in overlap_elements if x not in lst]
        items = np.random.choice(
            options, min(len(options), overlap_sizes[key]), replace=False
        )
        lst.extend(items)

    # print(f'Lists with Overlap: {[lst for lst in list_dict.values()]}')

    return list_dict


# Normalize dict of single element


def normalize_dict(arr_dict):
    # print(f'input: {arr_dict}')
    if len(arr_dict.keys()) <= 1:
        return arr_dict
    else:
        sum_arr = np.sum(list(arr_dict.values()))
        # print(f'\nSum_arr: {sum_arr}')
        for k, v in arr_dict.items():
            if sum_arr == 0:
                print(f"arr_dict: {arr_dict}, sum_arr: {sum_arr}")
                arr_dict[k] = 1 / len(arr_dict.keys())
            arr_dict[k] = v / sum_arr
        # print(f'arr_dict: {arr_dict}')
        return arr_dict


import torch
from collections import OrderedDict
from typing import Union, List, Tuple


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        List of parameters [, names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters
    
def get_optimizer_by_name(optm, parameters: list[torch.tensor], **kwargs):
    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop,
        "Adagrad": torch.optim.Adagrad,
        # You can easily add more optimizers here in the future
    }

    # Get the optimizer class, or default to SGD if the optimizer name is unknown
    optimizer_class = optimizers.get(optm, torch.optim.SGD)

    # # Combine parameters from all models
    # combined_params = []
    # for model in models:
    #     combined_params.extend(model.parameters())  # Adding model parameters to the list

    # Handle optimizer-specific arguments
    if optm == "SGD":
        # For SGD, we expect momentum
        kwargs["momentum"] = kwargs.get("momentum", 0.9)  # Default momentum if not provided
        # Remove 'betas' for SGD, as it's not used
        kwargs.pop("betas", None)

    elif optm == "Adam":
        # For Adam, we expect betas
        kwargs["betas"] = kwargs.get("betas", (0.9, 0.999))  # Default betas if not provided
        # Remove 'momentum' for Adam, as it's not used by the optimizer
        kwargs.pop("momentum", None)
    
    # You can apply further checks for other optimizers (RMSprop, Adagrad, etc.) here
    
    # Return the optimizer with the provided keyword arguments
    return optimizer_class(parameters, **kwargs)

