import torch
import numpy as np


def __num_elems(shape):
    """Returns the number of elements in the given shape

    Args:
        shape: torch.Size

    Return:
        tot_elems: int
    """
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems


def graph_size(model):
    """Returns the size of the given PyTorch model in bytes

    The size of the model is calculated by summing up the sizes of each
    trainable parameter. The sizes of parameters are calculated by multiplying
    the number of bytes in their dtype with their number of elements, captured
    in their shape attribute

    Args:
        model: PyTorch model
    Return:
        integer representing size of model (in bytes)
    """
    tot_size = 0
    for param in model.parameters():
        tot_elems = __num_elems(param.shape)
        dtype_size = int(param.element_size())
        param_size = tot_elems * dtype_size
        tot_size += param_size
    return tot_size


def process_sparse_grad(grads):
    """
    Args:
        grads: grad returned by LSTM model (only for the shakespaere dataset)
    Return:
        a flattened grad in numpy (1-D array)
    """

    indices = grads[0].indices
    values = grads[0].values
    first_layer_dense = torch.zeros((80, 8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads = first_layer_dense.flatten().numpy()
    for i in range(1, len(grads)):
        client_grads = np.append(
            client_grads, grads[i].flatten().numpy()
        )  # output a flattened array
    return client_grads


def process_sparse_grad2(grads):
    """
    :param grads: grad returned by LSTM model (only for the shakespaere dataset) (with indices)
    :return: grads with the same shape as weights
    """
    client_grads = []
    indices = grads[0].indices
    values = grads[0].values
    first_layer_dense = torch.zeros((80, 8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]
    client_grads.append(first_layer_dense.numpy())
    for i in range(1, len(grads)):
        client_grads.append(grads[i].numpy())
    return client_grads


def process_grad(grads):
    """
    Args:
        grads: grad
    Return:
        a flattened grad in numpy (1-D array)
    """

    client_grads = grads[0].numpy()

    for i in range(1, len(grads)):
        client_grads = np.append(
            client_grads, grads[i].numpy()
        )  # output a flattened array

    return client_grads


def cosine_sim(a, b):
    """Returns the cosine similarity between two arrays a and b"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product * 1.0 / (norm_a * norm_b)


def list_to_tensor(data):
    data_tesor = []
    if not isinstance(data, list):
        raise ValueError(f"Provided data is not list")
    else:
        for elements in data:
            # tesor_data = torch.Tensor(elements)
            element_tensor = np.empty((0,))
            if isinstance(elements, list):
                for element in elements:
                    tensor_data = torch.tensor(element)
                    element_tensor = np.append(element_tensor, tensor_data)
            # np_elements = np.array(elements)
            tensor_data = torch.from_numpy(element_tensor)
            data_tesor.append(tensor_data)
    return data_tesor
