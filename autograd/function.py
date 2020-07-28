from autograd.tensor import Tensor, Dependency, tensor_exp
import numpy as np


def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


# TODO: implement grad_fn of softmax
# def sofxmax(tensor: Tensor, axis: int = 0) -> Tensor:
#     max_sub = np.max(tensor.data, axis=axis)
#     data = np.exp(tensor.data - max_sub)
#     total = data.sum(axis=axis)
#     data = data / total
#
#     requires_grad = tensor.requires_grad
#     if requires_grad:
#         def grad_fn(grad: np.ndarray) -> np.ndarray:
#             return grad * data * (1 - data)
#
#         depends_on = [Dependency(tensor, grad_fn)]
#     else:
#         depends_on = []
#
#     return Tensor(data, requires_grad, depends_on)


def cross_entry_softmax(t1: Tensor, t2: Tensor, axis: int = 0) -> Tensor:
    max_sub = np.expand_dims(np.mean(t1.data, axis=axis), axis=axis)
    data = np.exp(t1.data - max_sub)
    # print("------------------------")
    # print("data", data)
    total = np.expand_dims(data.sum(axis=axis), axis=axis)
    # print("total", total)
    softmax_data = data / total
    # print("softmax", softmax_data)
    assert softmax_data.shape == t2.data.shape
    data = -(t2.data * np.log(softmax_data)).sum()

    requires_grad = t1.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (softmax_data - t2.data)

        depends_on = [Dependency(t1, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
