import unittest
from autograd import Tensor, tensor_exp

import numpy as np


class TestTensorExp(unittest.TestCase):
    def test_simple_exp(self):
        t1 = Tensor(np.random.randn(3, 4, 5), requires_grad=True)
        t2 = t1.exp()
        t2.backward(Tensor(np.ones((3, 4, 5))))
        assert np.allclose(t1.grad.data, t2.data)

    def test_simple_tensor_exp(self):
        t1 = Tensor(np.random.randn(3, 4, 5), requires_grad=True)
        t2 = tensor_exp(t1)
        t2.backward(Tensor(np.ones((3, 4, 5))))
        assert np.allclose(t1.grad.data, t2.data)
