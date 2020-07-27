import numpy as np
from typing import List
from autograd import Tensor, Parameter, Module
from autograd.optim import SGD
from autograd.function import tanh
from autograd.Layers import Linear, SimpleAttention

"""
print the numbers 1 to 100,
except  
    if the number is divisible by 3 print "fizz"
    if the number is divisible by 5 print "fizz"
    if the number is divisible by 15 print "fizz_buzz"

"""


def binary_encode(x: int) -> List[int]:
    return [x >> i & 1 for i in range(10)]


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


# @矩阵乘法
# （100，3）*（3，1）

x_train = Tensor([binary_encode(x) for x in range(101, 1024)])
y_train = Tensor([fizz_buzz_encode(x) for x in range(101, 1024)])


class FizzBuzzModule(Module):
    def __init__(self, num_hidden: int = 50) -> None:
        self.linear = Linear(10, num_hidden)
        self.linear2 = Linear(num_hidden, 4)

    def predict(self, in_puts: Tensor):
        # inputs (batch_size,10)
        x1 = self.linear(in_puts)
        x2 = tanh(x1)
        x3 = self.linear2(x2)  # (batch_size,4)
        return x3


# w = Tensor(np.random.randn(3), requires_grad=True)
# b = Tensor(np.random.randn(), requires_grad=True)
# w = Parameter(3)  # tensor(3,),requires_grad =True,random values
# b = Parameter()

optimizer = SGD(lr=0.001)
# learning_rate = 0.001
batch_size = 32
module = FizzBuzzModule()

starts = np.arange(0, x_train.shape[0], batch_size)
for epoch in range(10000):
    epoch_loss = 0.0

    np.random.shuffle(starts)
    for start in starts:
        end = start + batch_size

        module.zero_grad()
        inputs = x_train[start:end]

        predicted = module.predict(inputs)
        actual = y_train[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        # module.w -= module.w.grad * learning_rate
        # module.b -= module.b.grad * learning_rate
        optimizer.step(module)
    print(epoch, epoch_loss)

num_correct = 0
for x in range(1, 101):
    inputs = Tensor([binary_encode(x)])
    predicted = module.predict(inputs)[0]
    predicted_idx = np.argmax(predicted.data)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizz_buzz"]

    if predicted_idx == actual_idx:
        num_correct += 1
    print(x, labels[predicted_idx], labels[actual_idx])

print(num_correct, "/100")
