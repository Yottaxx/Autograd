import numpy as np
from autograd.tensor import Tensor

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=np.float))  # (3,)

# @矩阵乘法
# （100，3）*（3，1）
y_data = x_data @ coef + 5

w = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

learning_rate = 0.001
batch_size = 32
for epoch in range(100):
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        w.zero_grad()
        b.zero_grad()
        end = start + batch_size
        inputs = x_data[start:end]

        predicted = inputs @ w + b
        actual = y_data[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        w -= w.grad * learning_rate
        b -= b.grad * learning_rate

    print(epoch, epoch_loss)
