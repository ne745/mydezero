if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from dezero import Variable
import dezero.functions as F


class LinearRegression:
    def __init__(self, lr=0.1, iters=100) -> None:
        self.lr = lr
        self.iters = iters
        self.W = Variable(np.zeros((1, 1)))
        self.b = Variable(np.zeros(1))

    def predict(self, x):
        return F.matmul(x, self.W) + self.b

    def mean_squared_error(self, x0, x1):
        diff = x0 - x1
        return F.sum(diff ** 2) / len(diff)

    def fit(self, x, y):
        for _ in range(self.iters):
            y_pred = self.predict(x)
            loss = self.mean_squared_error(y, y_pred)

            self.W.cleargrad()
            self.b.cleargrad()
            loss.backward()

            self.W.data -= self.lr * self.W.grad.data
            self.b.data -= self.lr * self.b.grad.data
        return self.W.data.squeeze(), self.b.data.squeeze()


def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    linear_regression = LinearRegression()
    W, b = linear_regression.fit(x, y)

    print(W, b)

    fig, ax = plt.subplots()
    ax.plot(x.data, y.data, 'o')
    ax.plot([0, 1], [b, W + b])
    plt.savefig('output.jpg')


if __name__ == '__main__':
    main()
