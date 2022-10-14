if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from dezero import Variable
import dezero.functions as F


class Network:
    def __init__(self, num_input, num_hidden, num_output) -> None:
        self.W1 = Variable(0.01 * np.random.randn(num_input, num_hidden))
        self.b1 = Variable(np.zeros(num_hidden))
        self.W2 = Variable(0.01 * np.random.randn(num_hidden, num_output))
        self.b2 = Variable(np.zeros(num_output))

    def predict(self, x):
        y = F.linear(x, self.W1, self.b1)
        y = F.sigmoid(y)
        y = F.linear(y, self.W2, self.b2)
        return y

    def fit(self, x, y, lr=0.1, iters=100):
        for i in range(iters):
            y_pred = self.predict(x)
            loss = F.mean_squared_error(y, y_pred)

            self.W1.cleargrad()
            self.b1.cleargrad()
            self.W2.cleargrad()
            self.b2.cleargrad()
            loss.backward()

            self.W1.data -= lr * self.W1.grad.data
            self.b1.data -= lr * self.b1.grad.data
            self.W2.data -= lr * self.W2.grad.data
            self.b2.data -= lr * self.b2.grad.data

            if i % 1000 == 0:
                print(loss)


def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    nn = Network(1, 10, 1)
    nn.fit(x, y, 0.2, 10000)

    fig, ax = plt.subplots()
    ax.plot(x.data, y.data, 'o')

    t = np.arange(0, 1, 0.01)[:, np.newaxis]
    y_pred = nn.predict(t)
    ax.plot(t, y_pred.data)
    plt.savefig('output.jpg')


if __name__ == '__main__':
    main()
