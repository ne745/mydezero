if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from dezero import Variable
import dezero.functions as F
import dezero.layers as L


class Network:
    def __init__(self):
        self.l1 = L.Linear(10)
        self.l2 = L.Linear(1)

    def predict(self, x):
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y

    def fit(self, x, y, lr=0.1, iters=100):
        for i in range(iters):
            y_pred = self.predict(x)
            loss = F.mean_squared_error(y, y_pred)

            self.l1.cleargrads()
            self.l2.cleargrads()
            loss.backward()

            for layer in [self.l1, self.l2]:
                for p in layer.params():
                    p.data -= lr * p.grad.data

            if i % 1000 == 0:
                print(loss)


def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    nn = Network()
    nn.fit(x, y, 0.2, 10000)

    fig, ax = plt.subplots()
    ax.plot(x.data, y.data, 'o')

    t = np.arange(0, 1, 0.01)[:, np.newaxis]
    y_pred = nn.predict(t)
    ax.plot(t, y_pred.data)
    plt.savefig('output.jpg')


if __name__ == '__main__':
    main()
