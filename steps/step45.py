if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


def main():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)

    model = TwoLayerNet(10, 1)

    lr = 0.2

    for i in range(10000):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    x = Variable(np.random.randn(1, 100), name='x')
    model = MLP((10, 20, 30, 1))
    model.plot(x)


if __name__ == '__main__':
    main()
