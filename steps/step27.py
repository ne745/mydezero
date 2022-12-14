if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np

from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def sin_maclaurin(x, threshold=1e-4):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def main():
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()
    print(y.data)
    print(x.grad)

    x.cleargrad()
    y = sin_maclaurin(x)
    y.backward()
    print(y.data)
    print(x.grad)

    x.name = 'x'
    y.name = 'y'
    plot_dot_graph(y, verbose=False, to_file='my_sin.png')


if __name__ == '__main__':
    main()
