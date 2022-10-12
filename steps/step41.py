if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable
import dezero.functions as F


def main():
    x = Variable(np.array([1, 2, 3]))
    W = Variable(np.array([4, 5, 6]))
    y = F.matmul(x, W)
    y.backward()
    print(y)
    print(x.grad.shape)
    print(W.grad.shape)

    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()
    print(x.grad.shape)
    print(W.grad.shape)


if __name__ == '__main__':
    main()
