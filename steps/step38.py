if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable
import dezero.functions as F


def main():
    print('reshape')
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)
    print(x)
    print(y)
    print(y.grad)
    print(x.grad)

    x = Variable(np.random.rand(1, 2, 3))
    # y = x.reshape((2, 3))
    # y = x.reshape([2, 3])
    y = x.reshape(2, 3)
    print(y)

    print('transpose')
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward(retain_grad=True)
    print(x)
    print(y)
    print(y.grad)
    print(x.grad)

    x = Variable(np.random.rand(1, 2, 3))
    y = x.transpose()
    print(y)
    y = x.T
    print(y)


if __name__ == '__main__':
    main()
