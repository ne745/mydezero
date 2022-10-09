if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable


def main():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    y = x + c
    # y = F.sum(t)

    y.backward(retain_grad=True)
    print(y.grad)
    # print(t.grad)
    print(x.grad)
    print(c.grad)


if __name__ == '__main__':
    main()
