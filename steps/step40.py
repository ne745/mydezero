if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable


def main():
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    y.backward()
    print(y)
    print(x1.grad)


if __name__ == '__main__':
    main()
