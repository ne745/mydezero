if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable
from dezero.core import as_variable
import dezero.functions as F
from dezero.models import MLP


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


def main():
    np.random.seed(0)

    model = MLP((10, 3))

    x = Variable(np.array([
        [0.2, -0.4],
        [-0.3, -0.7],
        [0.5, 1.4],
    ]))
    y = model(x)
    p1 = F.softmax_simple(y)
    p2 = F.softmax(y)
    print(y)
    print(p1)
    print(p2)


if __name__ == '__main__':
    main()
