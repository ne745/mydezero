
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np

from dezero import Variable
import dezero.functions as F


def main():
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    iters = 3
    for _ in range(iters):
        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    labels = ['y=sin(x)', "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
