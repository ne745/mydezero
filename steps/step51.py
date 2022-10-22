if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

import dezero


def main():
    train_set = dezero.datasets.MNIST(train=True, transform=None)
    test_set = dezero.datasets.MNIST(train=False, transform=None)

    print(len(train_set))
    print(len(test_set))

    x, t = train_set[0]
    print(type(x), x.shape)
    print(t)

    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
