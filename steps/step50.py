if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import DataLoader
from dezero.datasets import Spiral


def main():
    max_epoch = 1
    batch_size = 10

    train_set = Spiral(train=True)
    test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    for epoch in range(max_epoch):
        for xy, t in train_loader:
            print(xy.shape, t.shape)
            break

        for xy, t in test_loader:
            print(xy.shape, t.shape)
            break


if __name__ == '__main__':
    main()
