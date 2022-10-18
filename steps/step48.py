if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

import dezero


def main():
    x, t = dezero.datasets.get_spiral(train=True)
    print(x.shape)
    print(t.shape)
    print(x[10], t[10])
    print(x[110], t[110])

    markers = ['o', 'x', '^']
    colors = ['orange', 'blue', 'green']
    for (xy), c in zip(x, t):
        plt.scatter(xy[0], xy[1], s=40, marker=markers[c], c=colors[c])

    plt.savefig('spiral_dataset.jpg')


if __name__ == '__main__':
    main()
