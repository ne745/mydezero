from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function(ABC):
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def main():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)


if __name__ == "__main__":
    main()
