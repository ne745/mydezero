from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        self.grad = None
        self.__creator = None

    @property
    def creator(self):
        return self.__creator

    @creator.setter
    def creator(self, func: Function):
        self.__creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.__creator]
        while funcs:
            f = funcs.pop()  # 1. 関数を取得
            x, y = f.input, f.output  # 2. 関数の入出力を取得
            x.grad = f.backward(y.grad)  # 3. backward メソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)  # 1 つ前の関数をリストに追加


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function(ABC):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.creator = self
        self.input = input
        self.output = output
        return output

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


def main():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
