from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Variable:
    def __init__(self, data):
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
        f = self.__creator  # 1. 関数を取得
        if f is not None:
            x = f.input  # 2. 関数の入力を取得
            x.grad = f.backward(self.grad)  # 3. 関数の backward メソッドを呼ぶ
            x.backward()  # 自分より 1 つ前の変数の backward メソッドを呼ぶ (再帰)


class Function(ABC):
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def main():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆向きに計算グラフのノードをたどる
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    print('-' * 50)
    print('バックプロパゲーション (自動再帰)')
    x = Variable(np.array(0.5))
    y = C(B(A(x)))
    # 逆伝播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    print('-' * 50)
    print('バックプロパゲーション (自動)')
    y.grad = np.array(1.0)
    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad)

    print('-' * 50)
    print('バックプロパゲーション (手動)')
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)

    print('-' * 50)
    print('数値微分')
    x = Variable(np.array(0.5))
    dy = numerical_diff(lambda x: C(B(A(x))), x)
    print(dy)


if __name__ == "__main__":
    main()
