from __future__ import annotations
from abc import ABC, abstractmethod
import unittest

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
    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.creator = self
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        pass


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


def add(x0, x1):
    return Add()(x0, x1)


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = exp(x)
        expected = np.array(np.exp(2.0))
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = exp(x)
        y.backward()
        expected = np.array(np.exp(3.0))
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


def main():
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)


if __name__ == "__main__":
    main()
