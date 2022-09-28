from __future__ import annotations
from abc import ABC, abstractmethod
import unittest
import weakref

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        self.grad = None
        self.__creator = None
        self.__generation = 0

    @property
    def creator(self):
        return self.__creator

    @creator.setter
    def creator(self, func: Function):
        self.__creator = func
        self.generation = func.generation + 1

    @property
    def generation(self):
        return self.__generation

    @generation.setter
    def generation(self, generation):
        self.__generation = generation

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 1. 関数を取得
            gys = [output().grad for output in f.outputs]  # 2. 関数の出力を取得
            gxs = f.backward(*gys)  # 3. backward メソッドを呼ぶ
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)  # 1 つ前の関数をリストに追加


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

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.creator = self
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
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
        return gy, gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
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
    for i in range(10):
        x = Variable(np.random.randn(10000))
        _ = square(square(square(x)))


if __name__ == "__main__":
    main()
