import numpy as np

from dezero.core import Function, as_variable
from dezero import utils


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        return transpose(gy)


def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)
