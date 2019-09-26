from .layer import *
from numpy.random import randint

def test_batchnorm2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 32, 32)
    l = BatchNorm2dLayer(num_features=16)
    out1, out2 = l.expand_in((32, 32, 32), out_shape, ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

    l = BatchNorm2dLayer(num_features=16)
    out1, out2 = l.expand_out(in_shape, (32, 32, 32), ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

def test_conv2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 32, 32)

    l = Conv2dLayer(16, 16, kernel_size=3)
    out1, out2 = l.expand_in((32, 32, 32), out_shape, ex)
    assert out1 == out_shape
    assert out2 == []

    l = Conv2dLayer(8, 16, kernel_size=3)
    out1, out2 = l.expand_out(in_shape, (32, 32, 32), ex)
    assert out1 == in_shape
    assert out2 == []

def test_linear2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16,)
    out_shape = (32,)

    l = LinearLayer(16, 32)
    out1, out2 = l.expand_in((32,), (32,))
    assert out1 == out_shape
    assert out2 == []

def test_maxpool2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 16, 16)

    l = MaxPool2dLayer(kernel_size=2)
    out1, out2 = l.expand_in((32, 32, 32), out_shape, expanded=ex)
    assert out1 == (32, 16, 16)
    assert out2 == ex

    l = MaxPool2dLayer(kernel_size=2)
    out1, out2 = l.expand_out(in_shape, (32, 16, 16), expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

def test_avgpool2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 16, 16)

    l = AvgPool2dLayer(kernel_size=2)
    out1, out2 = l.expand_in((32, 32, 32), out_shape, expanded=ex)
    assert out1 == (32, 16, 16)
    assert out2 == ex

    l = AvgPool2dLayer(kernel_size=2)
    out1, out2 = l.expand_out(in_shape, (32, 16, 16), expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

def test_adaptivemaxpool2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 16, 16)
    out_shape = (16, 1, 1)

    l = AdaptiveMaxPool2dLayer((1, 1))
    out1, out2 = l.expand_in((32, 16, 16), out_shape, ex)
    assert out1 == (32, 1, 1)
    assert out2 == ex

    l = AdaptiveMaxPool2dLayer((1, 1))
    out1, out2 = l.expand_out(in_shape, (32, 1, 1), expanded=ex)
    assert out1 == (32, 16, 16)
    assert out2 == ex

def test_adaptiveavgpool2d_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 16, 16)
    out_shape = (16, 1, 1)

    l = AdaptiveAvgPool2dLayer((1, 1))
    out1, out2 = l.expand_in((32, 16, 16), out_shape, ex)
    assert out1 == (32, 1, 1)
    assert out2 == ex

    l = AdaptiveAvgPool2dLayer((1, 1))
    out1, out2 = l.expand_out(in_shape, (32, 1, 1), expanded=ex)
    assert out1 == (32, 16, 16)
    assert out2 == ex


def test_relu_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 32, 32)

    l = ReluLayer()
    out1, out2 = l.expand_in((32, 32, 32), out_shape, expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

    l = ReluLayer()
    out1, out2 = l.expand_out(in_shape, (32, 32, 32), expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

def test_dropout_layer():
    ex = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
    in_shape = (16, 32, 32)
    out_shape = (16, 32, 32)

    l = DropoutLayer()
    out1, out2 = l.expand_in((32, 32, 32), out_shape, expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex

    l = DropoutLayer()
    out1, out2 = l.expand_out(in_shape, (32, 32, 32), expanded=ex)
    assert out1 == (32, 32, 32)
    assert out2 == ex


