from .layer import *
from numpy.random import randint

def test_batchnorm2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 16, 16)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = BatchNorm2dLayer(num_features=16)
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded
  
  l = BatchNorm2dLayer(num_features=16)
  out1, out2 = l.updated_dest(in_shape, (32, 16, 16), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded

def test_conv2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (8, 16, 16)
  ni = in_shape[0]
  nf = out_shape[0]

  l = Conv2dLayer(ni, nf, kernel_size=3)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (8, 16, 16)
  assert out2 == []

  l = Conv2dLayer(ni, nf , kernel_size=3)
  expanded = [(o, c) for o, c in zip(randint(8, size=8), range(8, 16))]
  out1, out2 = l.updated_dest(in_shape, (16, 16, 16), expanded=expanded)
  assert out1 == (16, 16, 16)
  assert out2 == []

def test_linear_layer():
  in_shape = (16,)
  out_shape = (32,)

  l = LinearLayer(in_shape[0], out_shape[0])
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  out1, out2 = l.updated_src((32,), out_shape, expanded=expanded)
  assert out1 == (32,)
  assert out2 == []

def test_maxpool2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 8, 8)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = MaxPool2dLayer(kernel_size=2)
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 8, 8)
  assert out2 == expanded
  
  l = MaxPool2dLayer(kernel_size=2)
  out1, out2 = l.updated_dest(in_shape, (32, 8, 8), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded


def test_avgpool2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 8, 8)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = AvgPool2dLayer(kernel_size=2)
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 8, 8)
  assert out2 == expanded
  
  l = AvgPool2dLayer(kernel_size=2)
  out1, out2 = l.updated_dest(in_shape, (32, 8, 8), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded

def test_adaptiveavgpool2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 1, 1)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = AdaptiveAvgPool2dLayer((1, 1))
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 1, 1)
  assert out2 == expanded
  
  l = AdaptiveAvgPool2dLayer((1,1))
  out1, out2 = l.updated_dest(in_shape, (32, 1, 1), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded

def test_adaptivemaxpool2d_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 1, 1)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = AdaptiveMaxPool2dLayer((1, 1))
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 1, 1)
  assert out2 == expanded
  
  l = AdaptiveMaxPool2dLayer((1,1))
  out1, out2 = l.updated_dest(in_shape, (32, 1, 1), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded

def test_relu_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 16, 16)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = ReluLayer()
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded
  
  l = ReluLayer()
  out1, out2 = l.updated_dest(in_shape, (32, 16, 16), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded

def test_dropout_layer():
  in_shape = (16, 16, 16)
  out_shape = (16, 16, 16)
  expanded = [(o, c) for o, c in zip(randint(16, size=16), range(16, 32))]
  
  l = DropoutLayer()
  out1, out2 = l.updated_src((32, 16, 16), out_shape, expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded
  
  l = DropoutLayer()
  out1, out2 = l.updated_dest(in_shape, (32, 16, 16), expanded=expanded)
  assert out1 == (32, 16, 16)
  assert out2 == expanded


# def test_flatten_layer()
