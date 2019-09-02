from torch import nn, cat, Tensor
import numpy as np
from math import ceil

from ..constants import NOISE_RATIO, COPY_RATIO

def add_noise(weight):
  noise_range = np.ptp(weight.flatten()) * NOISE_RATIO
  noise = np.random.uniform(-noise_range/2.0, noise_range/2.0, weight.shape)
  return np.add(weight, noise).float()

def ceil_power_of_two(x):
  x = ceil(x) - 1
  res = 0
  while x > 0:
    x = x // 2
    res = res + 1
    
  return 2**(res)

def adjust_stride(in_shape, out_shape):
  params = zip(in_shape[1:], out_shape[1:])
  return tuple(ceil_power_of_two(i/o) for i, o in params)

def get_layer_parameters(layer):
  gen = layer.named_parameters()
  return dict((name, value.data) for (name, value) in gen)

class StubLayer():
  def __init__(self):
    pass

  def updated_src(self, in_shape, out_shape, expanded):
    pass

  def updated_dest(self, in_shape, out_shape, expanded):
    pass

  def calculate_output(self, in_shape):
    return in_shape

  def verify_output(self, in_shape, out_shape):
    expected_out = self.calculate_output(in_shape)
    if expected_out == None: return False
    elif len(expected_out) != len(out_shape): return False
    else:
      for e, o in zip(expected_out, out_shape):
        if e <= 0 or o <= 0: continue
        elif e != o: return False

  def set_weight(self, **kwargs):
    if bias in kwargs:
      self.layer.bias.data = kwargs['data']
    if weight in kwargs:
      self.layer.weight.data = kwargs['weight']

  @property
  def layer_info(self):
    return {
      'layer': self.layer_name,
      'args': self.args,
      'parmeters': dict(self.parameter)
    }

  @property
  def parameter(self):
    return self.layer.named_parameters()

  @property
  def layer_name(self):
    pass

  @property
  def layer(self):
    return self._layer

class BatchNormLayer(StubLayer):
  def __init__(self, num_features, **kwargs):
    self.args = {
      'num_features': num_features
    }
    self._layer = self.layer_class(**self.args)

  def set_zeros(self):
    nn.init.constant_(self.layer.weight, 0)

  def set_identical(self):
    self.layer.reset_running_stats()
    self.layer.reset_parameters()
    nn.init.constant_(self.layer.bias, 0)
    nn.init.uniform_(self.layer.weight, a=1.0-NOISE_RATIO, b=1.0+NOISE_RATIO)

  def updated_src(self, in_shape, out_shape, expanded=[]):
    ni = in_shape[0] # input channels
    new_params = {
      'weight': Tensor(ni),
      'bias': Tensor(ni)
    }

    # Get original parameters.
    params = get_layer_parameters(self.layer) 

    # Expand original parameters
    for key in params.keys():
      new_params[key][:len(params[key])] = params[key]
      for o, c in expanded:
        if o < 0:
          new_params[key][c].fill_(0.)
        else:
          new_params[key][c] = params[key][o]
      new_params[key] = add_noise(new_params[key])

    # Update argument
    self.args['num_features'] = ni

    new_layer = self.layer_class(**self.args)
    new_layer.weight.data = new_params['weight']
    new_layer.bias.data = new_params['bias']
    self._layer = new_layer

    out_shape = in_shape
    return out_shape, expanded

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    nf = out_shape[0] # output channels
    new_params = {
      'weight': Tensor(nf),
      'bias': Tensor(nf)
    }
    
    # Get original parameters.
    params = dict((name, value.data) for (name, value) in self.layer.named_parameters())
    
    # Expand original parameters
    for key in params.keys():
      new_params[key][:len(params[key])] = params[key]
      for o, c in expanded:
        if o < 0:
          new_params[key][c].fill_(0.)
        else:
          new_params[key][c] = params[key][o]
          new_params[key] = add_noise(new_params[key])

    # Update argument
    self.args['num_features'] = nf
    
    # Update the layer
    new_layer = self.layer_class(**self.args)
    new_layer.weight.data = new_params['weight']
    new_layer.bias.data = new_params['bias']
    self._layer = new_layer

    # Generate new input shape
    in_shape = out_shape
    return in_shape, expanded

  def calculate_output(self, in_shape):
    if in_shape[0] != self.args['num_features']: return None
    else: return in_shape

  @property
  def layer_class(self):
    pass

  @property
  def n_dim(self):
    pass

  @property
  def layer_name(self):
    return 'batchnorm'

class BatchNorm2dLayer(BatchNormLayer):
  @property
  def layer_class(self):
    return nn.BatchNorm2d

  @property
  def n_dim(self):
    return 2

  @property
  def layer_name(self):
    return 'batchnorm2d'

class ConvLayer(StubLayer):
  def __init__(self, in_channels, out_channels, kernel_size, 
               stride=1, bias=True, **kwargs):
    if isinstance(kernel_size, int): kernel_size = (kernel_size,) * self.n_dim
    padding = tuple(ks//2 for ks in kernel_size)
    if isinstance(stride, int): stride = (stride,) * self.n_dim

    self.args = {
      'in_channels': in_channels,
      'out_channels': out_channels,
      'kernel_size': kernel_size,
      'stride': stride,
      'padding': padding,
      'bias': bias
    }
    self._layer = self.layer_class(**self.args)

  def set_identical(self):
    if self.args['in_channels'] != self.args['out_channels']:
      raise Exception('Cannot set identical weight')
    center = tuple(i//2 for i in self.args['kernel_size'])
    nf = self.args['in_channels']

    nn.init.constant_(self.layer.weight, 0.)
    if self.args['bias']: nn.init.constant_(self.layer.bias, 0.)
    for i in range(nf):
      self.layer.weight.data[i, i, center[0], center[1]] = 1.

    self.layer.weight.data = add_noise(self.layer.weight.data)

  def updated_src(self, in_shape, out_shape, expanded=[]):
    ni = in_shape[0] # input channels
    prev_ni = self.args['in_channels']
    nf = self.args['out_channels']

    # Update the args
    self.args['in_channels'] = ni

    # Adjust stride
    exp_shape = self.calculate_output(in_shape)
    if exp_shape[1:] != out_shape[1:]:
      stride = adjust_stride(in_shape, out_shape)
      self.args['stride'] = stride

    # Get original parameters
    params = get_layer_parameters(self.layer)

    # Expanded original parameters
    new_params = {}
    new_params['weight'] = Tensor(nf, ni, *self.args['kernel_size'])
    new_params['weight'][:, :prev_ni, ...] = params['weight']
    for o, c in expanded:
      if o < 0:
        new_params['weight'][:, c, ...].fill_(0.)
      else:
        new_params['weight'][:, c, ...] = params['weight'][:, o, ...]
    new_params['weight'] = add_noise(new_params['weight'])

    # Update the layer
    new_layer = self.layer_class(**self.args)
    new_layer.weight.data = new_params['weight']
    if self.args['bias']: new_layer.bias.data = params['bias']
    self._layer = new_layer

    # Generate new outpu shape
    out_shape = self.calculate_output(in_shape)
    return out_shape, []
  
  def updated_dest(self, in_shape, out_shape, expanded=[]):
    ni = self.args['in_channels']
    nf = out_shape[0] if out_shape[0] > 0 else self.args['out_channels']
    prev_nf = self.args['out_channels']

    # Update the args
    self.args['out_channels'] = nf

    # Adjust stride
    exp_shape = self.calculate_output(in_shape)
    if exp_shape[1:] != out_shape[1:]:
      stride = adjust_stride(in_shape, out_shape)
      self.args['stride'] = stride

    # Get original parameters
    params = get_layer_parameters(self.layer)

    # Expand original parameters
    new_params = {}
    new_params['weight'] = Tensor(nf, ni, *self.args['kernel_size'])
    if self.args['bias']: new_params['bias'] = Tensor(nf)

    for key in params.keys():
      new_params[key][:prev_nf, ...] = params[key]
      for o, c in expanded:
        if o < 0:
          new_params[key][c, ...].fill_(0.)
        else:
          new_params[key][c, ...] = params[key][o, ...] * COPY_RATIO
          new_params[key][o, ...] = params[key][o, ...] * (1 - COPY_RATIO)
      new_params[key] = add_noise(new_params[key])

    # Update the layer
    new_layer = self.layer_class(**self.args)
    new_layer.weight.data = new_params['weight']
    if self.args['bias']: new_layer.bias.data = new_params['bias']
    self._layer = new_layer

    # Generate new input shape
    in_shape = (ni,) + in_shape[1:]
    return in_shape, []
  
  def calculate_output(self, in_shape):
    if in_shape[0] != self.args['in_channels']: return None
    nf = self.args['out_channels']
    attr = zip(in_shape[1:],
               self.args['kernel_size'],
               self.args['stride'],
               self.args['padding'])
    ch = tuple((x+2*pd-ks)//st+1 for x, ks, st, pd in attr)
    out_shape = (nf,) + ch
    return out_shape

  @property
  def layer_class(self):
    pass

  @property
  def n_dim(self):
    pass

  @property
  def layer_name(self):
    return 'conv'

class Conv2dLayer(ConvLayer):
  @property
  def layer_class(self):
    return nn.Conv2d

  @property
  def n_dim(self):
    return 2  
  
  @property
  def layer_name(self):
    return 'conv2d'

class LinearLayer(StubLayer):
  def __init__(self, in_features, out_features, 
               bias=True, **kwargs):
    self.args = {
      'in_features': in_features,
      'out_features': out_features,
      'bias': bias
    }
    self._layer = nn.Linear(**self.args)

  def set_identical(self):
    if self.args['in_features'] != self.args['out_features']:
      raise Exception('Cannot set identical weight')
    nn.init.constant_(self.layer.weight.data, 0.)
    nn.init.constant_(self.layer.bias.data, 0.)

    ni = self.args['in_features']
    for i in range(ni):
      self.layer.weight.data[i, i] = 1.

    self.layer.weight.data = add_noise(self.layer.weight.data)

  def updated_src(self, in_shape, out_shape, expanded=[]):
    ni = in_shape[0]
    prev_ni = self.args['in_features']

    # Update the args
    self.args['in_features'] = ni

    # Get original parameters
    params = get_layer_parameters(self.layer)

    # Expand original parameters
    new_params = {}
    new_params['weight'] = Tensor(self.args['out_features'], ni)
    new_params['weight'][:,:prev_ni] = params['weight']
    for o, c in expanded:
      if o < 0:
        new_params['weight'][:, c].fill_(0.)
      else:
        new_params['weight'][:, c] = params['weight'][:, o]
    new_params['weight'] = add_noise(new_params['weight'])

    # Update the layer
    new_layer = nn.Linear(**self.args)
    new_layer.weight.data = new_params['weight']
    new_layer.bias.data = params['bias']
    self._layer = new_layer

    # Generate new output shape
    out_shape = self.calculate_output(in_shape)
    return out_shape, []

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    raise Exception('Unimplemented')

  def calculate_output(self, in_shape):
    if in_shape[0] != self.args['in_features']: return None
    return (self.args['out_features'],)

  @property
  def layer_name(self):
    return 'linear'

class PoolLayer(StubLayer):
  def __init__(self, kernel_size, **kwargs):
    if isinstance(kernel_size, int): kernel_size = (kernel_size,) * self.n_dim
    stride = kernel_size
    padding = (0,) * self.n_dim
    self.args = {
      'kernel_size': kernel_size,
      'stride': stride,
      'padding': padding,
      'ceil_mode': True
    }

    self._layer = self.layer_class(**self.args)

  def updated_src(self, in_shape, out_shape, expanded=[]):
    out_shape = self.calculate_output(in_shape)
    return out_shape, expanded

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    exp_shape = self.calculate_output(in_shape)
    if(out_shape[1:] != exp_shape[1:]):
      stride = adjust_stride(in_shape, out_shape)
      kernel_size = stride
      self.args['kernel_size'] = kernel_size
      self.args['stride'] = stride
      
      self._layer = self.layer_class(**self.args)

    in_shape = (out_shape[0],) + in_shape[1:]

    return in_shape, expanded

  def calculate_output(self, in_shape):
    nf = in_shape[0]
    attr = zip(in_shape[1:],
               self.args['kernel_size'],
               self.args['stride'],
               self.args['padding'])
    ch = tuple(ceil((x+2*pd-ks)/st)+1 for x, ks, st, pd in attr)
    out_shape = (nf,) + ch
    return out_shape

  @property
  def layer_class(self):
    pass

  @property
  def n_dim(self):
    pass

  @property
  def layer_name(self):
    return 'pool'

class MaxPool2dLayer(PoolLayer):
  @property
  def layer_class(self):
    return nn.MaxPool2d

  @property
  def n_dim(self):
    return 2

  @property
  def layer_name(self):
    return 'maxpool2d'

class AvgPool2dLayer(PoolLayer):
  @property
  def layer_class(self):
    return nn.AvgPool2d

  @property
  def n_dim(self):
    return 2

  @property
  def layer_name(self):
    return 'avgpool2d'

class AdaptivePoolLayer(StubLayer):
  def __init__(self, output_size, **kwargs):
    self.args = {
      'output_size': output_size
    }
    self._layer = self.layer_class(**self.args)

  def updated_src(self, in_shape, out_shape, expanded=[]):
    return self.calculate_output(in_shape), expanded

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    return (out_shape[0],) + in_shape[1:], expanded

  def calculate_output(self, in_shape):
    return (in_shape[0],) + self.args['output_size']

  @property
  def layer_class(self):
    pass

class AdaptiveAvgPool2dLayer(AdaptivePoolLayer):
  @property
  def layer_class(self):
    return nn.AdaptiveAvgPool2d

  @property
  def n_dim(self):
    return 2

  @property
  def layer_name(self):
    return 'adaptiveavgpool2d'

class AdaptiveMaxPool2dLayer(AdaptivePoolLayer):
  @property
  def layer_class(self):
    return nn.AdaptiveMaxPool2d

  @property
  def n_dim(self):
    return 2

  @property
  def layer_name(self):
    return 'adaptivemaxpool2d'

class ReluLayer(StubLayer):
  def __init__(self, **kwargs):
    self.args = {}
    self._layer = nn.ReLU()
  
  def updated_src(self, in_shape, out_shape, expanded=[]):
    return in_shape, expanded

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    return out_shape, expanded

  @property
  def layer_name(self):
    return 'relu'

class DropoutLayer(StubLayer):
  def __init__(self, p=0.5, **kwargs):
    self.args = {
      'p': p
    }
    self._layer = nn.Dropout(**self.args)
  
  def updated_src(self, in_shape, out_shape, expanded=[]):
    return in_shape, expanded

  def updated_dest(self, in_shape, out_shape, expanded=[]):
    return out_shape, expanded

  @property
  def layer_name(self):
    return 'dropout'

class CatLayer(StubLayer):
  class Cat(nn.Module):
    def forward(self, x):
      if isinstance(x, Tensor): return x
      elif isinstance(x, list): return cat(x, 1)

class AddLayer(StubLayer):
  class Add(nn.Module):
    def forward(self, x):
      if isinstance(x, Tensor): return x
      elif isinstance(x, list): return sum(x)

class FlattenLayer(StubLayer):
  class Flatten(nn.Module):
    def forward(self, x):
      return x.view(x.size(0), -1)

  def __init__(self, **kwargs):
    self.args = {}
    self._layer = self.Flatten()

  def updated_src(self, in_shape, out_shape, expanded=[]):
    out_shape = self.calculate_output(in_shape)
    feature_size = out_shape[0] // in_shape[0]

    new_expanded = []
    for o, c in expanded:
      if o < 0:
        origin = (-1,) * feature_size
      else:
        origin = range(o*feature_size, (o+1) * feature_size)
      copied = range(c*feature_size, (c+1)*feature_size)
      new_expanded.extend((o, c) for o, c in zip(origin, copied))

    return out_shape, new_expanded

  def calculate_output(self, in_shape):
    total = 1
    nf = in_shape[0]
    for i in in_shape:
      total = total * i
    out_shape = (total,)
    return out_shape

  @property
  def layer_name(self):
    return 'flatten'
