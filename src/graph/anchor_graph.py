from .graph import Graph
from ..transformer import Transformer
from ..nn.layer import *
from ..utils import partial

class Edge():
    def __init__(self, layer):
        self._layer = layer

    def expand_in(self, in_shape, out_shape, extended=[]):
        return self.layer.expand_in(in_shape, out_shape, extended)

    def expand_out(self, in_shape, out_shape, extended=[]):
        return self.layer.expand_out(in_shape, out_shape, extended)

    def verify_output(self, in_shape, out_shape):
        return self.layer.verify_output(in_shape, out_shape)

    def calculate_output(self, in_shape):
        return self.layer.calculate_output(in_shape)
    
    def __repr__(self):
        return self.layer.__repr__()

    @property
    def layer(self):
        return self._layer

    @property
    def dict(self):
        return self.layer.layer_info

class Node():
    def __init__(self, shape, method='none', rank=-1):
        self._shape = shape
        self._method = method
        self._rank = rank

    def set_shape(self, shape):
        self._shape = shape

    def set_rank(self, rank):
        self._rank = rank

    def __repr__(self):
        return f'{self.rank} - {self.shape}'

    @property
    def shape(self):
        return self._shape
    
    @property
    def rank(self):
        return self._rank

    @property
    def method(self):
        return self._method

    @property
    def dict(self):
        return {'shape': self.shape,
                'rank': self.rank,
                'method': self.method}

class AnchorGraph(Graph):
    def __init__(self, in_shape, out_shape):
        super(AnchorGraph, self).__init__(Node, Edge)

        self._in_shape = in_shape
        self._out_shape = out_shape
        self._input = self.add_node(in_shape, rank=0)
        self._output = self.add_node(out_shape)

        self.convs = []
        self.anchors = dict()

    def add_node(self, shape, method='none', rank=-1):
        return super(AnchorGraph, self).add_node(shape, method, rank)

    def add_edge(self, layer):
        return super(AnchorGraph, self).add_edge(layer)

    @property
    def variable(self):
        rank = len(self.anchors)
        conv = len(self.convs)
        return (rank + conv) * rank / 2 + conv

    @property
    def input(self):
        return self._input
    
    @property
    def output(self):
        return self._output

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._out_shape
