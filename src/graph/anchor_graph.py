from .graph import Graph

class Node():
  def __init__(self, layer):
    self._layer = layer
  
  def updated_src(self, in_shape, out_shape, extended=[]):
    return self._layer.updated_src(in_shape, out_shape, extended)

  def updated_dest(self, in_shape, out_shape, extended=[]):
    return self._layer.updated_dest(in_shape, out_shape, extended)

  def verify_output(self, in_shape, out_shape):
    return self._layer.verify_output(in_shape, out_shape)

  @property
  def layer(self):
    return self._layer

class Edge():
  def __init__(self, shape):
    self._shape = shpae

  def set_shape(self, shape):
    self._shape = shape

  @property
  def shape(self):
    return self._shape

class AnchorGraph(Graph):
  def __init__(self, in_shape, out_shape):
    super(AnchorGraph, self).__init__(Node, Edge)
    self._input = self.add_node( 

  def add_node(self, layer):
    return super(AnchorGraph, self).add_node(layer)

  def add_edge(self, src, dest, shape):
    return super(AnchorGraph, self).add_edge(src, dest, shape)

  def _rev_traversal(self, target, visited):
    visited[target] = True
    ret = []
    for key in self.prev_node(target).keys():
      if not visited[node]:
        ret.extend(self._rev_traversal(key))
    ret.append(visited)
    return ret

  def top_sort(self):
    
    
