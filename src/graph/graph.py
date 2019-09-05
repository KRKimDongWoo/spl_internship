class Graph():
  def __init__(self, node_class, edge_class):
    self._nodes = dict()
    self._node_id = 0
    self._edges = dict()
    self._edge_id = 0

    self._node_class = node_class
    self._edge_class = edge_class

    self._adjacent_fwd = dict()
    self._adjacent_bwd = dict()

  def id_to_node(self, nid):
    return self._nodes[nid]

  def id_to_edge(self, eid):
    return self._edges[eid]

  def next_node(self, nid):
    return self._adjacent_fwd[nid]

  def prev_node(self, nid):
    return self._adjacent_bwd[nid]

  def is_connected(self, src, dest):
    if dest in self.next_node(src): return True
    else: return False

  def add_node(self, *args, **kwargs):
    nid = self._node_id
    self._node_id = self._node_id + 1

    self._nodes[nid] = self.Node(*args, **kwargs)
    self._adjacent_fwd[nid] = dict()
    self._adjacent_bwd[nid] = dict()

    return nid

  def add_edge(self, src, dest, *args, **kwargs):
    eid = self._edge_id
    self._edge_id = self._edge_id + 1

    self._edges[eid] = self.Edge(*args, **kwargs)

    if self.is_connected(src, dest):
      raise Exception(f'Connection alreadyy exists: {src} to {dest}')
    else:
      self._adjacent_fwd[src][dest] = eid
      self._adjacent_bwd[dest][src] = eid

    return eid

  def remove_edge(self, src, dest):
    if self.is_connected(src, dest):
      del self._adajcent_fwd[src][dest]
      del self._adjacent_bwd[dest][src]

  @property
  def Node(self):
    return self._node_class

  @property
  def Edge(self):
    return self._edge_class
