from .graph import Graph

class NodeType():
  def __init__(self, label):
    self._label = label

class EdgeType():
  def __init__(self, weight):
    self._weight = weight

def test_add_nodes():
  gr = Graph(NodeType, EdgeType)
  nodes = []
  labels = ['a', 'b', 'c', 'd', 'e',
            'f', 'g', 'h', 'i', 'j']
  for label in labels:
    node = gr.add_node(label)
    nodes.append(node)

  assert len(nodes) == 10
  
  for node in nodes:
    assert gr.next_node(node) == {}
    assert gr.prev_node(node) == {}

def test_add_edges():
  gr = Graph(NodeType, EdgeType)
  nodes = []
  labels = ['a', 'b', 'c', 'd', 'e',
            'f', 'g', 'h', 'i', 'j']
  for label in labels:
    node = gr.add_node(label)
    nodes.append(node)
  
  for i in range(9):
    for j in range(i):
      eid = gr.add_edge(10*j+i)
      gr.redirect_edge(eid, j, i)
