from collections import OrderedDict
from graphviz import Digraph 

class Graph():
    def __init__(self, node_class, edge_class):
        self._nodes = dict()
        self._node_id = 0
        self._edges = dict()
        self._edge_id = 0
        
        self._node_class = node_class
        self._edge_class = edge_class
        
        self._adjacent_fwd = OrderedDict()
        self._adjacent_bwd = OrderedDict()

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
        self._adjacent_fwd[nid] = OrderedDict()
        self._adjacent_bwd[nid] = OrderedDict()
        
        return nid
    
    def add_edge(self, *args, **kwargs):
        eid = self._edge_id
        self._edge_id = self._edge_id + 1
        self._edges[eid] = self.Edge(*args, **kwargs)

        return eid

    def redirect_edge(self, eid, src, dest):
        if not eid in self._edges:
            raise Exception(f'Cannot find edge {eid}')
        try:

            self._adjacent_fwd[src][dest] = eid
            self._adjacent_bwd[dest][src] = eid
        except KeyError as e:
            print(f'Cannot find nid. {str(e)}')

    def remove_edge(self, src, dest):
        if self.is_connected(src, dest):
            del self._adjacent_fwd[src][dest]
            del self._adjacent_bwd[dest][src]

    def visualize(self, path, filename):
        digraph = Digraph(comment="Model")
        for nid, node in self._nodes.items():
            node = self.id_to_node(nid)
            digraph.node(str(nid), label=f'{nid}: {node}')
        for prev, next_list in self._adjacent_fwd.items():
            for next in next_list:
                eid = self._adjacent_fwd[prev][next]
                edge = self.id_to_edge(eid)
                digraph.edge(str(prev), str(next), label=f'{eid}: {edge}')
        
        digraph.format='svg'
        digraph.filename=filename
        digraph.directory=path
        digraph.render(view=False)

    @property
    def Node(self):
        return self._node_class
    
    @property
    def Edge(self):
        return self._edge_class
