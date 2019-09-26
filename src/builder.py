from torch import nn
from .nn.layer import Add, Cat

class GeneratedModule(nn.Module):
    def __init__(self, input, output, order):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.input = input
        self.output = output
        self.calc_order = []

        index = 0
        for prevs, layer, nid in order:
            self.layers[str(index)] = layer
            self.calc_order.append((prevs, str(index), nid))
            index += 1

    def forward(self, x):
        value = {self.input: x}
        for prevs, lid, nid in self.calc_order:
            inputs = [value[i] for i in prevs]
            value[nid] = self.layers[lid](*inputs)
        return value[self.output]
        
class ModelBuilder():
    def __init__(self, graph=None):
        self.graph = graph

    def set_graph(self, graph):
        self.graph = graph
        return self

    def layer_of(self, eid):
        return self.graph.id_to_edge(eid).layer.layer

    def reverse_dfs(self, target, visited):
        visited[target] = True
        ret = list()
        prevs = self.graph.prev_node(target)
        method = self.graph.id_to_node(target).method

        for nid, eid in prevs.items():
            if not visited[nid]:
                ret.extend(self.reverse_dfs(nid, visited))
            if method == 'none':
                ret.append(([nid], self.layer_of(eid), target))

        if method != 'none':
            if method == 'add':
                ret.append(([k for k in prevs], Add(), target))
            elif method =='cat':
                ret.append(([k for k in prevs], Cat(), target))
        return ret

    def top_sort(self):
        visited = dict()
        for key in self.graph._nodes.keys():
            visited[key] = False
        return self.reverse_dfs(self.graph.output, visited)

    def build(self):
        order = self.top_sort()
        input = self.graph.input
        output = self.graph.output
        
        return GeneratedModule(input, output, order)
