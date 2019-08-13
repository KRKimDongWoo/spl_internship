

class Graph:
    def __init__(self, input_shape, output_shape):
        self.nodes={}
        self.edges={}
        self._node_index = 0
        self._edge_index = 0

        self.input = self.add_node(Node(input_shape))
        self.output = self.add_node(Node(output_shape))


    def add_node(self, node):
        self._node_index = self._node_index + 1
        self.nodes[self._node_index] = node
        return self._node_index

    def add_edge(self, src, dest, edge):
        self._edge_index = self._edge_index + 1
        self.edges[self._edge_index] = edge

        self.nodes[src].add_out_edge(self._edge_index)
        self.nodes[dest].add_in_edge(self._edge_index)

        return self._edge_index

    def add_conv_edge(self, src, dest, ks=3):
        input_shape = self.nodes[src].shape
        output_shape = self.nodes[dest].shape

        ni, nf = input_shape[0], output_shape[0]
        pd = ks//2
        st = ceil(input_shape[1] / output_shape[1])

        layers = [
            torch.nn.Conv2d(ni, nf, ks, st, pd),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(nf)
        ]

        edge = Edge(src, dest, torch.nn.Sequential(*layers))
        return self.add_edge(src, dest, edge)

    def add_linear_edge(self, src, dest):
        ni = self.nodes[src].shape[0]
        no = self.nodes[dest].shape[0]

        edge = Edge(src, dest, torch.nn.Linear(ni, no))
        return self.add_edge(src, dest, edge)

    def add_flatten_edge(self, src, dest):
        edge = Edge(src, dest, Flatten())
        return self.add_edge(src, dest, edge)

    def add_identical_edge(self, src, dest):
        return

    def _node_as_layer(self, id):
        node = self.nodes[id]
        layer = node.layer
        inputs = []

        for edge_id in node.in_edge:
            inputs.append(self.edges[edge_id].src)

        return (layer, inputs, id)



    def _reverse_traversal(self, id, visited):
        ts = []
        visited[id] = True
        for edge_id in self.nodes[id].in_edge:
            edge = self.edges[edge_id]
            if visited[edge.src]: ts.extend(edge_id)
            else: ts.extend(self._reverse_traversal(edge.src, visited))

            if not edge.identical: ts.append((edge.as_layer(), edge.src, edge.dest))
        if self.nodes[id].multi_input:
            ts.append((self._node_as_layer(id)))
        return ts


    def generate_model(self):
        visited = {}
        for key in self.nodes.keys():
            visited[key] = False

        ts = self._reverse_traversal(2, visited)
        return Generator(ts)

#         it = 1
#         module = []
#         while(it != 2):
#             print('current', it)
#             node = self.nodes[it]
#             print(node.out_edge)
#             if(len(node.out_edge) == 1):
#                 edge = self.edges[node.out_edge[0]]
#                 module.append(edge.as_layer())
#                 it = edge.dest
#             else:
#                 (submodule, it) = self._generate_submodel(node)

#         return torch.nn.Sequential(*module)


