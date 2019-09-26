import os
from pickle import dump, load
from .graph.anchor_graph import AnchorGraph
from .nn.layer import *

layer_table = {
    'none': NoneLayer,
    'batchnorm': BatchNormLayer,
    'batchnorm2d': BatchNorm2dLayer,
    'conv': ConvLayer,
    'conv2d': Conv2dLayer,
    'linear': LinearLayer,
    'pool': PoolLayer,
    'maxpool2d': MaxPool2dLayer,
    'avgpool2d': AvgPool2dLayer,
    'adaptivemaxpool2d': AdaptiveMaxPool2dLayer,
    'adaptiveavgpool2d': AdaptiveAvgPool2dLayer,
    'relu': ReluLayer,
    'dropout': DropoutLayer,
    'flatten': FlattenLayer
}

class Storage():
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def save(self, graph, model_num):
        model_path = os.path.join(self.path, str(model_num))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        graph_path = os.path.join(model_path, 'graph.pickle')
        layer_path = os.path.join(model_path, 'layer.pickle')

        graph_dict = dict()
        layer_dict = dict()
       
        graph_dict['input'] = graph.input
        graph_dict['output'] = graph.output
        graph_dict['in_shape'] = graph.in_shape
        graph_dict['out_shape'] = graph.out_shape
        
        graph_dict['nodes'] = dict()
        graph_dict['adjacent'] = graph._adjacent_fwd
        graph_dict['convs'] = graph.convs
        graph_dict['anchors'] = graph.anchors

        for nid, node in graph._nodes.items():
            if nid == graph.input or nid == graph.output: continue

            graph_dict['nodes'][nid] = node.dict

        for eid, edge in graph._edges.items():
            layer_dict[eid] = edge.dict

        with open(graph_path, 'wb+') as handle:
            dump(graph_dict, handle)
        
        with open(layer_path, 'wb+') as handle:
            dump(layer_dict, handle)

        return model_num 

    def load(self, model_num):
        model_path = os.path.join(self.path, str(model_num))
        graph_path = os.path.join(model_path, 'graph.pickle')
        layer_path = os.path.join(model_path, 'layer.pickle')

        with open(graph_path, 'rb') as handle:
            graph_dict = load(handle)

        with open(layer_path, 'rb') as handle:
            layer_dict = load(handle)

        key_to_nid = dict()
        key_to_eid = dict()

        graph = AnchorGraph(graph_dict['in_shape'], graph_dict['out_shape'])
        key_to_nid[graph_dict['input']] = graph.input
        key_to_nid[graph_dict['output']] = graph.output

        for key, props in graph_dict['nodes'].items():
            nid = graph.add_node(**props)
            key_to_nid[key] = nid

        graph.convs.extend(key_to_nid[k] for k in graph_dict['convs'])
        
        for rank, key in graph_dict['anchors'].items():
            graph.anchors[rank] = key_to_nid[key]

        for nid, nexts in graph_dict['adjacent'].items():
            for key, eid in nexts.items():
                src = key_to_nid[nid]
                dest = key_to_nid[key] 
                layer_info = layer_dict[eid]
                layer_class = layer_table[layer_info['layer']]
                layer = layer_class(**layer_info['args'])
                layer.set_weight(**layer_info['parameters'])

                eid = graph.add_edge(layer)
                graph.redirect_edge(eid, src, dest)

        return graph
