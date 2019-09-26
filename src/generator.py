from .utils import * 
from .transformer import Transformer
from .graph.anchor_graph import AnchorGraph
from numpy.random import randint
from queue import PriorityQueue
from copy import deepcopy
from .storage import Storage

class Generator():
    def __init__(self, in_shape, out_shape, storage):
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.transformer = Transformer()
        self.storage = storage

        self.model_num = 0

    @property
    def level(self):
        return len(self.transformer.graph.anchors)

    @property
    def convs(self):
        return len(self.transformer.graph.convs)

    @property
    def graph(self):
        return self.transformer.graph

    def set_target(self, graph=None):
        if graph == None:
            self.transformer.graph = AnchorGraph(self.in_shape, self.out_shape)
        else:
            self.transformer.graph = graph

    def save_graph(self):
        model_num = self.model_num
        
        self.storage.save(self.transformer.graph, model_num)

        self.model_num += 1
        return model_num

    def load_graph(self, model_num):
        return self.storage.load(model_num)

    def generate(self, model_num=-1):
        if model_num < 0:
            self.set_target()
            self.transformer.add_init_block()
            ret = self.save_graph()
            return ret, ['added init block'] 
        else:
            graph = self.load_graph(model_num)
            self.set_target(graph)

        history = []
        if try_deeper(self.graph.variable):
            anc = graph.anchors[randint(self.level) + 1]
            self.transformer.insert_anchor_block(anc, method='cat')
            history.append(f'deeper layer on {anc}')

        for conv in deepcopy(self.graph.convs):
            nf = self.graph.id_to_node(conv).shape[0]
            if try_wider(nf):
                history.append(f'conv wider on {conv}')
                self.transformer.wider_conv(conv)
            if try_skip(len(self.graph.next_node(conv))):
                nl = layer_nums() - 1
                skip = skip_layers()
                anc = skip + self.graph.id_to_node(conv).rank
                if anc <= self.level:
                    anc = self.graph.anchors[anc]
                    history.append(f'skip con from {conv} to {anc}')
                    self.transformer.add_skip_connection(conv, anc, [64] * nl)
        ret = self.save_graph()
        return ret, history
        

