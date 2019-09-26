from .nn.layer import *
from .utils import next_channel
from copy import deepcopy
from random import randint

class Transformer():
    def __init__(self, graph=None):
        self.graph = graph

    def shape_of(self, nid):
        return self.graph.id_to_node(nid).shape

    def set_shape(self, nid, shape):
        self.graph.id_to_node(nid).set_shape(shape)

    def method_of(self, nid):
        return self.graph.id_to_node(nid).method

    def rank_of(self, nid):
        return self.graph.id_to_node(nid).rank

    def set_rank(self, nid, rank):
        self.graph.id_to_node(nid).set_rank(rank)

    def reconstruct(self, target, caller, expanded):
        # print(f'{target} reconstructing by {caller}')
        out_shape = self.shape_of(target)
        if self.method_of(target) == 'cat':
            exp_out = (-1,) + out_shape[1:]
        else:
            exp_out = out_shape

        prevs = self.graph.prev_node(target).items()
        for nid, eid in prevs:
            in_shape = self.shape_of(nid)
            edge = self.graph.id_to_edge(eid)
            # print(f'checking {nid} to {target}...')
            if not edge.verify_output(in_shape, exp_out):
                # print(f'false. in_shape:{in_shape}, out_shape:{exp_out}')
                next_shape, next_exp = edge.expand_out(in_shape,
                                                       out_shape,
                                                       expanded)
                if len(next_exp) == 0: 
                    continue
                else:
                    self.set_shape(nid, next_shape)
                    self.reconstruct(nid, target, next_exp)
            else:
                _ = 1
                # print('true!')

        if self.method_of(target) == 'cat':
            prev_nc = self.shape_of(target)[0]
            nc = 0
            offset = prev_nc
            target_ch = 0
            for nid, _ in prevs:
                if nid == caller:
                    offset = nc
                    target_ch = self.shape_of(nid)[0]
                nc = nc + self.shape_of(nid)[0]
            pushed = nc - prev_nc

            next_exp = [(-1 if o < 0 else offset+o, offset+c) for o, c in expanded]
            push_range = range(offset+target_ch, prev_nc)
            next_exp.extend((o, pushed+o) for o in push_range)
            
            expanded = next_exp
            next_shape = (nc,) + self.shape_of(target)[1:]
            self.set_shape(target, next_shape)
        
        nexts = self.graph.next_node(target).items()
        in_shape = self.shape_of(target)
        for nid, eid in nexts:
            out_shape = self.shape_of(nid)
            if self.method_of(nid) == 'cat':
                exp_out = (-1,) + out_shape[1:]
            else:
                exp_out = out_shape
            edge = self.graph.id_to_edge(eid)
            # print(f'checking {nid} to {target}...')
            if not edge.verify_output(in_shape, exp_out):
                # print(f'false. in_shape:{in_shape}, out_shape:{exp_out}')       
                next_shape, next_exp = edge.expand_in(in_shape,
                                                      out_shape,
                                                      expanded)
                if len(next_exp) == 0:
                    continue
                else:
                    self.set_shape(nid, next_shape)
                    self.reconstruct(nid, target, next_exp)
            elif self.method_of(nid) == 'cat':
                # print(f'true, but checking for concat layer.')
                self.reconstruct(nid, target, expanded)
            else: 
                _ = 1
                # print('true!')
        return

    def update_rank(self, target):
        rank = self.rank_of(target)
        nexts = self.graph.next_node(target)
        if self.method_of(target) != 'none':
            self.graph.anchors[rank] = target

        for nid in nexts:
            if self.method_of(nid) == 'none':
                if self.rank_of(nid) < rank:
                    self.set_rank(nid, rank)
                    self.update_rank(nid)
                else: continue
            else:
                if self.rank_of(nid) <= rank:
                    self.set_rank(nid, rank + 1)
                    self.update_rank(nid)
                else: continue

    def _verify_dest(self, src, dest, layer):
        in_shape = self.shape_of(src)
        out_shape = layer.calculate_output(in_shape)
        if dest == None:
            return self.graph.add_node(out_shape)
        else:
            dest_shape = self.shape_of(dest)
            if dest_shape == out_shape:
                return dest
            else:
                raise Exception(
                    f'''
                    Output shape not matched.
                    expected:{out_shape}, got:{dest_shape}
                    ''')

    def _add_layer(self, src, dest, layer_class, *args, **kwargs):
        layer = layer_class(*args, **kwargs)
        
        dest = self._verify_dest(src, dest, layer)
        eid = self.graph.add_edge(layer)
        self.graph.redirect_edge(eid, src, dest)
        return dest

    def add_conv2d_layer(self, src, dest=None, 
                         out_channels=None, kernel_size=3,
                         stride=1, bias=True, **kwargs):
        
        in_shape = self.shape_of(src)
        if dest != None:
            out_channels = self.shape_of(dest)[0]
        elif out_channels == None:
            raise Exception('Output shape not defined.')

        ni, nf = in_shape[0], out_channels

        return self._add_layer(src, dest, Conv2dLayer, 
                               ni, nf, kernel_size, stride, bias)

    def add_batchnorm2d_layer(self, src, dest=None, **kwargs):
        in_shape = self.shape_of(src)
        nf = in_shape[0]

        return self._add_layer(src, dest, BatchNorm2dLayer, nf)

    def add_relu_layer(self, src, dest=None, **kwargs):
        return self._add_layer(src, dest, ReluLayer)

    def add_flatten_layer(self, src, dest=None, **kwargs):
        return self._add_layer(src, dest, FlattenLayer)

    def add_linear_layer(self, src, dest=None, 
                         out_features=None, bias=True, **kwargs):
        in_shape = self.shape_of(src)
        if dest != None:
            out_features = self.shape_of(dest)[0]
        elif out_features == None:
            raise Exception('Output shape not defined.')

        ni, nf = in_shape[0], out_features

        return self._add_layer(src, dest, LinearLayer, ni, nf, bias)

    def add_dropout_layer(self, src,dest=None, p=0.5):
        return self._add_layer(src, dest, DropoutLayer, p)

    def add_maxpool2d_layer(self, src, dest=None, kernel_size=2, **kwargs):
        return self._add_layer(src, dest, MaxPool2dLayer, kernel_size)

    def add_avgpool2d_layer(self, src, dest=None, kernel_size=2, **kwargs):
        return self._add_layer(src, dest, AvgPool2dLayer, kernel_size)

    def add_adaptivemaxpool2d_layer(self, src, dest=None, 
                                    output_size=(1,1), **kwargs):
        return self._add_layer(src, dest, AdaptiveMaxPool2dLayer, output_size)

    def add_adaptiveavgpool2d_layer(self, src, dest=None, 
                                    output_size=(1,1), **kwargs):
        return self._add_layer(src, dest, AdaptiveAvgPool2dLayer, output_size)

    def add_id_layer(self, src, dest=None, method='none', **kwargs):
        if dest == None:
            dest = self.graph.add_node(self.shape_of(src), method)
            eid = self.graph.add_edge(NoneLayer())
            self.graph.redirect_edge(eid, src, dest)
        else:
            method = self.graph.id_to_node(dest).method
            if method == 'cat':
                if self.shape_of(src)[1:] != self.shape_of(dest)[1:]:
                    raise Exception('Shape difference on Cat')
                eid = self.graph.add_edge(NoneLayer())
                self.graph.redirect_edge(eid, src, dest)
                exp = [(-1, x) for x in range(self.shape_of(src)[0])]
                self.reconstruct(dest, -1, exp)

            elif method == 'add':
                in_shape = self.shape_of(src)
                out_shape = self.shape_of(dest)
                if in_shape[1:] != out_shape[1:]:
                    raise Exception('Shape difference on Add')
                if in_shape[0] != out_shape[0]:
                    src = self.add_conv2d_layer(src, 
                                                out_channels=out_shape[0],
                                                kernel_size=1, 
                                                stride=1,
                                                bias=False)
                eid = self.graph.add_edge(NoneLayer())
                self.graph.redirect_edge(eid, src, dest)
            else:
                raise Exception(f'Cannot use {method} method')
        return dest

    '''Multiple Node transformer'''
    def add_conv2d_block(self, src, dest=None, 
                         out_channels=None, kernel_size=3, stride=1, **kwargs):
        conv = self.add_conv2d_layer(src, 
                                     out_channels=out_channels, 
                                     kernel_size=kernel_size,
                                     stride=stride, 
                                     bias=False)
        bn = self.add_batchnorm2d_layer(conv)
        relu = self.add_relu_layer(bn)
        do = self.add_dropout_layer(relu, p=0.2)
        self.graph.convs.append(do)
        return do

    def add_anchor_block(self, src, dest=None, out_channels=None, kernel_size=3,
                         stride=1, method='none', **kwargs):
        conv = self.add_conv2d_block(src,
                                     out_channels=out_channels, 
                                     kernel_size=kernel_size,
                                     stride=stride)
        anc = self.add_id_layer(conv, dest, method=method)
        self.update_rank(src)
        return anc

    def insert_anchor_block(self, src, method='none', **kwargs):
        nexts = deepcopy(self.graph.next_node(src))
        nf = self.shape_of(src)[0]
        anc = self.add_anchor_block(src, out_channels=nf, method=method)
        for nid, eid in nexts.items():
            self.graph.remove_edge(src, nid)
            self.graph.redirect_edge(eid, anc, nid)
        self.update_rank(anc)
        return anc
        
    def add_linear_block(self, src, dest=None, out_features=None):
        if dest != None:
            nf = self.shape_of(dest)[0]
        elif out_features == None:
            raise Exception('Output shape not defined')
        else:
            nf = out_features

        lin = self.add_linear_layer(src, out_features=nf)
        do = self.add_dropout_layer(lin, dest, p=0.5)

        return do

    def add_skip_connection(self, src, dest, num_layers=[]):
        next = src
        for nl in num_layers:
            next = self.add_conv2d_block(next, out_channels=nl)
        anc = self.add_id_layer(next ,dest)
        self.update_rank(src)
        return anc

    def wider_conv(self, target):
        prev_nf = self.shape_of(target)[0]
        nf = next_channel(prev_nf)
        next_shape = (nf,) + self.shape_of(target)[1:]

        copy_range = range(prev_nf, nf)
        exp = [(randint(0, prev_nf - 1), x) for x in copy_range]
        self.set_shape(target, next_shape)
        self.reconstruct(target, -1, exp)


    def add_init_block(self):
        gr = self.graph
        output = gr.out_shape[0]
        anc = self.add_anchor_block(gr.input, out_channels=64, method='cat')
        pool = self.add_adaptiveavgpool2d_layer(anc, output_size=(1,1))
        flat = self.add_flatten_layer(pool)
        lin1 = self.add_linear_block(flat, out_features=output*16)
        lin2 = self.add_linear_block(lin1, gr.output)

        self.update_rank(anc)
        return gr.output
