from .transformer import Transformer
from .graph.anchor_graph import AnchorGraph

def test_transformer():
    gr = AnchorGraph((3, 32, 32), (10,))
    tf = Transformer(gr)
