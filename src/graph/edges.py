from torch import Tensor, randint
import torch.nn as nn

import numpy as np
from math import ceil




class Edge:
  def __init__(self, src, dest, layer=None, identical=False):
    self.src = src
    self.dest = dest
    self.identical = identical
    if not identical:
      self.layer = layer

  
