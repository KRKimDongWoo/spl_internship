from torch import nn, cat, Tensor

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Concat(nn.Module):
    def forward(self, x):
        if isinstance(x, Tensor):
            return x
        else:
            return cat(x, 1)

class Add(nn.Module):
    def forward(self, x):
        if isinstance(x, Tensor):
            return x
        elif isinstance(x, list):
            return sum(x)
