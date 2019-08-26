from torch import nn

class Generator(nn.Module):
    def __init__(self, ts):
        super().__init__()
        self.sequence = []
        self.ts = ts
        self.layers = nn.ModuleDict()

        index = 0
        for layer, input, output in ts:
            self.layers[str(index)] = layer
            self.sequence.append((str(index), input, output))
            index = index + 1

    def forward(self, x):
        value = {1: x}
        for layer, input, output in self.sequence:
            if isinstance(input, list):
                value[output] = self.layers[layer]([value[i] for i in input])
            else:
                value[output] = self.layers[layer](value[input])
        return value[2]
