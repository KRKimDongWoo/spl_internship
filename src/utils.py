from math import ceil, log, pi, cos
from numpy.random import choice, binomial 
from .constants import NEXT_CHANNEL

def partial(func, *args, **kwargs):
    return func(*args, **kwargs)

def next_channel(ch):
    for minimum, maximum in NEXT_CHANNEL.items():
        if minimum <= ch and ch < maximum:
            return maximum
    return ch

def rand_bool(p=0.5):
    return choice([True, False], p=[p, 1-p])

def wider_prob(ch):
    return (1+cos(pi/4*log(ch/32, 2)))/4

def try_wider(ch):
    return rand_bool(p=wider_prob(ch))

def layer_nums():
    bi = binomial(6, 1/3)
    return bi if bi > 0 else 1

def skip_layers():
    return layer_nums()

def try_skip(nexts):
    prob = (1/2)**nexts
    return rand_bool(prob)

def deeper_prob(variable):
    return 1 / 2 + 1 / (1 + variable)

def try_deeper(rank, conv):
    return rand_bool(p=deeper_prob(rank, conv))
