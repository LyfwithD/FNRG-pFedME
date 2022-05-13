import torch
import torch.nn as nn
from params import *
from functools import reduce


def fedavg(gather_list):
    size = len(gather_list)
    summation = reduce(lambda x,y: x+y, gather_list)
    return summation / size
