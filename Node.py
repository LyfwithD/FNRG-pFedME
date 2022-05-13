import torch.distributed as dist
import os
import numpy as np
from params import *


class Node(object):
    def __init__(self):
        os.environ['MASTER_ADDR'] = '192.168.1.101'
        os.environ['MASTER_PORT'] = '8000'
        os.environ['GLOO_SOCKET_IFNAME'] = args.net
        dist.init_process_group(rank=args.rank, world_size=args.world_size, backend=args.backend)
        self.rank = args.rank
        self.neighbours = np.load('adj_matrix.npy', allow_pickle=True)[self.rank]
        self.neighbour_size = len(self.neighbours)


    @staticmethod
    def sync_change_weights(self):
        pass
