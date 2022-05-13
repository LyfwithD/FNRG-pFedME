from Model import *
from Node import *


class Client(Node):
    def __init__(self):
        super().__init__()
        self.model = FullyConnected()

    def sync_change_weights(self):
        # gather model parameters to the server
        for tensor in self.model.weights:
            dist.gather(tensor=tensor, dst=0)

        # receiving the broadcasted model parameters from the server
        for tensor in self.model.weights:
            dist.broadcast(tensor=tensor, src=0)


class DecentralizedClient(Node):
    def __init__(self):
        super().__init__()
        self.model = FullyConnected()
        self.change_tensors = []
        for weight in self.model.weights:
            temp = []
            for _ in range(self.neighbour_size + 1):
                temp.append(weight.clone().detach().zero_())
            self.change_tensors.append(temp)

    def sync_change_weights(self):
        # gather model parameters to the server
        # group communication is not applicable
        # for it needs every node to know the entire graph
        # use asynchronous transmission techniques
        for neighbour in self.neighbours:
            for tensor in self.model.weights:
                dist.gather(tensor=tensor, gather_list=self.change_tensors, dst=self.rank)

        # receiving the broadcasted model parameters from the server
        for tensor in self.model.weights:
            dist.broadcast(tensor=tensor, src=0)
