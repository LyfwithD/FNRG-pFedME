from Model import *
from Node import *
from Aggregation import *

class Server(Node):
    def __init__(self):
        super().__init__()
        self.model = FullyConnected()
        # store the received tensor
        self.change_tensors = []
        for weight in self.model.weights:
            temp = []
            for _ in range(self.neighbour_size + 1):
                temp.append(weight.clone().detach().zero_())
            self.change_tensors.append(temp)

    def sync_change_weights(self):
        for gather_list in self.change_tensors:
            dist.gather(gather_list[0], gather_list, dst=0)

        for tensor_list in self.change_tensors:
            dist.broadcast(fedavg(tensor_list[1:]), src=0)






