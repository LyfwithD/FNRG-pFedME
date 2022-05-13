import torch
import torch.nn as nn
from functools import reduce
from torch.optim import SGD
from params import *
from utils import *

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax()
        )
        self.weights = list(self.model.parameters())
        self.optim = SGD(self.model.parameters(),lr=args.local_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def pFedME_step(self, train_loader, test_loader):
        for idx, (data, target) in enumerate(train_loader):
            initial_weights = [ele.clone().detach() for ele in self.weights]
            data = data.view(32, -1)
            while True:
                self.optim.zero_grad()
                self.pFedME_loss_fn(self.forward(data), target, initial_weights).backward()
                self.optim.step()
                if gradient_check(self.weights, args.error):
                    break
            # update the local weights
            map(lambda x,y: x.data.copy_(y.data - args.lam * args.local_lr * (y - x)), self.weights, initial_weights)
            if idx % 10 == 0:
                print("current local iteration:", idx)
                print('current model loss:', self.evaluate(test_loader))

    def pFedME_loss_fn(self, logits, target, ini_weights):
        loss = self.cross_entropy_loss(logits, target)
        loss += reduce(lambda x, y: x + y, map(lambda x, y:args.lam / 2 * torch.norm(x-y), self.weights, ini_weights))
        return loss

    def evaluate(self, test_loader):
        eval_loss = 0
        with torch.no_grad():
            for data in test_loader:
                img, label = data
                print(img.shape)
                img = img.view(img.shape[0], -1)
                eval_loss += self.cross_entropy_loss(self.forward(img), label)
        return eval_loss


if __name__ == '__main__':
    from Data import get_loaders
    train_loader, test_loader = get_loaders()
    model = FullyConnected()
    # print('start train')
    # model.pFedME_step(train_loader)
    print('start test')
    model.evaluate(test_loader)