from Server import *
from Client import *
from params import *
from Data import get_loaders

def launch():
    if args.is_Server:
        node = Server()
    else:
        node = Client()
    train_loader, test_loader = get_loaders()
    for epoch in range(args.epochs):
        print('current epoch is:', epoch)
        if not args.is_Server:
            node.model.pFedME_step(train_loader, test_loader)
        node.sync_change_weights()
    node.model.evaluate(test_loader)


if __name__ == '__main__':
    launch()