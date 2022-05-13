import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def get_single_loader(train=True, batch_size=32):
    dataset = datasets.MNIST(root='./data', train=train, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_loaders():
    train_loader = get_single_loader(train=True)
    test_loader = get_single_loader(train=False)
    return train_loader, test_loader

