import torch
from torchvision import datasets, transforms


def loadData(batchSize=64):

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainDataset = datasets.MNIST(
        './data',
        train=True,
        transform=transform,
        download=True
    )

    testDataset = datasets.MNIST(
        './data',
        train=False,
        transform=transform,
        download=True
    )

    trainLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True
    )

    testLoader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False
    )

    return trainLoader, testLoader