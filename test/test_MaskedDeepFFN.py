import torch
import torch.utils
import torch.nn as nn
import unittest
import paddle.sparse

from torchvision import datasets, transforms

from paddle.learning import train, test


class MaskedLinearLayerTest(unittest.TestCase):
    def test_default(self):
        model = paddle.sparse.MaskedDeepFFN(784, 10, [200, 100, 50])

        batch_size = 100
        epochs = 100

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        """
        MNIST
        """
        custom_transform = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
        ])
        # download and transform train dataset
        mnist_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/media/data/set/mnist/',
            download=True,
            train=True,
            transform=custom_transform),
            batch_size=batch_size,
            shuffle=True)

        # download and transform test dataset
        mnist_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/media/data/set/mnist/',
            download=True,
            train=False,
            transform=custom_transform),
            batch_size=batch_size,
            shuffle=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss_func = nn.CrossEntropyLoss()

        model.to(device)

        for epoch in range(10):
            print(train(mnist_train_loader, model, optimizer, loss_func, device, percentage=True))

        print(test(mnist_test_loader, model, device))

        print(model)