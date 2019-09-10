import torch
import torch.utils
import torch.nn as nn
import unittest
import numpy as np
import paddle.util
import networkx as nx
import uuid

from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms

from paddle.learning import train, test
from paddle.sparse import FiveWaySparseBlockNet


class SparseBlocksTest(unittest.TestCase):

    def test_dev(self):
        nodes = [1, 2, 3, 4, 5]

        structure = nx.DiGraph()
        structure.add_nodes_from(nodes)
        structure.add_edge(1, 3)
        structure.add_edge(1, 4)
        structure.add_edge(1, 5)
        structure.add_edge(2, 3)
        structure.add_edge(2, 4)
        structure.add_edge(3, 5)
        structure.add_edge(4, 5)

        custom_transform = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
        ])
        batch_size = 100
        epochs = 100

        """
        MNIST
        """
        # download and transform train dataset
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('/media/data/set/mnist/',
                                                                  download=True,
                                                                  train=True,
                                                                  transform=custom_transform),
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # download and transform test dataset
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('/media/data/set/mnist/',
                                                                 download=True,
                                                                 train=False,
                                                                 transform=custom_transform),
                                                  batch_size=batch_size,
                                                  shuffle=True)


        """
        Cifar10
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar10_dataset_root = '/media/data/set/cifar10'
        cifar10_train_set = datasets.CIFAR10(root=cifar10_dataset_root, train=True, download=True, transform=transform)
        cifar10_test_set = datasets.CIFAR10(root=cifar10_dataset_root, train=False, download=True, transform=transform)

        # Training
        n_training_samples = 20000
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        # Test
        n_test_samples = 5000
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)



        """
        Model
        """
        model = FiveWaySparseBlockNet(3*32*32, 10)
        print(model)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print('Epoch', epoch)
            loss, perc = train(cifar10_train_loader, model, optimizer, loss_func, device, percentage=True)
            print('Train', loss, perc)

            print('Test', test(cifar10_test_loader, model, device))

        torch.save(model, 'model-uuid_{}-epochs_{}.pt'.format(uuid.uuid4(), epochs))

    def test_loaders(self):

        batch_size = 100
        epochs = 100

        """
        MNIST
        """
        custom_transform = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
        ])
        # download and transform train dataset
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('/media/data/set/mnist/',
                                                                  download=True,
                                                                  train=True,
                                                                  transform=custom_transform),
                                                                   batch_size=batch_size,
                                                                   shuffle=True)

        # download and transform test dataset
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('/media/data/set/mnist/',
                                                                 download=True,
                                                                 train=False,
                                                                 transform=custom_transform),
                                                                  batch_size=batch_size,
                                                                  shuffle=True)

        """
        Cifar10
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar10_dataset_root = '/media/data/set/cifar10'
        cifar10_train_set = datasets.CIFAR10(root=cifar10_dataset_root, train=True, download=True, transform=transform)
        cifar10_test_set = datasets.CIFAR10(root=cifar10_dataset_root, train=False, download=True, transform=transform)

        # Training
        n_training_samples = 20000
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True)

        for features, labels in mnist_train_loader:
            print(features.shape)