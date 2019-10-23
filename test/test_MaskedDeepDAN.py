import unittest
import networkx as nx
import numpy as np
import paddle.util
import paddle.sparse

import torch
import torch.utils
import torch.nn as nn
from paddle.learning import train, test
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms


class MaskedDeepDANTest(unittest.TestCase):

    def test_dev(self):
        structure = paddle.sparse.CachedLayeredGraph()
        structure.add_nodes_from(np.arange(1, 7))

        block0_size = 50
        block1_size = 50
        block2_size = 30
        block3_size = 30
        block4_size = 30
        block5_size = 20
        block6_size = 20
        block0 = np.arange(1, block0_size+1)
        block1 = np.arange(block0_size+1, block0_size+block1_size+1)
        block2 = np.arange(block0_size+block1_size+1, block0_size+block1_size+block2_size+1)
        block3 = np.arange(block0_size+block1_size+block2_size+1, block0_size+block1_size+block2_size+block3_size+1)
        block4 = np.arange(block0_size+block1_size+block2_size+block3_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+1)
        block5 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1)
        block6 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+block6_size+1)

        # First layer
        for v in block0:
            for t in block2:
                structure.add_edge(v, t)
        for v in block0:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block4:
                structure.add_edge(v, t)

        # Second layer
        for v in block2:
            for t in block5:
                structure.add_edge(v, t)
        for v in block2:
            for t in block6:
                structure.add_edge(v, t)
        for v in block3:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block6:
                structure.add_edge(v, t)
        for v in block4:
            for t in block5:
                structure.add_edge(v, t)
        for v in block4:
            for t in block6:
                structure.add_edge(v, t)

        model = paddle.sparse.MaskedDeepDAN(784, 10, structure)
        print(model)


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

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print('Epoch', epoch)
            loss, perc = train(mnist_train_loader, model, optimizer, loss_func, device, percentage=True)
            print('Train', loss, perc)

            print('Test', test(mnist_test_loader, model, device))