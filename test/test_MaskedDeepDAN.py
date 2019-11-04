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
    def test_get_structure(self):
        initial_structure = paddle.sparse.CachedLayeredGraph()
        initial_structure.add_nodes_from(np.arange(100))
        model = paddle.sparse.MaskedDeepDAN(784, 10, initial_structure)

        structure = model.get_structure(include_input=True, include_output=True)
        print(structure)

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
        for v in block3:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block6:
                structure.add_edge(v, t)
        for v in block4:
            for t in block6:
                structure.add_edge(v, t)

        model = paddle.sparse.MaskedDeepDAN(784, 10, structure)
        print(model)