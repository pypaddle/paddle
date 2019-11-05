import torch
import torch.utils
import unittest
import paddle.sparse


class MaskedLinearLayerTest(unittest.TestCase):
    def test_get_structure(self):
        model = paddle.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])
        structure = model.generate_structure(include_input=True, include_output=True)
        print(structure)
        # TODO

    def test_default(self):
        model = paddle.sparse.MaskedDeepFFN(784, 10, [200, 100, 50])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        print(model)