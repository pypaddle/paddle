import numpy as np
import torch
import torch.nn as nn
import unittest
import paddle.sparse
import paddle.util


class MaskedLinearLayerTest(unittest.TestCase):
    def test_set_mask_explicitly_success(self):
        input_size = 5
        output_size = 2
        layer = paddle.sparse.MaskedLinearLayer(input_size, output_size)
        mask = torch.zeros((output_size, input_size), dtype=torch.bool)
        mask[0, 0] = 1
        mask[0, 1] = 1
        mask[1, 2] = 1

        layer.set_mask(mask)

        self.assertTrue(np.all(np.equal(np.array(mask), np.array(layer.mask))))

    def test_parameter_reset_success(self):
        # Arrange - initialize a masked layer and randomize its mask
        input_size = 5
        output_size = 7
        layer = paddle.sparse.MaskedLinearLayer(input_size, output_size)
        layer.apply(paddle.util.set_random_masks)
        initial_state = np.copy(layer.mask)

        # Act - Now the mask should be reset to only ones
        layer.reset_parameters()

        # Assert - The random mask and the resetted mask should not match
        self.assertEqual(layer.mask.size(), initial_state.shape)
        self.assertTrue((np.array(layer.mask) != initial_state).any())

    def test_mask_changes_output_success(self):
        input_size = 5
        output_size = 7
        layer = paddle.sparse.MaskedLinearLayer(input_size, output_size)
        input = torch.rand(input_size)

        layer.apply(paddle.util.set_random_masks)
        first_mask = np.copy(layer.mask)
        first_mask_output = layer(input).detach().numpy()
        layer.apply(paddle.util.set_random_masks)
        second_mask = np.copy(layer.mask)
        second_mask_output = layer(input).detach().numpy()

        self.assertTrue((first_mask != second_mask).any(), 'Masks for inference should not equal, but are randomly generated.')
        self.assertTrue(np.any(np.not_equal(first_mask_output, second_mask_output)))

    def test_back(self):
        input_size = 5
        output_size = 2
        model = paddle.sparse.MaskedLinearLayer(input_size, output_size)
        input = torch.ones(input_size) * 0.3

        output = model(input)
        print(output)

        output.backward(torch.ones(output_size) * 0.8)

        print('weight', model.weight)
        print('weight._grad', model.weight._grad)
        print('bias', model.bias)
        print('bias._grad', model.bias._grad)

    def test_transform_filter(self):
        def transform_filter(filter : np.array, lower_bound=-1, upper_bound=1):
            assert lower_bound < upper_bound
            assert len(filter.shape) is 2
            assert filter.shape[0] is filter.shape[1]
            assert filter.shape[0] > 1

            filter_size = filter.shape[0]
            window_size = abs(upper_bound-lower_bound)
            divisions = window_size/(filter_size-1)

            coordinates = np.arange(lower_bound, upper_bound+divisions, divisions)
            for x in coordinates:
                for y in coordinates[::-1]:
                    print(x, y)
            #xx, yy = np.meshgrid(coordinates, coordinates, indexing='ij')
            #print(np.meshgrid(coordinates, coordinates, indexing='xy'))
            #print(np.meshgrid(coordinates, coordinates, indexing='ij'))


        filter = lambda size: np.arange(size*size).reshape(size, size)

        print(filter(3))
        print(transform_filter(filter(3)))

        print(filter(4))
        print(transform_filter(filter(4)))

        print(filter(5))
        print(transform_filter(filter(5)))