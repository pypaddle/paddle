import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import paddle.util
from torch.autograd import Variable


class MaskedDeepDAN(nn.Module):
    """
    A deep directed acyclic network model which is capable of masked layers and masked skip-layer connections.
    """
    def __init__(self, input_size, num_classes, structure : nx.DiGraph):
        super(MaskedDeepDAN, self).__init__()

        layer_index, vertex_by_layer = paddle.util.build_layer_index(structure)
        layers = [l for l in vertex_by_layer]

        self.layer_first = MaskedLinearLayer(input_size, len(vertex_by_layer[0]))
        self.layers_main_hidden = nn.ModuleList([MaskedLinearLayer(len(vertex_by_layer[l-1]), len(vertex_by_layer[l])) for l in layers[1:]])
        self.layer_out = MaskedLinearLayer((len(vertex_by_layer[layers[-1]]), num_classes))
        self.activation = nn.ReLU()


class MaskedDeepFFN(nn.Module):
    """
    A deep feed-forward network model which is capable of masked layers.
    Masked layers can represent sparse structures between consecutive layers.
    This representation is suitable for feed-forward sparse networks, probably with density 0.5 and above per layer.
    """
    def __init__(self, input_size, num_classes, hidden_layers : list):
        super(MaskedDeepFFN, self).__init__()
        assert len(hidden_layers) > 0

        self.layer_first = MaskedLinearLayer(input_size, hidden_layers[0])
        self.layers_hidden = nn.ModuleList([MaskedLinearLayer(hidden_layers[l], h) for l, h in enumerate(hidden_layers[1:])])
        self.layer_out = MaskedLinearLayer(hidden_layers[-1], num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.layer_first(x))
        for layer in self.layers_hidden:
            out = self.activation(layer(out))
        return self.layer_out(out)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.layer_first.to(*args, **kwargs)
        for idx, h in enumerate(self.layers_hidden):
            self.layers_hidden[idx] = self.layers_hidden[idx].to(*args, **kwargs)
        self.layer_out.to(*args, **kwargs)
        return self


def maskable_layers(network):
    for child in network.children():
        if type(child) is MaskedLinearLayer:
            yield child
        elif type(child) is nn.ModuleList:
            for layer in maskable_layers(child):
                yield layer


def maskable_layers_with_name(network):
    for name, child in network.named_children():
        if type(child) is MaskedLinearLayer:
            yield name, child
        elif type(child) is nn.ModuleList:
            for name, layer in maskable_layers_with_name(child):
                yield name, layer


def prunable_layers(network):
    return maskable_layers(network)


def prunable_layers_with_name(network):
    return maskable_layers_with_name(network)


class MaskedLinearLayer(nn.Linear):
    def __init__(self, in_feature, out_features, bias=True, keep_layer_input=False):
        """
        :param in_feature:          The number of features that are inserted in the layer.
        :param out_features:        The number of features that are returned by the layer.
        :param bias:                Iff each neuron in the layer should have a bias unit as well.
        :param keep_layer_input:    Iff the Mask should also store the layer input for further calculations. This is
                                    needed by
        """
        super().__init__(in_feature, out_features, bias)
        # create a mask of ones for all weights (no element pruned at beginning)
        self.mask = Variable(torch.ones(self.weight.size()))
        self.saliency = None
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def get_saliency(self):
        if self.saliency is None:
            return self.weight.data.abs()
        else:
            return self.saliency

    def set_saliency(self, sal):
        if not sal.size() == self.weight.size():
            raise ValueError('mask must have same size as weight matrix')

        self.saliency = sal

    def get_mask(self):
        return self.mask

    def set_mask(self, mask=None):
        if mask is not None:
            self.mask = Variable(mask)
        self.weight.data = self.weight.data * self.mask.data

    def get_weight_count(self):
        return self.mask.sum()

    def get_weight(self):
        return self.weight

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        if not keep_mask:
            self.mask = Variable(torch.ones(self.weight.size()))
            self.saliency = None

    def forward(self, x):
        # eventually store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data
        weight = self.weight.mul(self.mask)
        return F.linear(x, weight, self.bias)

    def cuda(self, device=None):
        super().cuda(device)
        self.mask = self.mask.cuda(device)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        return self


class LinearBlockUnit(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super(LinearBlockUnit, self).__init__()
        self.act = nn.ReLU()
        #self.conv = nn.Conv2d(1, 10, kernel_size=3, stride=stride, padding=1)
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.act(x)
        out = self.linear(out)
        out = self.bn(out)
        return out

class FiveWaySparseBlockNet(nn.Module):
    def __init__(self, input_size, output_classes, hidden_block_size=100):
        super(FiveWaySparseBlockNet, self).__init__()

        self.input_blocks = nn.ModuleList()
        for _ in range(5):
            self.input_blocks.append(LinearBlockUnit(input_size, hidden_block_size))

        self.hidden_blocks = nn.ModuleList()
        for _ in range(5):
            self.hidden_blocks.append(LinearBlockUnit(hidden_block_size, hidden_block_size))

        self.output_blocks = nn.ModuleList()
        for _ in range(5):
            self.output_blocks.append(LinearBlockUnit(hidden_block_size, output_classes))

    def forward(self, x):
        input_to_blocks = {}
        for idx, block_unit in enumerate(self.input_blocks):
            input_to_blocks[idx] = block_unit(x)

        hidden_block_results = {}
        for idx, block_unit in enumerate(self.hidden_blocks):
            hidden_block_results[idx] = block_unit(input_to_blocks[idx])

        output_blocks = []
        for idx, block_unit in enumerate(self.output_blocks):
            output_blocks.append(block_unit(hidden_block_results[idx]))

        return torch.mean(torch.stack(output_blocks), dim=0)