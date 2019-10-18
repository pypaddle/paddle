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
        """
        super().__init__(in_feature, out_features, bias)

        self.register_buffer('mask', torch.ones((out_features, in_feature), dtype=torch.bool))
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def get_mask(self):
        return self.mask

    def set_mask(self, mask):
        self.mask = Variable(mask)

    def get_weight_count(self):
        return self.mask.sum()

    def get_weight(self):
        return self.weight

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        # hasattr() is necessary because reset_parameters() is called in __init__ of Linear(), but buffer 'mask'
        # may only be registered after super() call, thus 'mask' might not be defined as buffer / attribute, yet
        if hasattr(self, 'mask') and not keep_mask:
            self.mask = torch.ones(self.weight.size(), dtype=torch.bool)

    def forward(self, x):
        # Possibly store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data
        weight = self.weight.mul(self.mask)
        return F.linear(x, weight, self.bias)


