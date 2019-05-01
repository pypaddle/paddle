import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import export


@export
class MaskedDeepFFN(nn.Module):
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


@export
def prunable_layers(network):
    for child in network.children():
        if type(child) is MaskedLinearLayer:
            yield child
        elif type(child) is nn.ModuleList:
            for layer in prunable_layers(child):
                yield layer


@export
def prunable_layers_with_name(network):
    for name, child in network.named_children():
        if type(child) is MaskedLinearLayer:
            yield name, child
        elif type(child) is nn.ModuleList:
            for name, layer in prunable_layers_with_name(child):
                yield name, layer


@export
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