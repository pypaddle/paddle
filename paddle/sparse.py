import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import paddle.util
from torch.autograd import Variable


class LayeredGraph(nx.DiGraph):
    @property
    def num_layers(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def first_layer_size(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def last_layer_size(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def layers(self):
        """
        :rtype: list[int]
        """
        raise NotImplementedError()

    def get_layer(self, vertex: int):
        """
        :rtype: int
        """
        raise NotImplementedError()

    def get_vertices(self, layer: int):
        """
        :rtype: list[int]
        """
        raise NotImplementedError()

    def get_layer_size(self, layer: int):
        """
        :rtype: int
        """
        raise NotImplementedError()

    def layer_connected(self, layer_index1: int, layer_index2: int):
        """
        :rtype: bool
        """
        raise NotImplementedError()

    def layer_connection_size(self, layer_index1: int, layer_index2: int):
        """
        :rtype: int
        """
        raise NotImplementedError()


class CachedLayeredGraph(LayeredGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self._has_changed = True
        self._layer_index = None
        self._vertex_by_layer = None

    def add_cycle(self, nodes, **attr):
        super(LayeredGraph, self).add_cycle(nodes, **attr)
        self._has_changed = True

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        super(LayeredGraph, self).add_edge(u_of_edge, v_of_edge, **attr)
        self._has_changed = True

    def add_edges_from(self, ebunch_to_add, **attr):
        super(LayeredGraph, self).add_edges_from(ebunch_to_add, **attr)
        self._has_changed = True

    def add_node(self, node_for_adding, **attr):
        super(LayeredGraph, self).add_node(node_for_adding, **attr)
        self._has_changed = True

    def add_nodes_from(self, nodes_for_adding, **attr):
        super(LayeredGraph, self).add_nodes_from(nodes_for_adding, **attr)
        self._has_changed = True

    def add_path(self, nodes, **attr):
        super(LayeredGraph, self).add_path(nodes, **attr)
        self._has_changed = True

    def add_star(self, nodes, **attr):
        super(LayeredGraph, self).add_star(nodes, **attr)
        self._has_changed = True

    def add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr):
        super(LayeredGraph, self).add_weighted_edges_from(ebunch_to_add, weight='weight', **attr)
        self._has_changed = True

    def _get_layer_index(self):
        if self._has_changed or self._layer_index is None or self._vertex_by_layer is None:
            self._build_layer_index()
            self._has_changed = False

        return self._layer_index, self._vertex_by_layer

    def _layer_by_vertex(self, vertex: int):
        return self._get_layer_index()[0][vertex]

    def _vertices_by_layer(self, layer: int):
        return self._get_layer_index()[1][layer]

    def _build_layer_index(self):
        self._layer_index, self._vertex_by_layer = paddle.util.build_layer_index(self)

    @property
    def num_layers(self):
        return len(self._get_layer_index()[0].keys())

    @property
    def first_layer_size(self):
        return self.get_layer_size(self.layers[0])

    @property
    def last_layer_size(self):
        return self.get_layer_size(self.layers[-1])

    @property
    def layers(self):
        return [layer for layer in self._get_layer_index()[1]]

    def get_layer(self, vertex: int):
        return self._layer_by_vertex(vertex)

    def get_vertices(self, layer: int):
        return self._vertices_by_layer(layer)

    def get_layer_size(self, layer: int):
        return len(self._vertices_by_layer(layer))

    def layer_connected(self, layer_index1: int, layer_index2: int):
        """
        :rtype: bool
        """
        if layer_index1 is layer_index2:
            raise ValueError('Same layer does not have interconnections, it would be split up.')
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    return True
        return False

    def layer_connection_size(self, layer_index1: int, layer_index2: int):
        """
        :rtype: int
        """
        if layer_index1 is layer_index2:
            raise ValueError('Same layer does not have interconnections, it would be split up.')
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        size = 0
        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    size += 1
        return size




class MaskedDeepDAN(nn.Module):
    """
    A deep directed acyclic network model which is capable of masked layers and masked skip-layer connections.
    """
    def __init__(self, input_size, num_classes, structure : LayeredGraph):
        super(MaskedDeepDAN, self).__init__()

        #layer_index, vertex_by_layer = paddle.util.build_layer_index(structure)
        #layers = [l for l in vertex_by_layer]

        self._structure = structure
        assert structure.num_layers > 0

        self.layer_first = MaskedLinearLayer(input_size, structure.first_layer_size)
        self.layers_main_hidden = nn.ModuleList([MaskedLinearLayer(structure.get_layer_size(l-1), structure.get_layer_size(l)) for l in structure.layers[1:]])

        skip_layers = []
        self._skip_targets = {}
        for target_layer in structure.layers[2:]:
            target_size = structure.get_layer_size(target_layer)
            for distant_source_layer in structure.layers[:target_layer-1]:
                if structure.layer_connected(distant_source_layer, target_layer):
                    if target_layer not in self._skip_targets:
                        self._skip_targets[target_layer] = []
                    #print('Layer %s is connected to %s' % (distant_source_layer, target_layer))
                    skip_layer = MaskedLinearLayer(structure.get_layer_size(distant_source_layer), target_size)
                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append({'layer': skip_layer, 'source': distant_source_layer})
        self.layers_skip_hidden = nn.ModuleList(skip_layers)

        self.layer_out = MaskedLinearLayer(structure.last_layer_size, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        last_output = self.activation(self.layer_first(x))
        layer_results = {}
        for layer, layer_idx in zip(self.layers_main_hidden, self._structure.layers[1:]):
            out = self.activation(layer(last_output))

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target['layer']
                    source_idx = skip_target['source']

                    out += self.activation(source_layer(layer_results[source_idx]))

            layer_results[layer_idx] = out  # copy?
            last_output = out

        return self.layer_out(last_output)


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


