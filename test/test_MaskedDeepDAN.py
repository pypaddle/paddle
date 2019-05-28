import unittest
import networkx as nx
import paddle
import paddle.util


class MaskedDeepDANTest(unittest.TestCase):

    def test_dev(self):
        structure = nx.DiGraph()
        structure.add_nodes_from([1, 2, 3, 4, 5])
        structure.add_edge(1, 3)
        structure.add_edge(1, 4)
        structure.add_edge(1, 5)
        structure.add_edge(2, 3)
        structure.add_edge(2, 4)
        structure.add_edge(3, 5)
        structure.add_edge(4, 5)

        layer_index, vertex_by_layer = paddle.util.build_layer_index(structure)
        for layer in vertex_by_layer:
            print(layer)