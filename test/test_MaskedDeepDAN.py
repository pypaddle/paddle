import unittest
import numpy as np
import paddle.util
import paddle.sparse
import networkx as nx


class MaskedDeepDANTest(unittest.TestCase):
    def test_random_structures(self):
        random_graph = nx.watts_strogatz_graph(100, 3, 0.8)

        structure = paddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)

        model = paddle.sparse.MaskedDeepDAN(784, 10, structure)

        new_model = paddle.sparse.MaskedDeepDAN(784, 10, model.generate_structure())

    def test_get_structure(self):
        structure = paddle.sparse.CachedLayeredGraph()

        block0_size = 8
        block1_size = 8
        block2_size = 2
        block3_size = 2
        block4_size = 2
        block5_size = 2
        block6_size = 10
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
        for v in block0:
            for t in block5:
                structure.add_edge(v, t)
        for v in block1:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block4:
                structure.add_edge(v, t)
        for v in block1:
            for t in block6:
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

        new_structure = model.generate_structure(include_input=False, include_output=False)

        print('Obtained structure')
        #print('First layer size', new_structure.first_layer_size)
        #print('Last layer size', new_structure.last_layer_size)
        #print('Layers', new_structure.layers)
        #print(len(new_structure.nodes))

        """
        print('Drawing ..')
        nx.draw(structure)
        plt.show()
        nx.draw(new_structure)
        plt.show()"""

        model2 = paddle.sparse.MaskedDeepDAN(784, 10, new_structure)
        print(model2)

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