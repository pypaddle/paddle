import unittest
import paddle.sparse


class MaskedLinearLayerTest(unittest.TestCase):
    def test_default(self):
        layer = paddle.sparse.MaskedLinearLayer(5, 2)
