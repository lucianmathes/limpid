import unittest

from limpid import Layer, Sample


class TestLayer(unittest.TestCase):
    def test_layer_info_in_sample(self):
        """
        Test that the Layer parameters are handed to the Sample object.
        """
        lower_boundary = 0
        density = 4.5
        a, n, m = (1, 2, 3)

        l = Layer(density, (a, n, m))
        s = Sample([l])
        l_sample = s.layers[0]


if __name__ == "__main__":
    unittest.main()
