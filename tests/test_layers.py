import unittest
import torch
from src.model.layers import KANLayer

class TestKANLayer(unittest.TestCase):
    """
    Test suite for the KANLayer.
    """

    def test_layer_forward_pass_shape(self):
        """
        Tests the forward pass of a KANLayer to ensure correct output shape.
        """
        batch_size = 16
        in_features = 8
        out_features = 4

        layer = KANLayer(in_features, out_features)
        dummy_input = torch.randn(batch_size, in_features)
        output = layer(dummy_input)

        self.assertEqual(output.shape, (batch_size, out_features))

    def test_b_spline_basis_values(self):
        """
        Tests the B-spline basis function with a known, simple case.
        With grid_size=1, spline_order=1, we should have simple linear splines (hat functions).
        """
        in_features = 1
        grid_size = 1
        spline_order = 1

        # grid will be [-1, 1]
        layer = KANLayer(in_features, out_features=1, grid_size=grid_size, spline_order=spline_order)

        # Input at the center of the grid
        # For a linear spline (hat function), the center basis should be 1.0
        # and others should be 0.
        test_input = torch.tensor([[0.0]]) # a point exactly between -1 and 1

        # Expected knots: [-1, -1, 1, 1]
        # Basis functions are centered at the knots.
        # With input 0, the second basis function (related to the knot at -1) and
        # third (related to knot at 1) should have non-zero values.
        basis_values = layer.b_spline_basis(test_input)

        # With order 1, we expect 2 basis functions to be non-zero for an input
        # inside the grid range.
        self.assertEqual(basis_values.shape[-1], grid_size + spline_order)

        # The sum of basis functions should be 1 for any point within the grid.
        self.assertTrue(torch.allclose(torch.sum(basis_values, dim=-1), torch.tensor(1.0)))

if __name__ == '__main__':
    unittest.main()
