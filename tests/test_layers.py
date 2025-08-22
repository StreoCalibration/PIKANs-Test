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

    def test_b_spline_basis_outside_grid(self):
        """
        Tests B-spline basis for inputs outside the defined grid.
        The basis values should be zero for points outside the knot range.
        """
        layer = KANLayer(in_features=1, out_features=1, grid_size=5, spline_order=3)

        # Test a point to the left of the grid
        test_input_left = torch.tensor([[-2.0]])
        basis_values_left = layer.b_spline_basis(test_input_left)
        self.assertTrue(torch.all(basis_values_left == 0))

        # Test a point to the right of the grid
        test_input_right = torch.tensor([[2.0]])
        basis_values_right = layer.b_spline_basis(test_input_right)
        self.assertTrue(torch.all(basis_values_right == 0))

    def test_different_grid_and_spline_order(self):
        """
        Tests the b_spline_basis function with a different grid size and spline order.
        """
        layer = KANLayer(in_features=1, out_features=1, grid_size=8, spline_order=2)
        test_input = torch.rand(10, 1) * 2 - 1 # Random inputs in [-1, 1]

        basis_values = layer.b_spline_basis(test_input)

        # Check shape
        self.assertEqual(basis_values.shape, (10, 1, 8 + 2))

        # Check that the sum of basis functions is close to 1
        self.assertTrue(torch.allclose(torch.sum(basis_values, dim=-1), torch.ones(10, 1)))

    def test_gradient_flow(self):
        """
        Tests that gradients are flowing through the learnable parameters.
        """
        layer = KANLayer(in_features=4, out_features=2)
        dummy_input = torch.randn(8, 4, requires_grad=True)

        # Ensure all learnable parameters have requires_grad=True
        self.assertTrue(layer.spline_coeffs.requires_grad)
        self.assertTrue(layer.base_weight.requires_grad)
        self.assertTrue(layer.spline_scaler.requires_grad)

        output = layer(dummy_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients are not None
        self.assertIsNotNone(layer.spline_coeffs.grad)
        self.assertIsNotNone(layer.base_weight.grad)
        self.assertIsNotNone(layer.spline_scaler.grad)

        # Check that gradients are not all zero
        self.assertNotEqual(torch.sum(layer.spline_coeffs.grad**2), 0)
        self.assertNotEqual(torch.sum(layer.base_weight.grad**2), 0)
        self.assertNotEqual(torch.sum(layer.spline_scaler.grad**2), 0)

if __name__ == '__main__':
    unittest.main()
