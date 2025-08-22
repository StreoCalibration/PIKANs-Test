import unittest
import torch
from src.model.pikans import PIKANs

class TestModel(unittest.TestCase):
    """
    Test suite for the PIKANs model.
    """
    def test_model_forward_pass(self):
        """
        Tests the forward pass of the PIKANs model to ensure correct output shape.
        """
        B, C = 1024, 12  # Batch size, input channels (4 buckets * 3 wavelengths)
        model = PIKANs(layer_widths=[C, 64, 1])
        dummy_input = torch.randn(B, C)
        output = model(dummy_input)
        self.assertEqual(output.shape, (B, 1))

    def test_model_layer_structure(self):
        """
        Tests that the model is constructed with the correct number and types of layers.
        """
        layer_widths = [12, 64, 32, 1]
        model = PIKANs(layer_widths=layer_widths)

        # Expecting len(layer_widths) - 1 KAN layers
        self.assertEqual(len(model.layers), len(layer_widths) - 1)

        from src.model.layers import KANLayer
        for i, layer in enumerate(model.layers):
            self.assertIsInstance(layer, KANLayer)
            self.assertEqual(layer.in_features, layer_widths[i])
            self.assertEqual(layer.out_features, layer_widths[i+1])

    def test_model_with_different_widths(self):
        """
        Tests the model with a different set of layer widths.
        """
        B, C = 512, 8
        layer_widths = [C, 32, 16, 1]
        model = PIKANs(layer_widths=layer_widths)
        dummy_input = torch.randn(B, C)
        output = model(dummy_input)
        self.assertEqual(output.shape, (B, 1))

if __name__ == '__main__':
    unittest.main()