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

if __name__ == '__main__':
    unittest.main()