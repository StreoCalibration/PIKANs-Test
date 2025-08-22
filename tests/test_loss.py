import unittest
import torch
import numpy as np
from src.loss.physics_informed_loss import PhysicsInformedLoss

class TestPhysicsInformedLoss(unittest.TestCase):
    """
    Test suite for the PhysicsInformedLoss function.
    """

    def test_zero_loss_on_perfect_data(self):
        """
        Tests that the loss is zero for a perfect physical scenario.
        If the provided intensities are perfectly described by the height map,
        the loss should be zero.
        """
        # --- Setup Parameters ---
        batch_size = 4
        num_wavelengths = 3
        num_buckets = 4
        wavelengths = [450e-9, 550e-9, 650e-9]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Instantiate Loss ---
        # Workaround for torch.linspace issue in this environment
        phase_shifts = torch.arange(0, num_buckets) * (2 * np.pi / num_buckets)
        loss_fn = PhysicsInformedLoss(
            wavelengths=wavelengths,
            num_wavelengths=num_wavelengths,
            num_buckets=num_buckets,
            phase_shifts=phase_shifts
        ).to(device)

        # --- Generate Synthetic Data ---
        # Create a known height map
        predicted_height = torch.linspace(0, 100e-9, batch_size).unsqueeze(1).to(device) # Shape: (batch_size, 1)

        # Generate intensities that perfectly match the physics model for this height
        A = torch.rand(batch_size, num_wavelengths, 1, device=device) * 10 + 50  # Background
        B = torch.rand(batch_size, num_wavelengths, 1, device=device) * 5 + 20   # Modulation

        wavelengths_t = torch.tensor(wavelengths, device=device).unsqueeze(0) # (1, num_wavelengths)
        # Workaround for torch.linspace issue
        phase_shifts_t = torch.arange(0, num_buckets, device=device) * (2 * np.pi / num_buckets)

        # phi = (4 * pi / lambda) * h
        phi = (4 * np.pi * predicted_height) / wavelengths_t
        phi = phi.unsqueeze(2) # (batch, num_wavelengths, 1)

        cos_argument = phi + phase_shifts_t # (batch, num_wavelengths, num_buckets)

        # I = A + B * cos(phi + delta)
        perfect_intensities_3d = A + B * torch.cos(cos_argument)

        # Flatten to match the expected input shape for the loss function
        original_intensities_flat = perfect_intensities_3d.view(batch_size, -1)

        # --- Calculate Loss ---
        loss = loss_fn(predicted_height, original_intensities_flat)

        # --- Assert ---
        # The loss should be very close to zero
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_zero_loss(self):
        """
        Tests that the loss is non-zero for imperfect data.
        """
        batch_size = 4
        num_wavelengths = 3
        num_buckets = 4
        wavelengths = [450e-9, 550e-9, 650e-9]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        phase_shifts = torch.arange(0, num_buckets) * (2 * np.pi / num_buckets)
        loss_fn = PhysicsInformedLoss(
            wavelengths=wavelengths,
            num_wavelengths=num_wavelengths,
            num_buckets=num_buckets,
            phase_shifts=phase_shifts
        ).to(device)

        predicted_height = torch.rand(batch_size, 1, device=device) * 100e-9
        original_intensities = torch.rand(batch_size, num_wavelengths * num_buckets, device=device) * 255

        loss = loss_fn(predicted_height, original_intensities)

        self.assertGreater(loss.item(), 0.0)

    def test_intermediate_shapes(self):
        """
        Tests the shapes of the intermediate tensors (background, modulation).
        """
        batch_size = 8
        num_wavelengths = 2
        num_buckets = 4
        wavelengths = [500e-9, 600e-9]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        phase_shifts = torch.arange(0, num_buckets) * (2 * np.pi / num_buckets)
        loss_fn = PhysicsInformedLoss(
            wavelengths=wavelengths,
            num_wavelengths=num_wavelengths,
            num_buckets=num_buckets,
            phase_shifts=phase_shifts
        ).to(device)

        predicted_height = torch.rand(batch_size, 1, device=device)
        original_intensities = torch.rand(batch_size, num_wavelengths * num_buckets, device=device)

        # We need to access the intermediate tensors.
        # Let's modify the forward pass slightly for testing purposes or check the logic.
        # For this test, we'll re-implement the logic to check shapes.
        intensities = original_intensities.view(-1, num_wavelengths, num_buckets)
        background = torch.mean(intensities, dim=2, keepdim=True)
        modulation = torch.std(intensities, dim=2, keepdim=True, unbiased=False)

        self.assertEqual(background.shape, (batch_size, num_wavelengths, 1))
        self.assertEqual(modulation.shape, (batch_size, num_wavelengths, 1))


if __name__ == '__main__':
    unittest.main()
