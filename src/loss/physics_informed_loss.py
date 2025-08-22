"""
Physics-informed loss function for interferometry.
"""
import math
from typing import List, Union
import torch
from torch import nn, Tensor


class PhysicsInformedLoss(nn.Module):
    """
    Calculates a physics-informed loss based on the interferometry equation:
    I = A + B * cos(phi + delta)

    The loss is the Mean Squared Error (MSE) between the original intensities
    measured by the sensor and the intensities reconstructed using the model's
    predicted height map.
    """
    def __init__(self, wavelengths: Union[List[float], Tensor],
                 num_wavelengths: int, num_buckets: int, phase_shifts: Tensor):
        """
        Args:
            wavelengths: A tensor or list containing the wavelengths used.
            num_wavelengths: The number of different wavelengths.
            num_buckets: The number of phase shift buckets.
            phase_shifts: A tensor containing the phase shifts.
        """
        super().__init__()
        if not isinstance(wavelengths, Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
        self.register_buffer('wavelengths', wavelengths)
        self.register_buffer('phase_shifts', phase_shifts)

        self.num_wavelengths = num_wavelengths
        self.num_buckets = num_buckets

    def forward(self, predicted_height: Tensor, original_intensities: Tensor) -> Tensor:
        """
        Calculates the physics-informed loss.

        Args:
            predicted_height: The height map predicted by the KAN model.
                              Shape: (batch_size, 1)
            original_intensities: The raw intensity values from the sensor.
                                 Shape: (batch_size, num_wavelengths * num_buckets)

        Returns:
            The calculated physics loss value.
        """
        # Reshape intensities for easier processing
        # Shape: (batch_size, num_wavelengths, num_buckets)
        intensities = original_intensities.view(
            -1, self.num_wavelengths, self.num_buckets
        )
        device = intensities.device

        # --- Estimate background and modulation from original intensities ---
        # Background is the average intensity per pixel per wavelength
        background = torch.mean(intensities, dim=2, keepdim=True)

        # Modulation is related to the amplitude of the sinusoid,
        # approximated by sqrt(2) * std(I).
        # We use unbiased=False for the population standard deviation.
        modulation = torch.sqrt(
            torch.tensor(2.0, device=device)
        ) * torch.std(intensities, dim=2, keepdim=True, unbiased=False)
        # Add a small epsilon to prevent it from being zero.
        modulation = modulation + 1e-8

        # --- Reconstruct intensities using the physical model ---
        # Calculate the phase from the predicted height: phi = (4 * pi / lambda) * h
        phi = (4 * math.pi * predicted_height.to(device)) / self.wavelengths.unsqueeze(0)
        phi = phi.unsqueeze(2)  # Shape: (batch_size, num_wavelengths, 1)

        # Calculate the full phase argument for the cosine function
        cos_argument = phi + self.phase_shifts.to(device)

        # Reconstruct the intensities
        reconstructed_intensities = background + modulation * torch.cos(cos_argument)

        # --- Calculate the loss ---
        # The loss is the MSE between original and reconstructed intensities
        loss = nn.functional.mse_loss(reconstructed_intensities, intensities)

        return loss