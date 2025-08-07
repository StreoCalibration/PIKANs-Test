import torch
import torch.nn as nn
import numpy as np

class PhysicsInformedLoss(nn.Module):
    """
    Calculates a physics-informed loss based on the interferometry equation:
    I = A + B * cos(phi + delta)

    The loss is the Mean Squared Error (MSE) between the original intensities
    measured by the sensor and the intensities reconstructed using the model's
    predicted height map.
    """
    def __init__(self, wavelengths, num_wavelengths, num_buckets):
        """
        Args:
            wavelengths (torch.Tensor or list): A tensor or list containing the wavelengths used.
            num_wavelengths (int): The number of different wavelengths.
            num_buckets (int): The number of phase shift buckets.
        """
        super().__init__()
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
        self.register_buffer('wavelengths', wavelengths)
        
        self.num_wavelengths = num_wavelengths
        self.num_buckets = num_buckets

        # Define phase shifts based on the number of buckets.
        # This assumes a standard, evenly spaced phase-shifting setup.
        phase_shifts = torch.linspace(0, 2 * np.pi, num_buckets, endpoint=False)
        self.register_buffer('phase_shifts', phase_shifts) # Shape: (num_buckets,)

    def forward(self, predicted_height, original_intensities):
        """
        Calculates the physics-informed loss.

        Args:
            predicted_height (torch.Tensor): The height map predicted by the KAN model.
                                             Shape: (batch_size, 1)
            original_intensities (torch.Tensor): The raw intensity values from the sensor.
                                                 Shape: (batch_size, num_wavelengths * num_buckets)

        Returns:
            torch.Tensor: The calculated physics loss value.
        """
        # Reshape intensities for easier processing
        # Shape: (batch_size, num_wavelengths, num_buckets)
        intensities = original_intensities.view(-1, self.num_wavelengths, self.num_buckets)
        device = intensities.device

        # --- Estimate A (background) and B (modulation) from original intensities ---
        # A is the average intensity per pixel per wavelength
        A = torch.mean(intensities, dim=2, keepdim=True)  # Shape: (batch_size, num_wavelengths, 1)
        
        # B is related to the amplitude of the sinusoid, approximated by sqrt(2) * std(I)
        B = torch.sqrt(torch.tensor(2.0, device=device)) * torch.std(intensities, dim=2, keepdim=True)
        # Add a small epsilon to B to prevent it from being zero.
        B = B + 1e-8 # Shape: (batch_size, num_wavelengths, 1)


        # --- Reconstruct intensities using the physical model ---
        # Calculate the phase from the predicted height
        # phi = (4 * pi / lambda) * h
        phi = (4 * np.pi * predicted_height.to(device)) / self.wavelengths.unsqueeze(0) # Shape: (batch_size, num_wavelengths)
        phi = phi.unsqueeze(2) # Shape: (batch_size, num_wavelengths, 1)

        # Calculate the full phase argument for the cosine function
        cos_argument = phi + self.phase_shifts.to(device) # Shape: (batch_size, num_wavelengths, num_buckets)

        # Reconstruct the intensities
        reconstructed_intensities = A + B * torch.cos(cos_argument)

        # --- Calculate the loss ---
        # The loss is the mean squared error between original and reconstructed intensities
        loss = nn.functional.mse_loss(reconstructed_intensities, intensities)

        return loss