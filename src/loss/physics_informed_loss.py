import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    """
    Physics-Informed Loss Function.
    This loss penalizes predictions that violate the physical principles
    of interferometry. It reconstructs the intensity values from the
    predicted height and compares them to the original input intensities.
    The underlying physical model is I = A + B*cos(phi), where phi is
    the phase related to height.
    """
    def __init__(self, wavelengths):
        super(PhysicsLoss, self).__init__()
        if not isinstance(wavelengths, torch.Tensor):
            raise TypeError("wavelengths must be a torch.Tensor")
        # 파장 값을 버퍼로 등록하여 모델과 동일한 장치(CPU/GPU)로 자동 이동되도록 함
        self.register_buffer('wavelengths', wavelengths)
        print("Physics-Informed Loss initialized with physical model.")

    def forward(self, predicted_height, original_intensities):
        """
        Calculates the physics-informed loss.

        Args:
            predicted_height (torch.Tensor): The height map predicted by the model.
                                             Shape: (batch_size, 1).
            original_intensities (torch.Tensor): The original 12-channel intensity values.
                                                 Shape: (batch_size, 12).

        Returns:
            torch.Tensor: The calculated physics-informed loss value.
        """
        # Reshape intensities to (batch_size, 3, 4) for 3 wavelengths, 4 buckets
        intensities_reshaped = original_intensities.view(-1, 3, 4)

        # Extract I1, I2, I3, I4 for each wavelength
        # Shape of each: (batch_size, 3)
        I1 = intensities_reshaped[:, :, 0]
        I2 = intensities_reshaped[:, :, 1]
        I3 = intensities_reshaped[:, :, 2]
        I4 = intensities_reshaped[:, :, 3]

        # Estimate empirical background A and modulation B from original intensities
        # These are considered as ground truth physical parameters for the given pixel.
        # Shape of A_emp and B_emp: (batch_size, 3)
        A_emp = (I1 + I2 + I3 + I4) / 4.0
        B_emp = torch.sqrt((I1 - I3)**2 + (I4 - I2)**2) / 2.0

        # Calculate the predicted phase from the model's predicted height
        # self.wavelengths shape: (3,) -> view as (1, 3) for broadcasting
        # predicted_height shape: (batch_size, 1)
        # phi_pred shape: (batch_size, 3)
        phi_pred = (4 * torch.pi * predicted_height) / self.wavelengths.view(1, -1)

        # Reconstruct the 4-bucket intensities using the predicted phase and empirical A, B
        phase_shifts = torch.tensor([0.0, torch.pi / 2.0, torch.pi, 3.0 * torch.pi / 2.0], device=predicted_height.device).view(1, 1, -1)
        reconstructed_intensities = A_emp.unsqueeze(2) + B_emp.unsqueeze(2) * torch.cos(phi_pred.unsqueeze(2) + phase_shifts)

        # The loss is the Mean Squared Error between the original intensities
        # and the intensities reconstructed from the predicted height.
        loss = F.mse_loss(reconstructed_intensities, intensities_reshaped)

        return loss