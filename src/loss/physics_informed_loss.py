import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustPhysicsLoss(nn.Module):
    def __init__(self, wavelengths, adaptive_mode=True):
        super().__init__()
        self.register_buffer('wavelengths', wavelengths)
        self.adaptive_mode = adaptive_mode
        
        # 학습 가능한 위상간격 (백업용)
        init_shifts = torch.tensor([0.0, torch.pi/2, torch.pi, 3*torch.pi/2])
        self.learnable_shifts = nn.Parameter(init_shifts.clone())
        
    def forward(self, predicted_height, original_intensities):
        intensities_reshaped = original_intensities.view(-1, 3, 4)
        
        if self.adaptive_mode:
            # Carré 알고리즘 사용 (위상간격 무관)
            return self.carre_loss(predicted_height, intensities_reshaped)
        else:
            # 학습된 위상간격 사용
            return self.standard_loss(predicted_height, intensities_reshaped)
    
    def carre_loss(self, predicted_height, intensities):
        """Carré 기반 강건한 손실"""
        total_loss = 0
        
        for w in range(3):
            I = intensities[:, w, :]
            I1, I2, I3, I4 = I[:, 0], I[:, 1], I[:, 2], I[:, 3]
            
            # Carré 위상 추출
            actual_phase = torch.atan2(I4 - I2, I1 - I3 + 1e-8)
            
            # 예측 위상
            pred_phase = (4 * torch.pi * predicted_height.squeeze()) / self.wavelengths[w]
            
            # 위상 정합 손실
            phase_loss = 1 - torch.cos(actual_phase - pred_phase)
            total_loss += torch.mean(phase_loss)
            
        return total_loss / 3