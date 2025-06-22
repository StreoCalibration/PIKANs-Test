import torch
import torch.nn as nn
from .layers import KANLayer

class PIKANs(nn.Module):
    """
    Physics-Informed Kolmogorov-Arnold Network (PIKANs).
    As per Logical and Development View, this is the core network module.
    This model is composed of a sequence of KAN layers.
    """
    def __init__(self, layer_widths, grid_size=5, spline_order=3):
        super(PIKANs, self).__init__()
        
        print("PIKANs model initialized with KAN layers.")
        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            in_features, out_features = layer_widths[i], layer_widths[i+1]
            self.layers.append(
                KANLayer(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x