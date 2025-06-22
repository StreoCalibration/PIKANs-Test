import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    KAN Layer implementation based on B-splines.
    Each connection between an input and output neuron is represented by a
    learnable activation function, which is a sum of a base function and a spline function.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, base_activation=nn.SiLU):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid for splines. It's fixed and not learnable.
        # The grid is defined over a range, e.g., [-1, 1], assuming normalized inputs.
        grid_range = torch.linspace(-1.0, 1.0, grid_size + 1)
        self.register_buffer('grid', grid_range.unsqueeze(0).repeat(in_features, 1))
        # self.grid shape: (in_features, grid_size + 1)

        # Learnable spline coefficients. Number of basis functions = grid_size + spline_order.
        num_basis = grid_size + spline_order
        self.spline_coeffs = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        nn.init.xavier_uniform_(self.spline_coeffs)

        # Learnable weights for the base and spline functions to scale them.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_scaler)

        # Base activation function
        self.base_activation = base_activation()

    def b_spline_basis(self, x):
        """
        Computes the B-spline basis function values for the given input x.
        Uses the Cox-de Boor recursion formula in a vectorized manner.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: B-spline basis values of shape (batch_size, in_features, num_basis).
        """
        # Ensure grid and x have correct shapes for broadcasting
        grid = self.grid.unsqueeze(0)  # (1, in_features, grid_size + 1)
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)

        # Create the full knot vector by padding the grid
        first_knot = grid[:, :, :1]
        last_knot = grid[:, :, -1:]
        knots = torch.cat([
            first_knot.expand(-1, -1, self.spline_order),
            grid,
            last_knot.expand(-1, -1, self.spline_order)
        ], dim=-1)
        # knots shape: (1, in_features, grid_size + 1 + 2 * spline_order)

        # Cox-de Boor recursion
        # Order 0 (degree -1) basis
        basis = ((x >= knots[:, :, :-1]) & (x < knots[:, :, 1:])).to(x.dtype)

        # Higher order basis
        for k in range(1, self.spline_order + 1):
            den1 = knots[:, :, k:-1] - knots[:, :, :-k-1]
            den2 = knots[:, :, k+1:] - knots[:, :, 1:-k]
            
            # Avoid division by zero
            den1[den1 == 0] = 1.0
            den2[den2 == 0] = 1.0
            
            term1 = (x - knots[:, :, :-k-1]) / den1 * basis[:, :, :-1]
            term2 = (knots[:, :, k+1:] - x) / den2 * basis[:, :, 1:]
            basis = term1 + term2
            
        return basis

    def forward(self, x):
        # 1. Base function part: y_base_j = sum_i w_base_{j,i} * silu(x_i)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. Spline function part
        spline_basis = self.b_spline_basis(x)  # (batch, in_features, num_basis)
        spline_activation = torch.einsum('bin,oin->boi', spline_basis, self.spline_coeffs)
        spline_output = torch.einsum('boi,oi->bo', spline_activation, self.spline_scaler)

        # 3. Combine base and spline parts
        return base_output + spline_output