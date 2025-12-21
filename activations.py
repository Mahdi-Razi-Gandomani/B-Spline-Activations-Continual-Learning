import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class bSpline(nn.Module):
    def __init__(self, num_control_points, degree, start_point, end_point, init='relu', grid_size=1000):
        super(bSpline, self).__init__()
        self.num_control_points = num_control_points
        self.degree = degree
        self.start_point = start_point
        self.end_point = end_point
        self.grid_size = grid_size
        
        # Init methods
        if init == 'relu':
            x = torch.linspace(start_point, end_point, num_control_points)
            cp = torch.where(x >= 0, x, 0)
        elif init == 'leaky_relu':
            x = torch.linspace(start_point, end_point, num_control_points)
            cp = torch.where(x >= 0, x, 0.1 * x)
        elif init == 'identity':
            cp = torch.linspace(start_point, end_point, num_control_points)
        else:
            cp = torch.randn(num_control_points) * 1
        
        self.control_points = nn.Parameter(cp)
        
        # Uniform knots
        knots = torch.linspace(start_point, end_point, num_control_points + degree + 1)
        self.register_buffer('knots', knots)
        
        # Precompute basis functions
        self.register_buffer('grid_points', torch.linspace(start_point, end_point, grid_size))
        self._precompute_basis()
    
    def _compute_basis(self, x):
        n = self.num_control_points
        k = self.degree
        
        # Degree 0
        knots_left = self.knots[:-1].unsqueeze(0)
        knots_right = self.knots[1:].unsqueeze(0)
        basis = ((x >= knots_left) & (x < knots_right)).float()
    
        basis[:, -1] = torch.where(x.squeeze(-1) == self.knots[-1], torch.ones_like(x.squeeze(-1)), basis[:, -1])
        

        for deg in range(1, k + 1):
            num_basis = n + k - deg
            k_i = self.knots[: num_basis]
            k_i_deg = self.knots[deg : deg+num_basis]
            k_i1 = self.knots[1 : num_basis+1]
            k_i1_deg = self.knots[deg+1 : deg+num_basis+1]
            
            eps = 1e-10
            d1 = k_i_deg - k_i
            d2 = k_i1_deg - k_i1
            d1_safe = torch.where(d1.abs() > eps, d1, torch.ones_like(d1))
            d2_safe = torch.where(d2.abs() > eps, d2, torch.ones_like(d2))
            
            w1 = (x - k_i) / d1_safe
            w2 = (k_i1_deg - x) / d2_safe
            w1 = torch.where(d1.abs() > eps, w1, torch.zeros_like(w1))
            w2 = torch.where(d2.abs() > eps, w2, torch.zeros_like(w2))
            
            basis = w1 * basis[:, : num_basis] + w2 * basis[:, 1 : num_basis+1]
        
        return basis[:, : n]
    
    def _precompute_basis(self):
        x = self.grid_points.unsqueeze(1)
        basis = self._compute_basis(x)
        self.register_buffer('basis_grid', basis)
    

    def forward(self, x):
        x_flat = x.reshape(-1)
        
        x_clamped = torch.clamp(x_flat, self.start_point, self.end_point)
        ind = (x_clamped - self.start_point) / (self.end_point - self.start_point) * (self.grid_size - 1)
        
        # Interpolation
        ind_floor = torch.floor(ind).long()
        ind_ceil = torch.clamp(ind_floor + 1, max=self.grid_size - 1)
        ind_floor = torch.clamp(ind_floor, min = 0, max=self.grid_size - 1)
        
        weight = ind - ind_floor.float()
        
        basis_floor = self.basis_grid[ind_floor]
        basis_ceil = self.basis_grid[ind_ceil]
        
        basis_interp = basis_floor + weight.unsqueeze(-1) * (basis_ceil - basis_floor)
        output = (basis_interp * self.control_points).sum(dim=-1)

        
        return output.reshape(x.shape)



class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


    

# import torch
# import torch.nn as nn
# from scipy.interpolate import BSpline
# import numpy as np

# m = bSpline(num_control_points=5, degree=3, start_point=-2.0, end_point=2.0, init='identity')
# x_np = np.linspace(m.start_point, m.end_point, 100)
# x_torch = torch.tensor(x_np, dtype=torch.float32)
# y_torch = m(x_torch).detach().numpy()


# t = m.knots.cpu().numpy()
# c = m.control_points.detach().cpu().numpy()
# k = m.degree
# bs = BSpline(t, c, k)
# y_scipy = bs(x_np)
# print(y_torch)
# print(y_scipy)

# print(np.allclose(y_torch, y_scipy, 1e-6))


def get_activation(name, **kwargs):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'swish':
        return Swish(beta=kwargs.get('beta', 1.0))
    elif name == 'bspline':
        act = bSpline(num_control_points=kwargs.get('num_control_points', 20), degree=kwargs.get('degree', 3), start_point=kwargs.get('start_point', -2.0),
                end_point=kwargs.get('end_point', 2.0), init=kwargs.get('init', 'relu'), grid_size=kwargs.get('grid_size', 1000)
        )
        return act
    else:
        raise ValueError('Unknown activation')

            

