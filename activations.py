import torch
import torch.nn as nn
import torch.nn.functional as F



class bSpline(nn.Module):
    def __init__(self, num_control_points, degree, start_point, end_point, init):
        super(bSpline, self).__init__()
        self.num_control_points = num_control_points
        self.degree = degree
        self.start_point = start_point
        self.end_point = end_point

        # Init methods
        if init == 'relu':
            x = torch.linspace(start_point, end_point, num_control_points)
            cp = torch.clamp(x, min=0)
        elif init == 'leaky_relu':
            x = torch.linspace(start_point, end_point, num_control_points)
            cp = torch.where(x >= 0, x, 0.1 * x)
        elif init == 'identity':
            cp = torch.linspace(start_point, end_point, num_control_points)
        else:
            # cp = torch.randn(num_control_points) * 0.1
            cp = torch.randn(num_control_points) * 1

        self.control_points = nn.Parameter(cp)

        # Knot vector
        knots = torch.linspace(start_point, end_point, num_control_points + degree + 1)
        self.register_buffer('knots', knots)

    def forward(self, x):
        n = self.num_control_points
        k = self.degree
        x_shape = x.shape
        x_flat = x.reshape(-1, 1)

        # Degree 0
        knots_left = self.knots[:-1].unsqueeze(0)
        knots_right = self.knots[1:].unsqueeze(0)
        basis = ((x_flat >= knots_left) & (x_flat < knots_right)).float()


        basis[:, -1] = torch.where(
            x_flat.squeeze(-1) == self.knots[-1],
            torch.ones_like(x_flat.squeeze(-1)),
            basis[:, -1]
        )

        
        for deg in range(1, k + 1):
            num_basis = n + k - deg

            k_i = self.knots[:num_basis]
            k_i_deg = self.knots[deg:deg+num_basis]
            k_i1 = self.knots[1:num_basis+1]
            k_i1_deg = self.knots[deg+1:deg+num_basis+1]
            
            eps = 1e-10
            denom1 = k_i_deg - k_i
            denom2 = k_i1_deg - k_i1
            denom1_safe = torch.where(denom1.abs() > eps, denom1, torch.ones_like(denom1))
            denom2_safe = torch.where(denom2.abs() > eps, denom2, torch.ones_like(denom2))

            w1 = (x_flat - k_i) / denom1_safe
            w2 = (k_i1_deg - x_flat) / denom2_safe
            w1 = torch.where(denom1.abs() > eps, w1, torch.zeros_like(w1))
            w2 = torch.where(denom2.abs() > eps, w2, torch.zeros_like(w2))

            basis = w1 * basis[:, :num_basis] + w2 * basis[:, 1:num_basis+1]

        output = (basis[:, :n] * self.control_points).sum(dim=-1)

        return output.reshape(x_shape)


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

# m = bSpline(num_cp=5, degree=3, start=-2.0, end=2.0, init='identity')
# x_np = np.linspace(m.start, m.end, 100)
# x_torch = torch.tensor(x_np, dtype=torch.float32)
# y_torch = m(x_torch).detach().numpy()


# t = m.knots.cpu().numpy()
# c = m.cp.detach().cpu().numpy()
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
        act = bSpline(num_control_points=kwargs.get('num_control_points', 15), degree=kwargs.get('degree', 1), start_point=kwargs.get('start_point', -1.0),
                end_point=kwargs.get('end_point', 1.0), init=kwargs.get('init', 'leaky_relu')
        )
        return act
    else:
        raise ValueError('Unknown activation')

            

