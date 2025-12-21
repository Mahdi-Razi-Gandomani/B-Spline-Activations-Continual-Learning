import torch.nn as nn
from activations import get_activation



class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, act, act_cfg=None, shared_act=False):
        super().__init__()
        if act_cfg is None:
            act_cfg = {}

        layers = []
        acts = []
        prev = in_size
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if shared_act and i > 0:
                acts.append(acts[0])
            else:
                acts.append(get_activation(act, **act_cfg))
            
            prev = h

        layers.append(nn.Linear(prev, out_size))

        self.layers = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        act_idx = 0
        for layer in self.layers[ : -1]:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.acts[act_idx](x)
                act_idx += 1
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x



def create_model(model_type, in_size, out_size, act, **kwargs):

    if model_type == 'mlp':
        return MLP(in_size,kwargs.get('hidden_sizes', [256, 256]),out_size, act, kwargs.get('act_cfg', {}), kwargs.get('shared_act', False))
    else:
        raise ValueError('Unknown model')
