import torch
import torch.nn as nn
from functools import partial
from .nerf import Embedding
from .nerf import NeRF  

class Sine(nn.Module):
    # refer : https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L28
    def __init__(self, alpha=30):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.alpha * input)
 

def get_acti(name, inplace=True, **kwargs):
    name = name.lower()
    if name == 'relu':
        acti = nn.ReLU(inplace=inplace)
    elif name == "siren":
        acti = Sine(**kwargs)
    else:
        raise NotImplementedError(name)
    return acti


class NeRF_Acti(NeRF):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="relu"):
        self.acti_type = acti_type
        super().__init__(D, W, in_channels_xyz, in_channels_dir, skips)
    
    def build_models(self):
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, get_acti(self.acti_type))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                get_acti(self.acti_type))

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(self.W//2, 3),
                        nn.Sigmoid())

NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="siren")
