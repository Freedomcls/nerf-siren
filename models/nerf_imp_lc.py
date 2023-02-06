import torch
import torch.nn as nn
from functools import partial
from .nerf import Embedding
from .nerf import NeRF  
import numpy as np

class Sine(nn.Module):
    # refer : https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L28
    def __init__(self, alpha=30):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.alpha * input)

class Sine_1(nn.Module):
    # refer : https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L28
    def __init__(self, alpha=1):
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
        first_omega_0 = 30
        hidden_omega_0 = 1
        # hidden_omega_0 = 30.

        for i in range(self.D):
            if i == 0:
                # layer = nn.Linear(self.in_channels_xyz, self.W)
                layer = SineLayer(self.in_channels_xyz, self.W, 
                                  is_first=True, omega_0=first_omega_0)
            elif i in self.skips:
                # layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
                layer = SineLayer(self.W + self.in_channels_xyz, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            else:
                # layer = nn.Linear(self.W, self.W)
                layer = SineLayer(self.W, self.W, 
                                      is_first=False, omega_0=hidden_omega_0)
            
            # layer = nn.Sequential(layer, get_acti(self.acti_type))

            # if i == 0:
            #     layer = nn.Sequential(layer, get_acti(self.acti_type))
            # else:
            #     layer = nn.Sequential(layer, Sine_1())
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)
        # with torch.no_grad():
        #         self.xyz_encoding_final.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
        #                                       np.sqrt(6 / self.W) / hidden_omega_0)

        # direction encoding layers
        # self.dir_encoding = nn.Sequential(
        #                         nn.Linear(self.W + self.in_channels_dir, self.W//2),
        #                         # nn.ReLU(True))
        #                         # get_acti(self.acti_type))
        #                         Sine_1())
        self.dir_encoding = SineLayer(self.W + self.in_channels_dir, self.W//2, 
                                      is_first=False, omega_0=hidden_omega_0)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        with torch.no_grad():
                self.sigma.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)
        # self.rgb = nn.Sequential(
        #                 nn.Linear(self.W//2, 3),
        #                 nn.Sigmoid())

        # m = nn.Sigmoid()
        self.rgb = nn.Linear(self.W//2, 3)
        with torch.no_grad():
                self.rgb.weight.uniform_(-np.sqrt(6 / self.W) / hidden_omega_0, 
                                              np.sqrt(6 / self.W) / hidden_omega_0)
        self.rgb = nn.Sequential(
                        self.rgb,
                        nn.Sigmoid())

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0, 
                                             np.sqrt(6/self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
    
class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3,
                 hidden_features=256, hidden_layers=4, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

# NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=3, in_channels_dir=3, skips=[4], acti_type="siren")
NeRF_Siren = partial(NeRF_Acti,  D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="siren")
# NeRF_Siren = partial(Siren,  in_features=63, out_features=3,
#                  hidden_features=256, hidden_layers=4, outermost_linear=False, 
#                  first_omega_0=30, hidden_omega_0=30.)


class NeRFFeats(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRFFeats, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.feats_C = 64  # hard code

        self.skips = skips
        self.build_models()


    def build_models(self):
        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(self.W//2, 3),
                        nn.Sigmoid())
        
        self.render_feats = nn.Sequential(
                                nn.Linear(self.W, self.W),
                                nn.ReLU(True),
                                nn.Linear(self.W, self.W//2),
                                nn.ReLU(True),
                                nn.Linear(self.W//2, self.feats_C),
                                nn.ReLU(True))
        
    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        feats = self.render_feats(xyz_)

        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)  # add input_dir to get RGB
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma, feats], -1)  # 3 1 64

        return out
        