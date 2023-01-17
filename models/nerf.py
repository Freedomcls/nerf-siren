import torch
from torch import nn
import numpy as np
from models import swin_encoder
from PIL import Image
from opt import get_opts
from models.train_utils import requires_grad
import os
import torchvision.transforms as transforms
from datasets import augmentations
import random

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
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
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
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
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out
        

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    # def forward(self, x):
    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        # print("xxxxxxxx",x,x.shape)
        # print("freq",freq,freq.shape)
        # print("phshi",phase_shift,phase_shift.shape)
        
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        # freq = 30.
        # phase_shift = 0
        return torch.sin(freq * x + phase_shift)

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init
    
def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class SemanticNeRF(NeRF):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="relu"):
    # def __init__(self, D=8, W=256, in_channels_xyz=3, in_channels_dir=3, skips=[4], acti_type="relu"):
        self.acti_type = acti_type
        super().__init__(D, W, in_channels_xyz, in_channels_dir, skips)

        self.hparams = get_opts()
        # self.img = torch.nn.Parameter(torch.zeros([1, 99, 224, 224]))
        # nn.init.kaiming_uniform_(self.img, mode='fan_in', nonlinearity='relu')

        batch_size = self.hparams.batch_size
        # self.img = self.img[:batch_size]
        # self.img = self.img.cuda()

        self.network = nn.ModuleList([
            FiLMLayer(self.in_channels_xyz, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W + self.in_channels_xyz, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
            FiLMLayer(self.W, self.W),
        ])
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        self.final_layer = nn.Linear(self.W, 1)

        self.color_layer_sine = FiLMLayer(self.W + self.in_channels_dir, self.W)
        self.color_layer_linear = nn.Sequential(nn.Linear(self.W, 3))

        # self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.encoder = self.set_encoder()
        self.load_weights()

        self.source_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                augmentations.ToOneHot(99),
                transforms.ToTensor()])

        self.image = self.load_data()
        # self.img = self.image.view(1, *self.image.size()).float()

        self.img = [x.view(1, *x.size()).float() for x in self.image]

        # self.img = torch.nn.Parameter(self.img)
        # self.img = [torch.nn.Parameter(x) for x in self.img]
        self.img = torch.nn.ParameterList([torch.nn.Parameter(x,requires_grad=False) for x in self.img])

        # self.gridwarper = UniformBoxWarp(51)  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.
    
    def load_data(self):
        from_ims = []
        for filename in os.listdir('/home/chenlinsheng/3D-nerf-da/semantic-masks'):
            # from_path_1 = self.source_paths[index]
            # print(index)
            from_im = Image.open('/home/chenlinsheng/3D-nerf-da/semantic-masks/' + filename)
            from_im = from_im.convert('L')
            # from_im = self.update_labels(from_im)
            from_im = self.source_transform(from_im)
            from_ims.append(from_im)
        # idx = random.randint(1, 14)
        # from_im = Image.open('/home/chenlinsheng/3D-nerf-da/semantic-masks/' + str(idx) + '.png')
        # from_im = from_im.convert('L')
        # # from_im = self.update_labels(from_im)
        # from_im = self.source_transform(from_im)
        return from_ims

    def load_weights(self):        
        # Load pretrained weights for SwinEncoder
        print('Loading encoders weights from swin_tiny_patch4_window7_224!')
        encoder_ckpt = torch.load('/home/chenlinsheng/3D-nerf-da/pretrained_model/swin_tiny_patch4_window7_224.pth', map_location='cpu')['model']
        # if self.opts.label_nc != 0:
        encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('patch_embed' in k or 'head' in k)}
        # else:
        #     encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('head' in k)}
        self.encoder.load_state_dict(encoder_ckpt, strict=False)


    def set_encoder(self):
        # hard code to match 'swin_tiny_patch4_window7_224.yaml' at the moment
        encoder = swin_encoder.SwinTransformer(
            # img_size=224, patch_size=4, in_chans=99, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], 
            img_size=224, patch_size=4, in_chans=99, num_classes=512*9, embed_dim=96, depths=[2, 2, 6, 2], 
            num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False
        )
        
        return encoder
    
    def requires_grad(self, enc_flag):
        requires_grad(self.encoder, enc_flag)

    
    # def forward(self, x, device, sigma_only=False):
    def forward(self, x, sigma_only=False):
        # input = self.gridwarper(input)
        
        # batch_size = self.hparams.batch_size
        # img = img[:batch_size]
        # img = img.cuda()
        # print("img",self.img)
        # self.img = [x[:1].to(device) for x in self.img]
        # self.img.to(device)
        # print("img", self.img, self.img.shape)
        
        enc_out_dict = self.encoder(self.img)
        frequencies, phase_shifts = enc_out_dict['latent_code']
        # frequencies_1 = frequencies*15
        frequencies = frequencies*15 + 30

        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for index, layer in enumerate(self.network):
            start = index * self.W
            end = (index+1) * self.W
            # print("index",index)
            if index == 4:
                xyz_ = torch.cat([input_xyz, xyz_], -1)  
                # print("frequencies",frequencies[..., start:end].shape)
                # print("xyz_", xyz_.shape)
                # print("phase_shifts", phase_shifts[..., start:end].shape)
                xyz_ = layer(xyz_, frequencies[..., start:end], phase_shifts[..., start:end])  # xyz [40000, 256] freq&phase [1, 256]
                # xyz_ = layer(xyz_)
            else:
                # print("frequencies",frequencies[..., start:end].shape)
                # print("xyz_", xyz_.shape)
                # print("phase_shifts", phase_shifts[..., start:end].shape)
                xyz_ = layer(xyz_, frequencies[..., start:end], phase_shifts[..., start:end])
                # xyz_ = layer(xyz_)

        sigma = self.final_layer(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        # dir_encoding_input = torch.cat([xyz_, input_dir], -1)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.color_layer_sine(dir_encoding_input,frequencies[..., -self.W:], phase_shifts[..., -self.W:])
        # dir_encoding = self.color_layer_sine(dir_encoding_input)

        # rgb = self.rgb(dir_encoding)
        rgb = torch.sigmoid(self.color_layer_linear(dir_encoding))
        
        out = torch.cat([rgb, sigma], -1)

        return out

class swinNeRF(NeRF):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4], acti_type="relu"):
    # def __init__(self, D=8, W=256, in_channels_xyz=3, in_channels_dir=3, skips=[4], acti_type="relu"):
        self.acti_type = acti_type
        super().__init__(D, W, in_channels_xyz, in_channels_dir, skips)
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        self.build_models()
        
        self.encoder = self.set_encoder()
        self.load_weights()

        # self.img = torch.nn.Parameter(torch.zeros([1, 99, 224, 224]))
        # nn.init.kaiming_uniform_(self.img, mode='fan_in', nonlinearity='relu')
        
        self.source_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                augmentations.ToOneHot(99),
                transforms.ToTensor()])

        # self.img = self.image.view(1, *self.image.size()).float()
        # self.img = torch.nn.Parameter(self.img)
        
        self.image = self.load_data()
        self.img = [x.view(1, *x.size()).float() for x in self.image]
        self.img = torch.nn.ParameterList([torch.nn.Parameter(x,requires_grad=False) for x in self.img])

    def load_data(self):
        from_ims = []
        for filename in os.listdir('/home/chenlinsheng/3D-nerf-da/semantic-masks'):
            # from_path_1 = self.source_paths[index]
            # print(index)
            from_im = Image.open('/home/chenlinsheng/3D-nerf-da/semantic-masks/' + filename)
            from_im = from_im.convert('L')
            # from_im = self.update_labels(from_im)
            from_im = self.source_transform(from_im)
            from_ims.append(from_im)
        # idx = random.randint(1, 14)
        # from_im = Image.open('/home/chenlinsheng/3D-nerf-da/semantic-masks/' + str(idx) + '.png')
        # from_im = from_im.convert('L')
        # # from_im = self.update_labels(from_im)
        # from_im = self.source_transform(from_im)
        return from_ims

    def load_weights(self):        
        # Load pretrained weights for SwinEncoder
        print('Loading encoders weights from swin_tiny_patch4_window7_224!')
        encoder_ckpt = torch.load('/home/chenlinsheng/3D-nerf-da/pretrained_model/swin_tiny_patch4_window7_224.pth', map_location='cpu')['model']
        # if self.opts.label_nc != 0:
        encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('patch_embed' in k or 'head' in k)}
        # else:
        #     encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if not ('head' in k)}
        self.encoder.load_state_dict(encoder_ckpt, strict=False)


    def set_encoder(self):
        # hard code to match 'swin_tiny_patch4_window7_224.yaml' at the moment
        encoder = swin_encoder.SwinTransformer(
            # img_size=224, patch_size=4, in_chans=99, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], 
            img_size=224, patch_size=4, in_chans=99, num_classes=512*9, embed_dim=96, depths=[2, 2, 6, 2], 
            num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False
        )
        
        return encoder
    
    def requires_grad(self, enc_flag):
        requires_grad(self.encoder, enc_flag)

    def build_models(self):
        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer)
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

    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        else:
            raise TypeError("Invalid type for move_to")
        
    # def forward(self, x, device, sigma_only=False):
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
        # img = [x.to(device) for x in img]
        
        # net = self.encoder
        # devices = []
        # sd = net.state_dict()
        # for v in sd.values():
        #     device = v.device

       
        # self.move_to(self.img, device)
        
        
        # feature = self.encoder(self.img)
        
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
            start = i * self.W
            end = (i+1) * self.W
            # print("index",index)
            m = nn.ReLU()
            # print("xyz_shape", xyz_.shape)
            # print("feature", feature[..., start:end].shape)
            # xyz_ = xyz_ + feature[..., start:end]
            xyz_ = m(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out