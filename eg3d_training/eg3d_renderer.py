import torch
# from torch_utils import misc
from eg3d_training.triplane import TriPlaneGenerator
import numpy as np

class EG3D_Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # device = 'cpu'
        device = torch.device("cuda")
        seed = 0
        init_args = ()

        # init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 16384, 'channel_max': 256, 'fused_modconv_default': 'inference_only', 'rendering_kwargs': {'depth_resolution': 64, 'depth_resolution_importance': 64, 
        # 'ray_start': 2.0, 'ray_end': 6.0, 'box_warp': 3.0, 'avg_camera_radius': 1.7, 'white_back': True,'avg_camera_pivot': [0, 0, 0], 
        # 'image_resolution': 128, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 
        # 'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC', 'c_gen_conditioning_zero': False, 
        # 'gpc_reg_prob': 0.8, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 
        # 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4, 
        # 'sr_kwargs': {'channel_base': 16384, 'channel_max': 256, 'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 25, 'img_resolution': 128, 'img_channels': 3}

        # init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only', 'rendering_kwargs': {'depth_resolution': 64, 'depth_resolution_importance': 64, 
        # 'ray_start': 2.0, 'ray_end': 6.0, 'box_warp': 3.0, 'avg_camera_radius': 1.7, 'white_back': True,'avg_camera_pivot': [0, 0, 0], 
        # 'image_resolution': 128, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 
        # 'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC', 'c_gen_conditioning_zero': False, 
        # 'gpc_reg_prob': 0.8, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 
        # 'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4, 
        # 'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 0, 'img_resolution': 128, 'img_channels': 3}
        
        init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only', 'rendering_kwargs': {'depth_resolution': 64, 'depth_resolution_importance': 64, 
        'ray_start': 0.1, 'ray_end': 10.0, 'box_warp': 15.0, 'avg_camera_radius': 2.7, 'white_back': False,'avg_camera_pivot': [0, 0, 0], 
        'image_resolution': 128, 'disparity_space_sampling': False, 'clamp_mode': 'softplus', 
        'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC', 'c_gen_conditioning_zero': False, 
        'gpc_reg_prob': 0.8, 'c_scale': 1.0, 'superresolution_noise_mode': 'none', 'density_reg': 0.25, 'density_reg_p_dist': 0.004, 
        'reg_type': 'l1', 'decoder_lr_mul': 1.0, 'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4, 
        'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'}, 'conv_clamp': None, 'c_dim': 0, 'img_resolution': 128, 'img_channels': 3}

        self.G = TriPlaneGenerator(*init_args, **init_kwargs)
        self.z = torch.nn.Parameter(torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)))
        # G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        # misc.copy_params_and_buffers(G, G_new, require_all=True)
        # G_new.neural_rendering_resolution = G.neural_rendering_resolution
        # G_new.rendering_kwargs = G.rendering_kwargs
        # G = G_new

    # def render(self, conditioning_params, ray_origins, ray_directions, semantic_map):
    def render(self, conditioning_params, ray_origins, ray_directions):
        truncation_psi = 1
        truncation_cutoff = None
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)
        ws = self.G.mapping(self.z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # img_dict = self.G.synthesis(ws, ray_origins, ray_directions)

        img_dict = self.G.synthesis2(ws,ray_origins, ray_directions)
        # img_dict = self.G.synthesis2(ws,ray_origins, ray_directions, semantic_map)
        img_dict['rgb_fine'] = img_dict['rgb_fine'].squeeze(0)
        img_dict['depth_fine'] = img_dict['depth_fine'].squeeze(0)
        img_dict['opacity_fine'] = img_dict['opacity_fine'].squeeze(0)
        img_dict['rgb_coarse'] = img_dict['rgb_coarse'].squeeze(0)
        img_dict['depth_coarse'] = img_dict['depth_coarse'].squeeze(0)
        img_dict['opacity_coarse'] = img_dict['opacity_coarse'].squeeze(0)
        return img_dict

    def sample(self, coordinates, directions):
        conditioning_params = -1
        return self.G.sample(coordinates, directions, self.z, conditioning_params)
       

if __name__ == '__main__':
    eg3d_renderer = EG3D_Renderer()
    conditioning_params = -1
    # conditioning_params = torch.ones(1,12)
    rayo = torch.ones((128*128,3))
    rayd = torch.ones((128*128,3))
    image_dict = eg3d_renderer.render(conditioning_params, rayo, rayd)
    a = image_dict['rgb_fine']
    print(torch.max(a),torch.min(a))
    b = image_dict['depth_fine']
    c = image_dict['opacity_fine']
    print(a.shape,b.shape,c.shape)
    print(1)
