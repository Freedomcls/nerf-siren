import torch
# from torch_utils import misc
from eg3d_training.eg3d_renderer import EG3D_Renderer
import numpy as np

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