import torch
from torchsearchsorted import searchsorted
import os
import torch.nn.functional as F
import numpy as np
from .ConvNetWork import Voxelizer
import MinkowskiEngine as ME
import ast

DEBUG = ast.literal_eval(os.environ.get("DEBUG", "False"))

__all__ = ['render_rays', "render_rays_3d"]

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                _cls_num=6,
                network=None,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded

            # get outputs chunk by chunk 
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]
            

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights


    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result


def render_rays_3d(models,
                  embeddings,
                  rays,
                  N_samples=64,
                  use_disp=False,
                  perturb=0,
                  noise_std=1,
                  N_importance=0,
                  chunk=1024*32,
                  white_back=False,
                  test_time=False,
                  _cls_num=6,
                  no_grad_on_nerf=True,
                ):
    """
    render cls results.
    """
    def inference(model, points, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        # print(xyz_.shape, "xyz")

        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        # print(B, chunk, len(range(0, B, chunk)))
        
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # 3 channel encoder to 10
            if not weights_only:
                # training split
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            if no_grad_on_nerf:
                with torch.no_grad(): 
                    out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]
            else:
                out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]
            
        out = torch.cat(out_chunks, 0)
        # print(out.shape, xyz_.shape, N_rays, N_samples)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            # cls + 4 = 23
            rgbsigma = out.reshape(N_rays, N_samples_, -1) # ! need check shape
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)
            """
            NOTE: The class of xyz should be N_rays due to it is image-independent
            """
            if DEBUG:
                print(rgbsigma.shape, "?")

        
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # use weight to sample xyz
        N_sample = weights.shape[1]
        clspoints = torch.zeros((N_rays, N_sample, _cls_num)).cuda() # all is background
        # set thresh avoid oom 
        if  test_time:
            _thresh = 0.5
        else:
            _thresh = 0

        # sample_weights = weights
        sample_weights = weights
        sample_mask = sample_weights>_thresh
        sample_points = xyz_[sample_mask.reshape(-1)] 
        # normalize 
        norm_sp = np.linalg.norm(sample_points.detach().cpu().numpy()) # 1.6 not support torch.linalg.norm
        sample_points = sample_points / norm_sp

        rgbs_points = rgbs.reshape(-1,3)[sample_mask.reshape(-1)]

        sample_points = torch.cat([sample_points,  rgbs_points], dim=1) # pts, 6,
        sample_points = sample_points.transpose(1, 0) # 6, pts
        sample_points = torch.unsqueeze(sample_points, 0) # 1, 6, pts
        sample_points = sample_points.contiguous()
        points_preds, _, _ = points(sample_points)
        points_preds = points_preds[0] # pts = sample_masks
        # points_preds = F.log_softmax(points_preds, dim=1)
        clspoints[sample_mask] = points_preds # N rays, N_sample, cls
        
        # orginal rgb ways, use sum
        cls_final = torch.sum(weights.unsqueeze(-1)*clspoints, -2) # N_rays, cls

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        #  sum N_samples rgb results
        if DEBUG:
            import ipdb; ipdb.set_trace()
            print(torch.min(xyz_, dim=0), torch.max(xyz_, dim=0), sigmas)
            print(weights)
            print(rgb_final, cls_final.shape, clspoints.shape)
            print(torch.argmax(clspoints, dim=-1))
            print(cls_final)

        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)
        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, cls_final, weights,

    # Extract models from lists
    model_coarse = models[0]
    points = models[-1] 

    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # near far will effect sampling 

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)
    # print(xyz_coarse_sampled.shape, "coarse", )
    if test_time:
        # print(xyz_coarse_sampled.shape, "coarse", )
        weights_coarse = \
            inference(model_coarse, points, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, cls_coarse, weights_coarse = \
            inference(model_coarse, points, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'cls_coarse': cls_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)
        # print(xyz_fine_sampled.shape, "fine", )
        model_fine = models[1]
        rgb_fine, depth_fine, cls_fine, weights_fine = \
            inference(model_fine, points, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['cls_fine'] = cls_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result


def render_rays_3d_conv(models,
                  embeddings,
                  rays,
                  N_samples=64,
                  use_disp=False,
                  perturb=0,
                  noise_std=1,
                  N_importance=0,
                  chunk=1024*32,
                  white_back=False,
                  test_time=False,
                  _cls_num=11,
                  network='3DUNet',
                ):
    """
    render cls results.
    """
    def inference(model, points, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []        
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk]) # 3 channel encoder to 10
            if not weights_only:
                # training split
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            # with torch.no_grad():
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]
        
        out = torch.cat(out_chunks, 0)
        # print(out.shape, xyz_.shape, N_rays, N_samples)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            # cls + 4 = 23
            rgbsigma = out.reshape(N_rays, N_samples_, -1) # ! need check shape
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)
            """
            NOTE: The class of xyz should be N_rays due to it is image-independent
            """
            if DEBUG:
                print(rgbsigma.shape, "?")

        
        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        #  sum N_samples rgb results
        if DEBUG:
            # import ipdb; ipdb.set_trace()
            print(torch.min(xyz_, dim=0), torch.max(xyz_, dim=0), sigmas)
            print(weights)
            print(rgb_final, cls_final.shape, clspoints.shape)
            print(torch.argmax(clspoints, dim=-1))
            print(cls_final)

        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)
        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        ################### cls #################
        # use weight to sample xyz
        N_sample = weights.shape[1]
        clspoints = torch.zeros((N_rays, N_sample, _cls_num),dtype=torch.float32).cuda() # all is background
        # set thresh avoid oom
        if  test_time:
            # _thresh = 0
            _thresh = 0.00001
        else:
            _thresh = 0.00001

        # sample_weights = weights
        sample_weights = weights
        sample_mask = sample_weights>_thresh
        sample_points = xyz_[sample_mask.reshape(-1)]

        # print(sample_points[100:200], "11111")
        # normalize
        # norm_sp = np.linalg.norm(sample_points.detach().cpu().numpy()) # 1.6 not support torch.linalg.norm
        # sample_points = sample_points / norm_sp
        # print(sample_points[100:200])

        rgbs_points = rgbs.reshape(-1,3)[sample_mask.reshape(-1)]

        if True:
            sample_weights_points = sample_weights.reshape(-1,1)[sample_mask.reshape(-1)]
            # sample_weights = sample_weights.unsqueeze(-1)
            # print(sample_points.shape, rgbs_points.shape, sample_weights_points.shape)
            sample_points = torch.cat([sample_points,  rgbs_points, sample_weights_points], dim=1)
        else:
            sample_points = torch.cat([sample_points,  rgbs_points], dim=1) # pts, 6,
        if sample_points.shape[0] < 3200:
            points_preds = torch.zeros((sample_points.shape[0],_cls_num),dtype=torch.float32).cuda()
            # print('00000000000000000000000')
        elif network == 'pointnet':
            sample_points = sample_points.transpose(1, 0) # 6, pts
            sample_points = torch.unsqueeze(sample_points, 0) # 1, 6, pts
            sample_points = sample_points.contiguous()
            points_preds, _, _ = points(sample_points)
            points_preds = points_preds[0]  # pts = sample_masks
        else:
            # voxelizer = Voxelizer(voxel_size=0.001, ignore_label=0)
            # sample_points_coords = sample_points[:,:3].detach().cpu()
            # sample_points_features = sample_points[:,3:]
            # sample_points_coords, inds, _ = voxelizer.voxelize(sample_points_coords)
            # sample_points_coords = torch.cat((torch.zeros(sample_points_coords.shape[0]).unsqueeze(1),sample_points_coords),1) # (n, (batch,x,y,z)) batch always 0
            # sample_points_features = sample_points_features[inds] # (n, (r,g,b))
            # input = ME.TensorField(
            #     coordinates=sample_points_coords,
            #     features=sample_points_features,
            #     device="cuda",
            # )
            # points_preds = points(input)
            # voxel_size = 0.1 # train use 0.1
            voxel_size = 0.1
            coords = sample_points[:,:3]
            colors = sample_points[:,3:]
            in_field = ME.TensorField(
                features=colors,
                coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),  # (n, 4) dim 0 represents bs_id
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device="cuda",
            )
            # print('in field', in_field.shape, in_field.D)
            # Convert to a sparse tensor
            sinput = in_field.sparse()
            # print('sinput', sinput.shape, sinput.D)
            # Output sparse tensor
            soutput = points(sinput)
            # get the prediction on the input tensor field
            out_field = soutput.slice(in_field)
            points_preds = out_field.F
            # points_preds = F.log_softmax(points_preds,dim=-1)
            # points_preds = F.softmax(points_preds,dim=-1)

        clspoints = clspoints.reshape(-1,6)
        # print('11111',xyz_.dtype,clspoints.dtype,sample_mask.dtype,points_preds.dtype)
        clspoints[sample_mask.reshape(-1)] = points_preds.float()
        clspoints = clspoints.reshape((N_rays, N_sample, _cls_num))
        # clspoints[sample_mask] = points_preds # N rays, N_sample, cls
        # orginal rgb ways, use sum
        cls_final = torch.sum(weights.unsqueeze(-1)*clspoints, -2) # N_rays, cls
        cls_final = F.log_softmax(cls_final, dim=-1)
        ################### cls #################

        return rgb_final, depth_final, cls_final, weights,

    # Extract models from lists
    model_coarse = models[0]
    points = models[-1] 

    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # near far will effect sampling 

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)
    # print(z_vals, near, far)
    # exit()
    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)
    # print(xyz_coarse_sampled.shape, "coarse", )
    if test_time:
        # print(xyz_coarse_sampled.shape, "coarse", )
        weights_coarse = \
            inference(model_coarse, points, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, cls_coarse, weights_coarse = \
            inference(model_coarse, points, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'cls_coarse': cls_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)
        # print(xyz_fine_sampled.shape, "fine", )
        model_fine = models[1]
        rgb_fine, depth_fine, cls_fine, weights_fine = \
            inference(model_fine, points, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['cls_fine'] = cls_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result
