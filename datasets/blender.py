import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
from .ray_utils import *
import random

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),is_crop=False):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True
        self.is_crop = is_crop

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)

        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_poses = []
            for frame in self.meta['frames']:
                matrix = self.meta['frames'][0]['transform_matrix']
                pose = np.array(matrix)[:3,:4]
                # pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                c2w = c2w.reshape(-1)
                self.all_poses += [c2w.repeat(rays_d.shape[0],1)]
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_poses = torch.cat(self.all_poses, 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'c2w':self.all_poses[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            c2w = c2w.reshape(-1)
            c2w = c2w.repeat(rays_d.shape[0],1)
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample

class BlenderDatasetWithClsBatch(BlenderDataset):
    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths)*10
        if self.split == 'val':
            return 4 # only validate 8 images (to support <=8 gpus)
            # return len(self.image_paths)
        return len(self.meta['frames'])

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)

        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_parse = []

            for frame in self.meta['frames']:
                matrix = self.meta['frames'][0]['transform_matrix']
                pose = np.array(matrix)[:3,:4]
                # pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")

                ### parse
                parse_path = image_path.replace('train','labels')
                parse_res = Image.open(parse_path)
                parse_res = np.asarray((parse_res))/10
                # print(np.max(parse_res))
                parse_res = cv2.resize(parse_res, (self.img_wh[1], self.img_wh[0]),interpolation=cv2.INTER_NEAREST)
                parse_res = Image.fromarray(parse_res)
                # parse_res = parse_res.resize(self.img_wh, Image.LANCZOS)
                parse_res = self.transform(parse_res)
                parse_res = parse_res.reshape(-1, 1).contiguous()
                # print(torch.max(parse_res))
                self.all_parse  += [parse_res]
                ####


                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_parse = torch.cat(self.all_parse, 0)

    def __getitem__(self, idx):
        idx = idx % 10
        if self.split == 'train': # use data in the buffers
            if self.is_crop:
                crop_size = (32,32)
                h, w = self.img_wh[1], self.img_wh[0]
                h_crop_begin = random.randint(0, h - crop_size[0] - 1)
                w_crop_begin = random.randint(0, w - crop_size[1] - 1)
                rays = self.all_rays.reshape((len(self.meta['frames']), h, w, 8))
                rgbs = self.all_rgbs.reshape((len(self.meta['frames']),h, w, 3))
                parse = self.all_parse.reshape((len(self.meta['frames']), h, w, 1))
                rays = rays[idx, h_crop_begin:h_crop_begin+crop_size[0],w_crop_begin:w_crop_begin+crop_size[1],:]
                rgbs = rgbs[idx, h_crop_begin:h_crop_begin+crop_size[0],w_crop_begin:w_crop_begin+crop_size[1],:]
                parse = parse[idx, h_crop_begin:h_crop_begin+crop_size[0],w_crop_begin:w_crop_begin+crop_size[1],:]
                rays = rays.reshape(-1,8)
                rgbs = rgbs.reshape(-1,3)
                parse = parse.reshape(-1,1)
                # print(rays.shape, rgbs.shape, parse.shape)
                sample = {'rays':rays, 'rgbs':rgbs, 'parse':parse}
            else:
                h, w = self.img_wh[1], self.img_wh[0]
                frame = self.meta['frames'][idx]
                c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
                sample = {'rays': self.all_rays[idx*h*w:(idx+1)*h*w],
                          'rgbs': self.all_rgbs[idx*h*w:(idx+1)*h*w],
                          'parse': self.all_parse[idx*h*w:(idx+1)*h*w],
                          'c2w': c2w,
                          }
    
        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

            ### parse
            parse_path = image_path.replace('val','labels')
            parse_res = Image.open(parse_path)
            parse_res = np.asarray((parse_res))
            parse_res = cv2.resize(parse_res, (self.img_wh[1], self.img_wh[0]),interpolation=cv2.INTER_NEAREST)
            parse_res = Image.fromarray(parse_res)
            # parse_res = parse_res.resize(self.img_wh, Image.LANCZOS)
            parse_res = self.transform(parse_res)
            parse_res = parse_res.reshape(-1, 1).contiguous()
            ####
            sample['parse'] = parse_res

        return sample