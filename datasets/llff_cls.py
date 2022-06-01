import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .llff import normalize, average_poses, center_poses, create_spiral_poses, create_spheric_poses
import cv2

# DEBUG = os.environ.get("DEBUG", False)
DEBUG = False
def merge_cls():
    # cls_map: parse results
    atts =   ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    # new_atts = ['skin', 'brow', 'brow', 'eye', 'eye', 'eye', 'ear', 'ear', 'ear', 
    #         'nose', 'mouth', 'lip', 'lip',  'neck',  'cloth', 'cloth', 'head', 'head']
    new_atts = ['skin', 'face', 'face', 'face', 'face', 'face', 'head', 'head', 'head', 
            'face', 'face', 'face', 'face',  'neck',  'cloth', 'cloth', 'head', 'head']
    
    # face 表示五官， nose lip eys .etc
    new_map = {
        'skin':1, 
        'face':2, 
        'neck':3, 
        'head':4,
        'cloth': 5,
    }
    
    # 19-> 10
    # new_map = {
    #     'skin':1, 
    #     'brow':2, 
    #     'eye':3, 
    #     'ear':4,   
    #     'nose':5,
    #     'mouth':6, 
    #     'lip':7, 
    #     'neck':8,  
    #     'head':9, 
    #     'cloth': 10,
    # }
    ids_map = {}
    for i, (att, new_att) in enumerate(zip(atts, new_atts), 1):
        ids_map[i] = new_map[new_att]
    return ids_map

def convert_pred(pred, scale=10):
    pred = np.array(pred, dtype=np.float)
    # print(pred.shape, pred)
    # print(pred[pred==255])
    ids_map = merge_cls()
    for ids in ids_map:
        pred[pred==int(ids)*scale] = int(ids_map[ids])
        # print(int(ids_map[ids]))
    # print(pred[pred==255])
    return pred


class LLFFClsDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(1920, 1080), spheric_poses=False, val_num=1, batch_size=2):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.edited_ids = [25, 50, 75, 100, 125, 175, 200, 225, 250, 275, \
            300, 325, 350, 375, 400, 1250, 1275, 1300, 1325, 1350, 1375, 1400, \
            1425, 1450]
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.define_transforms()

        self.read_meta()
        self.white_back = False
        self.batch_size = batch_size

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'edit_imgs/*')))
                        # load full resolution image then resize

        self.raw_parse_path = sorted(glob.glob(os.path.join(self.root_dir, 'raw_parse/*.png'))) 
        # self.parse_path = sorted(glob.glob(os.path.join(self.root_dir, 'edit_parse/*.png'))) 
        # * use raw parse results, need to replace with 


        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses) # pose的点是normalize过的
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
        
        # post-process pose
        self._poses = []
        for i, image_path in enumerate(self.image_paths):
            ids = int(image_path.split("/")[-1].split(".")[0].split("_")[-1])
            if ids not in self.edited_ids:
                continue 
            self._poses.append(np.expand_dims(self.poses[i], 0))
        self._poses = np.concatenate(self._poses, axis=0)
        
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_parse = [] # face-parsing 

            # for i, (image_path, parse_path) in enumerate(zip(self.image_paths, self.parse_path)):
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx: # exclude the val image
                    continue
                ids = int(image_path.split("/")[-1].split(".")[0].split("_")[-1])
                dir_name = image_path.split("/")[-1].split(".")[0]
                # print(ids, dir_name)
                if ids not in self.edited_ids:
                    continue 
                parse_path = os.path.join(self.root_dir, f'edit_parse/{dir_name}.png')
                # parse_path = os.path.join(self.root_dir, f'edit_parse/{dir_name}.jpg) # use rgb HxWx3

                assert os.path.exists(parse_path)
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                parse_res = Image.open(parse_path)

                # parse_res = cv2.imread(parse_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                # parse_res = parse_res.T #cv2 load inverse h w
                # print(img.size, parse_res.shape)
                
                assert list(parse_res.size[:2]) == list(img.size[:2]),\
                    f"{parse_res.size}!={img.size}"
                
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''

                img = img.resize(self.img_wh, Image.LANCZOS) 
                # parse_res = parse_res.resize(self.img_wh, Image.LANCZOS)
                """
                Check this to know LANZOS sampling:
                https://gis.stackexchange.com/questions/10931/what-is-lanczos-resampling-useful-for-in-a-spatial-context
                """
                parse_res = convert_pred(np.asarray(parse_res))
                #print(np.max(parse_res))
                parse_res = cv2.resize(parse_res, (self.img_wh[1], self.img_wh[0]),interpolation=cv2.INTER_NEAREST) 
                parse_res = Image.fromarray(parse_res)
                # parse_res = parse_res.resize(self.img_wh, Image.LANCZOS) 

                # parse_res = cv2.resize(parse_res, (self.img_wh[1], self.img_wh[0]))
                # print(parse_res.shape)


                img = self.transform(img) # (3, h, w)
                parse_res = self.transform(parse_res)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                parse_res = parse_res.reshape(-1, 1).contiguous() # (h*w, 1) 
                #print(torch.max(parse_res))
                # parse_res = parse_res.view(3, -1).permute(1, 0) # (h*w, 3) RGB 
                # print("train", parse_res.shape, parse_res[parse_res!=0])
                # print(parse_res.numpy().shape)
                # cv2.imwrite(f"./debug/parse_{i}.jpg", parse_res.numpy().reshape(self.img_wh[1], self.img_wh[0]))
                # exit()


                self.all_rgbs += [img]
                self.all_parse  += [parse_res]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3) 
            self.all_parse = torch.cat(self.all_parse, 0) 
            
        
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]
            # use raw results as val bcz may not exists
            self.parse_path_val = self.raw_parse_path[val_idx]

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5   # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0) # 卡所有的pose的百分位数
                self.poses_test = create_spiral_poses(radii, focus_depth, n_poses=60) # Nx3x4
                
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        # idx means pixel, batch idx 
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'parse': self.all_parse[idx],
                      }
            # print(self.all_parse[idx], "parse train", self.all_parse.shape, self.all_parse[idx].shape)
            # print(self.all_rgbs[idx], "parse rgb", self.all_rgbs.shape, self.all_rgbs[idx].shape)


        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
                                            #   self.focal, 2.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

                parse = cv2.imread(self.parse_path_val, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                parse = parse.T
                parse = cv2.resize(parse, (self.img_wh[1], self.img_wh[0]), interpolation=cv2.INTER_LANCZOS4)
                parse = self.transform(parse) # (1, h, w)

                parse = parse.view(1, -1).permute(1, 0) # (h*w, 1)
                # parse = parse.view(3, -1).permute(1, 0) # (h*w, 3)
                if DEBUG:
                    print(parse.shape, "val")
                
                sample["parse"] = parse

        return sample


class LLFFClsDatasetImgBatch(LLFFClsDataset):
    def __len__(self):
        if self.split == 'train':
            return len(self.edited_ids) - 1 # (1 for val)
        if self.split == 'val':
            return self.val_num

        return len(self.poses_test)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            h, w = self.img_wh[1], self.img_wh[0]
            
            sample = {'rays': self.all_rays[idx*h*w:(idx+1)*h*w],
                      'rgbs': self.all_rgbs[idx*h*w:(idx+1)*h*w],
                      'parse': self.all_parse[idx*h*w:(idx+1)*h*w],
                      }  
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            
            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

                parse = cv2.imread(self.parse_path_val, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                parse = parse.T
                parse = cv2.resize(parse, (self.img_wh[1], self.img_wh[0]), interpolation=cv2.INTER_LANCZOS4)
                parse = self.transform(parse) # (1, h, w)

                parse = parse.view(1, -1).permute(1, 0) # (h*w, 1)
                # parse = parse.view(3, -1).permute(1, 0) # (h*w, 3)
                if DEBUG:
                    print(parse.shape, "val")
                
                sample["parse"] = parse

        return sample
