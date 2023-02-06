import os, sys
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import cv2
import imageio
from imgviz import label_colormap
import torch
import math
import torchvision.transforms as transforms
from datasets import augmentations
from PIL import Image
# from torchvision.models import Vgg19_out
from utils.vgg_precep_loss import Vgg16GenFeats

feats_lvl = [0]  # lvl-1-2-3 will downsample shape may mismatch
MODEL = Vgg16GenFeats()

def create_rays(num_rays, Ts_c2w, height, width, fx, fy, cx, cy, near, far, c2w_staticcam=None, depth_type="z",
            use_viewdirs=True, convention="opencv"):
    """
    convention: 
    "opencv" or "opengl". It defines the coordinates convention of rays from cameras.
    OpenCv defines x,y,z as right, down, forward while OpenGl defines x,y,z as right, up, backward (camera looking towards forward direction still, -z!)
    Note: Use either convention is fine, but the corresponding pose should follow the same convention.

    """
    print('prepare rays')

    rays_cam = get_rays_camera(num_rays, height, width, fx, fy, cx, cy, depth_type=depth_type, convention=convention) # [N, H, W, 3]

    dirs_C = rays_cam.view(num_rays, -1, 3)  # [N, HW, 3]
    rays_o, rays_d = get_rays_world(Ts_c2w, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # c2w_staticcam: If not None, use this transformation matrix for camera,
            # while using other c2w argument for viewing directions.
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_world(c2w_staticcam, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    return rays

def get_rays_camera(B, H, W, fx, fy,  cx, cy, depth_type, convention="opencv"):

    assert depth_type == "z" or depth_type == "euclidean"
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))  # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode

    i = i.t().float()
    j = j.t().float()

    size = [B, H, W]

    i_batch = torch.empty(size)
    j_batch = torch.empty(size)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    if convention == "opencv":
        x = (i_batch - cx) / fx
        y = (j_batch - cy) / fy
        z = torch.ones(size)
    elif convention == "opengl":
        x = (i_batch - cx) / fx
        y = -(j_batch - cy) / fy
        z = -torch.ones(size)
    else:
        assert False

    dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3]

    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3, keepdim=True)
        dirs = dirs * (1. / norm)

    return dirs


def get_rays_world(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
    origins = T_WC[:, :3, -1]  # Bx3
    origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]
    return origins, dirs_W



def gen_image_feats(img, feats_layers, norm=True, resize=False):
    """ use vgg16 """
    img = torch.from_numpy(img).float()  # HWC
    img = img.permute(2, 0, 1).unsqueeze(0)  # 1CHW
    img_feats, _ = MODEL(img, feats_layers, norm=norm, resize=resize)
    
    return img_feats


class ReplicaDatasetCache(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800), is_crop=False):

        data_dir = root_dir
        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
        self.semantic_class_dir = os.path.join(data_dir, "semantic_class")
        self.semantic_instance_dir = os.path.join(data_dir, "semantic_instance")
        if not os.path.exists(self.semantic_instance_dir):
            self.semantic_instance_dir = None

        total_num = 900
        step = 5
        train_ids = list(range(0, total_num, step))
        test_ids = [x+step//2 for x in train_ids]  

        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.test_ids = test_ids
        self.test_num = len(test_ids)

        self.img_w, self.img_h = img_wh
        self.set_params_replica()
        self.use_viewdir = False
        self.convention = "opencv"
        self.white_back = False
        self.split = split

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        if self.semantic_instance_dir is not None:
            self.instance_list = sorted(glob.glob(self.semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.train_samples = {'image': [], 'depth': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

        self.test_samples = {'image': [], 'depth': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

        

        if split == 'train':
        # training samples
            self.poses = []

            self.feats_gt = [[] for _ in range(len(feats_lvl))]  # add precep loss 

            self.image_paths = []
            for idx in train_ids:
                image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
                depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
                if self.semantic_instance_dir is not None:
                    instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

                if (self.img_h is not None and self.img_h != image.shape[0]) or \
                        (self.img_w is not None and self.img_w != image.shape[1]):
                    image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    # NOTE: here use self.img_w, self.img_h, as vgg train image resolution 224x224
                    # print(image.shape)
                    image_feats = gen_image_feats(image, feats_layers=feats_lvl)  
                    _ = [self.feats_gt[i].append(image_feats[i]) for i in range(len(image_feats))]
                    

                    depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                    if self.semantic_instance_dir is not None:
                        instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

                T_wc = self.Ts_full[idx]

                self.train_samples["image"].append(image)
                self.train_samples["depth"].append(depth)
                self.train_samples["semantic"].append(semantic)
                if self.semantic_instance_dir is not None:
                    self.train_samples["instance"].append(instance)
                self.train_samples["T_wc"].append(T_wc)
                self.poses.append(T_wc[:3,:4])
                if idx % 9 == 0:
                    self.image_paths.append(self.rgb_list[idx])
            for key in self.train_samples.keys():  # transform list of np array to array with batch dimension
                self.train_samples[key] = np.asarray(self.train_samples[key])
            self.read_meta_train()

            print()
            print("Training Sample Summary:")
            for key in self.train_samples.keys(): 
                print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        else:
            # test samples
            self.feats_gt = [[] for _ in range(len(feats_lvl))]  # add precep loss 
            for idx in test_ids:
                image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
                depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
                if self.semantic_instance_dir is not None:
                    instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

                if (self.img_h is not None and self.img_h != image.shape[0]) or \
                        (self.img_w is not None and self.img_w != image.shape[1]):
                    image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    image_feats = gen_image_feats(image, feats_layers=feats_lvl)  
                    _ = [self.feats_gt[i].append(image_feats[i]) for i in range(len(image_feats))]

                    depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                    if self.semantic_instance_dir is not None:
                        instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                T_wc = self.Ts_full[idx]

                self.test_samples["image"].append(image)
                self.test_samples["depth"].append(depth)
                self.test_samples["semantic"].append(semantic)
                if self.semantic_instance_dir is not None:
                    self.test_samples["instance"].append(instance)
                self.test_samples["T_wc"].append(T_wc)

            for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
                self.test_samples[key] = np.asarray(self.test_samples[key])
            self.read_meta_test()

            print()
            print("Testing Sample Summary:")
            for key in self.test_samples.keys(): 
                print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))

            # self.semantic_classes = np.unique(
            #     np.concatenate(
            #         (np.unique(self.train_samples["semantic"]), 
            #     np.unique(self.test_samples["semantic"])))).astype(np.uint8)
            # self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including the void class of 0

            # self.colour_map_np = label_colormap()[self.semantic_classes]
            # self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
            # # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss is applied on this frame

            # # remap existing semantic class labels to continuous label ranging from 0 to num_class-1
            # self.train_samples["semantic_clean"] = self.train_samples["semantic"].copy()
            # self.train_samples["semantic_remap"] = self.train_samples["semantic"].copy()
            # self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_clean"].copy()

            # self.test_samples["semantic_remap"] = self.test_samples["semantic"].copy()

            # for i in range(self.num_semantic_class):
            #     self.train_samples["semantic_remap"][self.train_samples["semantic"]== self.semantic_classes[i]] = i
            #     self.train_samples["semantic_remap_clean"][self.train_samples["semantic_clean"]== self.semantic_classes[i]] = i
            #     self.test_samples["semantic_remap"][self.test_samples["semantic"]== self.semantic_classes[i]] = i
        self.source_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            augmentations.ToOneHot(99),
            transforms.ToTensor()])

        self.image = self.load_data()
        # self.img = self.image.view(1, *self.image.size()).float()
        # self.img = [x.view(1, *x.size()).float() for x in self.image]

    def load_data(self):
        from_ims = []
        for filename in os.listdir('./semantic-masks'):
            # from_path_1 = self.source_paths[index]
            # print(index)
            from_im = Image.open('./semantic-masks/' + filename)
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


    def set_params_replica(self):
        self.H = self.img_h
        self.W = self.img_w

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W/self.H

        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.focal = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = 0.1, 10.0
        self.bounds = np.array([self.near, self.far])
        self.c2w_staticcam = None

    def read_meta_train(self):
        self.train_Ts = torch.from_numpy(self.train_samples["T_wc"]).float()
        self.train_image = torch.from_numpy(self.train_samples["image"]).float().contiguous()
        self.train_semantic = torch.from_numpy(self.train_samples["semantic"]).float().contiguous()
        self.all_rgbs = self.train_image.reshape(-1, self.train_image.shape[-1]) # [num_train*H*W, 8]
        # self.all_semantic = self.train_semantic.reshape(-1,self.train_semantic.shape[-1])
        self.all_rays = create_rays(self.train_num, self.train_Ts, self.H, self.W, self.fx, self.fy, self.cx, self.cy,
                        self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)
        num_img, num_ray, ray_dim = self.all_rays.shape
        self.all_rays = self.all_rays.reshape(num_img*num_ray,ray_dim)
        self.train_feats = []
        for i, feats in enumerate(self.feats_gt):
            # each feat: 1xCxHxW -> NxCxHxW(vstack) ->  NxHxWxC(permute) -> N*H*WxC(reshape)
            assert i == 0, "current we only support 0-lvl feats unsample" 
            feats = torch.vstack(feats).permute(0, 2, 3, 1)
            feats = feats.reshape(self.all_rgbs.shape[0], feats.shape[-1]) 
            print(i, " feats: ",  feats.shape)
            self.train_feats.append(feats)
        

    def read_meta_test(self):
        self.test_Ts = torch.from_numpy(self.test_samples["T_wc"]).float()  # [num_test, 4, 4]
        self.test_image = torch.from_numpy(self.test_samples["image"]).float().contiguous()  # [num_test, H, W, 3]
        self.all_rays = create_rays(self.test_num, self.test_Ts, self.H, self.W, self.fx, self.fy,
                                self.cx, self.cy, self.near, self.far, use_viewdirs=self.use_viewdir, convention=self.convention)
        num_img, num_ray, ray_dim = self.all_rays.shape
        self.all_rgbs = self.test_image.reshape(num_img, num_ray, self.test_image.shape[-1]) # [num_test, H*W, 3]
        self.val_feats = []
        for i, feats in enumerate(self.feats_gt):
            # each feat: 1xCxHxW -> NxCxHxW(vstack) ->  NxHxWxC(permute) -> N*H*WxC(reshape)
            assert i == 0, "current we only support 0-lvl feats unsample" 
            feats = torch.vstack(feats).permute(0, 2, 3, 1)
            feats = feats.reshape(self.all_rgbs.shape[0], self.all_rgbs.shape[1], feats.shape[-1]) 
            print(i, " feats: ",  feats.shape)
            self.val_feats.append(feats)

    def __len__(self):
        return self.all_rays.shape[0]
            
    def __getitem__(self, idx):
        # i=0
        if self.split == 'train':
            # sample = {'rays':self.all_rays[idx], 'rgbs':self.all_rgbs[idx], 'img':self.image}
            # print("img",self.image[0].shape)
            # for x in self.image:
            #     i=i+1
            # print("how many img",i)
            sample = {'rays':self.all_rays[idx], 'rgbs':self.all_rgbs[idx], "feats": self.train_feats[0][idx]}

        else:
            rays = self.all_rays[idx]
            rgbs = self.all_rgbs[idx]
            
            # sample = {'rays':rays, 'rgbs':rgbs,'img':self.image}
            sample = {'rays':rays, 'rgbs':rgbs, "idx": idx, "feats": self.val_feats[0][idx]}
        return sample


