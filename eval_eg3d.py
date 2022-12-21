import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from eg3d_training.eg3d_renderer import EG3D_Renderer 

from utils import load_ckpt, color_cls
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
import cv2
import ast
torch.backends.cudnn.benchmark = True

DEBUG = ast.literal_eval(os.environ.get("DEBUG", "False"))

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'blender_cls_ib' ,'llff', "llff_cls", "llff_cls_ib", "replica"],
                        help='which dataset to validate')
    parser.add_argument('-sn', '--semantic_network', type=str, default='pointnet',
                        choices=['pointnet', 'conv3d'], 
                        help='use which network to extract semantic features')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


def batch_inference(eg3d_renderer, rays):
    B = rays.shape[0]
    chunk = 1024 * 4
    results = defaultdict(list)
    for i in range(0, B, chunk):
        conditioning_params = -1
        batch_results = eg3d_renderer.render(conditioning_params, rays[i:i+chunk,:3], rays[i:i+chunk,3:6])
        for k, v in batch_results.items():
            results[k] += [v]

    for k,v in results.items():
        results[k] = torch.cat(v, 0)
    return results
        

if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh
    _cls = 6 # hard code

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if 'llff' in args.dataset_name:
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    eg3d_renderer = EG3D_Renderer()

    load_ckpt(eg3d_renderer, args.ckpt_path, model_name='eg3d_renderer')
    
    eg3d_renderer.cuda().eval()

    models = [eg3d_renderer]

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        print(rays.shape)
        conditioning_params = -1
        with torch.no_grad():
            results = batch_inference(eg3d_renderer, rays)
            # results = eg3d_renderer.render(conditioning_params, rays[:,:3], rays[:,3:6])
        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
                
        if args.save_depth:
            depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(f'depth_{i:03d}', 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
