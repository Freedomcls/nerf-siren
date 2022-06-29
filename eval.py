import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays, render_rays_3d, render_rays_3d_conv
from models.nerf import *
from models.nerf_cls import NeRF_3D
from models.pointnets import PointNetDenseCls
from models.ConvNetWork import *

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
    parser.add_argument('--mode', default="normal",
                        type=str, choices=['d3', 'd3_ib', 'normal'],
                        help='use which system')
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'blender_cls_ib' ,'llff', "llff_cls", "llff_cls_ib"],
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


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      render_func,
                      **kwargs,
                      ):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32 # hard code 
    results = defaultdict(list)
    for i in range(0, B, chunk):
        # print(rays[i:i+chunk].shape, B)
        rendered_ray_chunks = \
            render_func(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
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

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    if args.semantic_network == "pointnet":
        points = PointNetDenseCls(k=_cls, inc=6)
    elif args.semantic_network == "conv3d":
        points = MinkUNet14A(in_channels=4, out_channels=_cls)
        points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(points)


    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    load_ckpt(points, args.ckpt_path, model_name='points')
    
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()
    points.cuda().eval()

    models = [nerf_coarse, nerf_fine, points]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)
    if args.mode == "normal":
        render_func = render_rays
    elif "d3" in args.mode:
        if args.semantic_network == "pointnet":
            render_func = render_rays_3d
        elif args.semantic_network == "conv3d":
            render_func = render_rays_3d_conv

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    render_func=render_func,
                                    _cls_num=_cls,
                                    network=args.semantic_network,
                                    )
        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        if "d3" in args.mode:
            cls_num = _cls
            raw_cls_pred = results["cls_fine"].view(h, w, cls_num).cpu().numpy()
            cls_pred = np.argmax(raw_cls_pred, axis=-1)
            cv2.imwrite(os.path.join(dir_name, 'r_%d.png'%i), cls_pred * 10)
            # imageio.imwrite(os.path.join(dir_name, f'{i:03d}_cls.png'), cls_pred * 255)
            if DEBUG:
                print(cls_pred[cls_pred!=0])

            color_cls((img_pred*255).astype(np.uint8), cls_pred, \
                f"./results/{args.dataset_name}/{args.scene_name}_cls_map", prefix=str(i))
                
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
