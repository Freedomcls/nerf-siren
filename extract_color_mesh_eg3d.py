import torch
import os
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
import plyfile
import time
import skimage
from argparse import ArgumentParser

from utils import load_ckpt

from datasets import dataset_dict
from eg3d_training.eg3d_renderer import EG3D_Renderer 

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--vis_type', type=str, default='color',
                        choices=['color', 'label'],
                        help='which type to vis')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'replica'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output ply filename')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of samples to infer the acculmulated opacity')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--occ_threshold', type=float, default=0.2,
                        help='''threshold to consider a vertex is occluded.
                                larger=fewer occluded pixels''')

    #### method using vertex normals ####
    parser.add_argument('--use_vertex_normal', action="store_true",
                        help='use vertex normals to compute color')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of fine samples to infer the acculmulated opacity')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='the near bound factor to start the ray')

    return parser.parse_args()

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")


if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
    dataset = dataset_dict[args.dataset_name](**kwargs)

    eg3d_renderer = EG3D_Renderer()
    load_ckpt(eg3d_renderer, args.ckpt_path, model_name='eg3d_renderer')
    eg3d_renderer.cuda().eval()


    # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
    max_batch=1000000
    shape_res = 256
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=3.0 * 1)#.reshape(1, -1, 3)
    samples = samples.cuda()
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1)).cuda()
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3)).cuda()
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = eg3d_renderer.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head])['sigma']
                sigmas[:, head:head+max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    shape_format = '.ply'
    if shape_format == '.ply':
        outdir = './'
        seed = 0
        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
    elif shape_format == '.mrc': # output mrc
        with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas
    exit()


    # # define the dense grid for query
    # N = args.N_grid
    # xmin, xmax = args.x_range
    # ymin, ymax = args.y_range
    # zmin, zmax = args.z_range
    # # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    # x = np.linspace(xmin, xmax, N)
    # y = np.linspace(ymin, ymax, N)
    # z = np.linspace(zmin, zmax, N)

    # xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    # dir_ = torch.zeros_like(xyz_).cuda()
    #        # sigma is independent of direction, so any value here will produce the same result

    # # predict sigma (occupancy) for each grid location
    # print('Predicting occupancy ...')
    # with torch.no_grad():
    #     B = xyz_.shape[0]
    #     out_chunks = []
    #     for i in tqdm(range(0, B, args.chunk)):
    #         with torch.no_grad():
    #             coordinates = xyz_[i:i+args.chunk].unsqueeze(0)
    #             directions = dir_[i:i+args.chunk].unsqueeze(0)
    #             sigma = eg3d_renderer.sample(coordinates, directions)['sigma'].squeeze(0)
    #             out_chunks += [sigma]
    #     rgbsigma = torch.cat(out_chunks, 0)

    # sigma = rgbsigma[:, -1].cpu().numpy()
    # sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # # perform marching cube algorithm to retrieve vertices and triangle mesh
    # print('Extracting mesh ...')
    # vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    # ##### Until mesh extraction here, it is the same as the original repo. ######

    # vertices_ = (vertices/N).astype(np.float32)
    # ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    # x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    # y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    # vertices_[:, 0] = x_
    # vertices_[:, 1] = y_
    # vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    # vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    # face['vertex_indices'] = triangles

    # PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
    #          PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    # exit()
    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"{args.scene_name}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    W, H = args.img_wh
    K = np.array([[dataset.focal, 0, W/2],
                  [0, dataset.focal, H/2],
                  [0,             0,   1]]).astype(np.float32)

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)

    if args.use_vertex_normal: ## use normal vector method as suggested by the author.
                               ## see https://github.com/bmild/nerf/issues/44
        mesh.compute_vertex_normals()
        rays_d = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        near = dataset.bounds.min() * torch.ones_like(rays_d[:, :1])
        far = dataset.bounds.max() * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t

        nerf_coarse = NeRF()
        load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        nerf_coarse.cuda().eval()

        results = f([nerf_coarse, nerf_fine], embeddings,
                    torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                    args.N_samples,
                    args.N_importance,
                    args.chunk,
                    dataset.white_back)

    else: ## use my color average method. see README_mesh.md
        ## buffers to store the final averaged color
        non_occluded_sum = np.zeros((N_vertices, 1))
        v_color_sum = np.zeros((N_vertices, 3))

        # Step 2. project the vertices onto each training image to infer the color
        print('Fusing colors ...')
        for idx in tqdm(range(len(dataset.image_paths))):
            ## read image of this pose
            if args.vis_type == 'label':
                image_path = dataset.image_paths[idx]
                parse_path = image_path.replace('train','labels')
                parse_res = Image.open(parse_path)
                parse_res = np.asarray((parse_res))/10
                parse_res = cv2.resize(parse_res, tuple(args.img_wh),interpolation=cv2.INTER_NEAREST)
                parse_res = parse_res.astype(np.uint8)
                part_colors = [[255, 0, 0], [255, 0, 255], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
                parse_res_color = np.zeros((parse_res.shape[0], parse_res.shape[1], 3))
                num_cls = np.max(parse_res)
                # print(num_cls)
                for i in range(1, num_cls+1):
                    index = np.where(parse_res==i)
                    # print(i, index, vis_pred)
                    parse_res_color[index[0], index[1], :] = part_colors[i]
                image = parse_res_color
            else:
                image = Image.open(dataset.image_paths[idx]).convert('RGB')
                image = image.resize(tuple(args.img_wh), Image.LANCZOS)
                image = np.array(image)

            ## read the camera to world relative pose
            P_c2w = np.concatenate([dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
            ## project vertices from world coordinate to camera coordinate
            vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
            vertices_cam[1:] *= -1 # (3, N) in "right down forward"
            ## project vertices from camera coordinate to pixel coordinate
            vertices_image = (K @ vertices_cam).T # (N, 3)
            depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
            vertices_image = vertices_image[:, :2]/depth
            vertices_image = vertices_image.astype(np.float32)
            vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
            vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

            ## compute the color on these projected pixel coordinates
            ## using bilinear interpolation.
            ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
            ## so we split the input into chunks.
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image, 
                                    vertices_image[i:i+remap_chunk, 0],
                                    vertices_image[i:i+remap_chunk, 1],
                                    interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors) # (N_vertices, 3)
            
            ## predict occlusion of each vertex
            ## we leverage the concept of NeRF by constructing rays coming out from the camera
            ## and hitting each vertex; by computing the accumulated opacity along this path,
            ## we can know if the vertex is occluded or not.
            ## for vertices that appear to be occluded from every input view, we make the
            ## assumption that its color is the same as its neighbors that are facing our side.
            ## (think of a surface with one side facing us: we assume the other side has the same color)

            ## ray's origin is camera origin
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            ## ray's direction is the vector pointing from camera origin to the vertices
            rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            ## the far plane is the depth of the vertices, since what we want is the accumulated
            ## opacity along the path from camera origin to the vertices
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis] # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth # weight by inverse depth
                                                                    # near=more confident in color
            non_occluded += opacity < args.occ_threshold
            
            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    if args.use_vertex_normal:
        v_colors = results['rgb_fine'].cpu().numpy() * 255.0
    else: ## the combined color is the average color among all views
        v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
        
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'), 
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    print('Done!')
