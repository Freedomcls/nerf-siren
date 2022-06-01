
from pytorch_lightning import LightningModule
from models.nerf import Embedding, NeRF
from models.nerf_cls import NeRF_3D
from models.pointnets import PointNetDenseCls
from models.rendering import render_rays, render_rays_3d, render_rays_3d_conv
from models.ConvNetWork import *

# optimizer, scheduler, visualization
from utils import *
from losses import loss_dict
from metrics import *
from torch.utils.data import DataLoader
from datasets import dataset_dict
from collections import defaultdict

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]


        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]
        if hparams.pretrained:
            load_ckpt(self.nerf_coarse, hparams.pretrained, model_name='nerf_coarse')
            load_ckpt(self.nerf_fine, hparams.pretrained, model_name='nerf_fine')
            print('Model load finished')

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0] # B is equal to H*W
        results = defaultdict(list)
        # set chunk as large as possible
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if 'llff' in self.hparams.dataset_name:
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,  # slpit all rays (Num_img * H * W)
                          pin_memory=True) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays) # all pics rays concat
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }



class NeRF3DSystem(NeRFSystem):
    def __init__(self, hparams):
        super(NeRF3DSystem, self).__init__(hparams)
        # self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        # self.embedding_dir = Embedding(3, 4) # 4 is the default number
        # self.embeddings = [self.embedding_xyz, self.embedding_dir]
        _cls = 6
        self._cls = _cls
        self._vis = True
        if self.hparams.semantic_network == 'pointnet':
            self.points = PointNetDenseCls(k=_cls, inc=6) # add rgb
            self.render_fun = render_rays_3d
        elif self.hparams.semantic_network == 'conv3d':
            self.points = MinkUNet14A(in_channels=3, out_channels=_cls)
            self.points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.points)
            # print(self.points)
            self.render_fun = render_rays_3d_conv
        else:
            raise NotImplementedError(self.hparams.semantic_network)

        self.models += [self.points]

        #if hparams.pretrained:
        #    load_ckpt(self.nerf_coarse, hparams.pretrained, model_name='nerf_coarse')
        #    load_ckpt(self.nerf_fine, hparams.pretrained, model_name='nerf_fine')
        

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        parse = batch["parse"] 
        return rays, rgbs, parse

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs, parse = self.decode_batch(batch)
        
        results = self(rays)
        # print(rays.shape, rgbs.shape, parse.shape)
        loss = self.loss(results, rgbs, parse)
        log['train/total_loss'] = loss["sum"]
        log['train/rgb_loss'] = loss["rgb"]
        log['train/cls_loss'] = loss["cls"]

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            N, B, c = rgbs.shape
            pred_rgb = results[f'rgb_{typ}'].reshape(N, B, c)
            psnr_ = psnr(pred_rgb, rgbs)
            log['train/psnr'] = psnr_

        # save steps results
        if self._vis:
            clss = results[f"cls_{typ}"].reshape(N, B, self._cls)
            for i in range(N):
                each_rgb = pred_rgb[i].reshape(self.hparams.img_wh[1], self.hparams.img_wh[0], -1).detach().cpu().numpy()
                each_gt_rgb = rgbs[i].reshape(self.hparams.img_wh[1], self.hparams.img_wh[0], -1).detach().cpu().numpy()
                
                each_cls = clss[i].reshape(self.hparams.img_wh[1], self.hparams.img_wh[0], self._cls).detach().cpu().numpy()
                each_cls = np.argmax(each_cls, axis=-1)
                
                each_gt_cls = parse[i].reshape(self.hparams.img_wh[1], self.hparams.img_wh[0]).detach().cpu().numpy()
                color_cls(each_rgb * 255., each_cls, savedir=f"./mid_results/{self.hparams.exp_name}", prefix=f"e{self.current_epoch}_b{i}_pred")
                color_cls(each_gt_rgb * 255., each_gt_cls, savedir=f"./mid_results/{self.hparams.exp_name}", prefix=f"e{self.current_epoch}_b{i}_gt")            

        return {'loss': loss["sum"],
                'progress_bar': {'train_psnr': psnr_ }, 
                'log': log
               }
               
    def forward(self, rays):
        """Render cls additionaly.
        In train B = batchsize (if chunk > batchsize, it is useless)
        In val B = w * h
        """
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                self.render_fun(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            network=self.hparams.semantic_network,
                            _cls_num=self._cls,
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def validation_step(self, batch, batch_nb):
        pass
        # rays, rgbs, parse = self.decode_batch(batch)
        # rays = rays.squeeze() # (H*W, 3)
        # rgbs = rgbs.squeeze() # (H*W, 3)
        # parse = parse.squeeze() # (H*W, CLS)
        # results = self(rays)
        # loss = self.loss(results, rgbs, parse)
        # log = {'val_loss': loss["sum"]}
        # typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        # if batch_nb == 0:
        #     W, H = self.hparams.img_wh
        #     img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        #     img = img.permute(2, 0, 1) # (3, H, W)
        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        #     stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        #     self.logger.experiment.add_images('val/GT_pred_depth',
        #                                        stack, self.global_step)
        # B, c = rgbs.shape
        # pred_rgb = results[f'rgb_{typ}'].reshape(B, c)
        # psnr_ = psnr(pred_rgb, rgbs)
        # log['val_psnr'] = psnr_
        # log['val_cls_loss'] = loss["cls"]
        # log['val_rgb_loss'] = loss["rgb"]
        # return log
        return {"val_loss": 0}


    def validation_epoch_end(self, outputs):
        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': 0,
                                'val_psnr': 0},
                'log': {'val/loss': 0,
                        'val/psnr': 0}
            }


class NeRF3DSystem_ib(NeRF3DSystem):
    def __init__(self, hparams):
        super(NeRF3DSystem_ib, self).__init__(hparams)
        if hparams.pretrained:
            load_ckpt(self.nerf_coarse, hparams.pretrained, model_name='nerf_coarse')
            load_ckpt(self.nerf_fine, hparams.pretrained, model_name='nerf_fine')
            print('Model load finished')

               
    def forward(self, rays):
        """Render cls additionaly."""
        rays_flat = rays.reshape(-1, 8) # batch_size * imgh * imgh 
        B = rays_flat.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                self.render_fun(self.models,
                            self.embeddings,
                            rays_flat[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            network=self.hparams.semantic_network,
                            _cls_num=self._cls,
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)        
        return results
