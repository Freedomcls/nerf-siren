import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
import ast

DEBUG = ast.literal_eval(os.environ.get("DEBUG", "False"))


from models.lpips.networks import get_network, LinLayers
from models.lpips.utils import get_state_dict
from models.lpips.perceptual import Perceptual_loss134, VGGLoss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        # pretrained network
        self.net = get_network()  # .to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)  # .to("cuda")
        self.lin.load_state_dict(get_state_dict())
        # self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, inputs, targets):
        feat_x, feat_y = self.net(inputs['rgb_coarse']), self.net(targets)
        # print("feature_shape",feat_x[0].shape)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        loss = torch.sum(torch.cat(res, 0))
        
        if 'rgb_fine' in inputs:
            # inputs['rgb_fine'] = inputs['rgb_fine'].transpose(1,0)
            # targets = targets.transpose(1,0)
            feat_x, feat_y = self.net(inputs['rgb_fine']), self.net(targets)

            diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
            res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

            loss += torch.sum(torch.cat(res, 0))                
        
        return loss

class PercepMseLoss(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    # def __init__(self, net_type: str = 'alex', version: str = '0.1'):

    #     assert version in ['0.1'], 'v0.1 is only supported now'

    #     super(PerceptualLoss, self).__init__()
    def __init__(self):
        super(PercepMseLoss, self).__init__()

        # pretrained network
        self.net = get_network()  # .to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)  # .to("cuda")
        self.lin.load_state_dict(get_state_dict())
        # self.lin.load_state_dict(get_state_dict(net_type, version))
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        # perceptual loss
        print("mse_inputs_target", inputs['rgb_coarse'].shape, targets.shape)
        feat_x, feat_y = self.net(inputs['rgb_coarse']), self.net(targets)
        print("feature_shape",feat_x[0].shape)
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        per_loss = torch.sum(torch.cat(res, 0))
        # per_loss = torch.sum(torch.cat(res, 0)) / inputs['rgb_coarse'].shape[0]
        
        if 'rgb_fine' in inputs:
            # inputs['rgb_fine'] = inputs['rgb_fine'].transpose(1,0)
            # targets = targets.transpose(1,0)
            feat_x, feat_y = self.net(inputs['rgb_fine']), self.net(targets)

            diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
            res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

            per_loss += torch.sum(torch.cat(res, 0))

        # print("per_loss",per_loss)
        
        # mse loss
        mse_loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            mse_loss += self.loss(inputs['rgb_fine'], targets)        
        # print("mse_loss",mse_loss)

        loss = 0.4*per_loss + mse_loss

        return loss

class perceptual(nn.Module):
    def __init__(self):
        super(perceptual, self).__init__()
        
        # self.vgg_fea= Vgg19_out()
        # self.img1_vggFea = vgg_fea(img1_torch)
    
        self.total_perceptual_loss = VGGLoss()
        self.perceptual_loss134 = Perceptual_loss134()

    def forward(self, inputs, targets):
        if targets.shape[0] == 76800:
            inputs_1 = inputs['rgb_coarse'].view(1,3,320,240)
            tar = targets.view(1,3,320,240)
            if 'rgb_fine' in inputs:
                inputs_2 = inputs['rgb_fine'].view(1,3,320,240)                
        elif targets.shape[0] == 1024:
            inputs_1 = inputs['rgb_coarse'].view(1,3,32,32)
            tar = targets.view(1,3,32,32)
            if 'rgb_fine' in inputs:
                inputs_2 = inputs['rgb_fine'].view(1,3,32,32)
        else:
            print("shape error")        

        print("inputs_1", inputs_1.shape)

        loss = self.total_perceptual_loss(inputs_1, tar)
        # loss = self.perceptual_loss134(inputs_1, tar)

        if 'rgb_fine' in inputs:
            loss += self.total_perceptual_loss(inputs_2, tar)
            # loss = self.perceptual_loss134(inputs_2, tar)           
        
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):#target shape (1024, 3) (76800,3)
        # print("mse_inputs_target", inputs['rgb_coarse'].shape, targets.shape)
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)
        return loss



class VGG16PercepLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        render_feats = inputs["feats_coarse"]
        loss = torch.nn.functional.l1_loss(render_feats, targets)
        if "feats_fine" in inputs:
            loss += torch.nn.functional.l1_loss(inputs["feats_fine"], targets)
        return loss

class MSEVGG16PercepLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
        self.vgg = VGG16PercepLoss()
    
    def forward(self, inputs, rgb, feats):
        loss = {}
        loss["mse"] = self.mse(inputs, rgb)
        loss["precp"] = self.vgg(inputs, feats)
        loss["sum"] = loss["mse"] + loss["precp"]
        return loss
        

class MSECELoss(nn.Module):
    def __init__(self):
        super(MSECELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
    
    def forward(self, inputs, rgb_target, cls_target, weight=0.):
        cls_target = cls_target.squeeze()
        loss = {}
        mse_wg = weight
        ce_wg = 1 - weight
        # mse_loss = self.mse
        mse_loss = self.mse(inputs['rgb_coarse'].reshape(-1,3), rgb_target.reshape(-1,3))

        cls_target = cls_target.to(torch.long).reshape(-1)
        obj_mask = (cls_target != 0 ).to(dtype=torch.long, device=cls_target.device)
        # print(inputs['cls_coarse'].shape, cls_target.shape, "loss")

        if DEBUG:
            print(inputs['cls_coarse'].shape, cls_target.shape, "loss")
            pred_res = torch.max(inputs['cls_coarse'], axis=-1)
            print(pred_res , cls_target)
            print(obj_mask.sum())

        ce_loss = self.ce(inputs['cls_coarse'], cls_target)
        if "rgb_fine" in inputs:
            mse_loss += self.mse(inputs['rgb_fine'], rgb_target)
            ce_loss += self.ce(inputs['cls_fine'], cls_target)

        mse_loss *= mse_wg
        ce_loss *= ce_wg
        loss["sum"] = mse_loss + ce_loss
        loss["rgb"] = mse_loss
        loss["cls"] = ce_loss

        print(ce_loss)
        return loss


class MSENLLLoss(nn.Module):
    # need update render and nerf model
    def __init__(self):
        super(MSENLLLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, rgb_target, cls_target, weight=0.):

        # print(inputs['cls_coarse'].shape, cls_target.shape)
        loss = {}
        cls_target = torch.squeeze(cls_target)
        cls_target = cls_target.to(torch.long)
        obj_mask = (cls_target != 0 ).to(dtype=torch.long, device=cls_target.device)

        cls_coarse = inputs['cls_coarse'].cuda()
        # ingore non-sample points
        # print(cls_coarse.shape, rgb_target.shape, cls_target.shape, inputs['rgb_coarse'].shape)
        
        rgb_loss = self.loss(inputs['rgb_coarse'].reshape(-1,3), rgb_target.reshape(-1,3))
    
        _print_mask = cls_target !=0
        # if DEBUG: print(torch.max(cls_coarse, dim=-1)[1][_print_mask], cls_target[_print_mask], "***")
        # cls_loss = F.nll_loss(cls_coarse[obj_mask], cls_target[obj_mask], reduction='mean')
        cls_loss = F.nll_loss(cls_coarse, cls_target.reshape(-1), reduction='mean')

        if 'rgb_fine' in inputs:
            rgb_loss += self.loss(inputs['rgb_fine'], rgb_target.reshape(-1,3))
            cls_fine = inputs['cls_fine'].cuda()
            # add obj_mask when rgb fine
            # cls_loss += F.nll_loss(cls_fine[obj_mask], cls_target[obj_mask], reduction='mean')
            cls_loss += F.nll_loss(cls_fine, cls_target.reshape(-1), reduction='mean')
            # if DEBUG: print(torch.max(cls_fine, dim=-1)[1][_print_mask], cls_target[_print_mask], "***", cls_loss)

        weight = 0.99
        loss["rgb"] = rgb_loss * weight
        loss["cls"] = cls_loss * (1-weight) 
        loss["sum"] = loss["rgb"] + loss["cls"]

        return loss




# loss_dict = {'mse': MSELoss, "msece": MSECELoss, "msenll": MSENLLLoss, "perceptual": PerceptualLoss, "pm": PercepMseLoss}
loss_dict = {'mse': MSELoss, "msece": MSECELoss, "msenll": MSENLLLoss, "perceptual": perceptual, "pm": PercepMseLoss, "pm16": MSEVGG16PercepLoss}
