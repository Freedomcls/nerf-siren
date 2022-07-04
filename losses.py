import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
import ast

DEBUG = ast.literal_eval(os.environ.get("DEBUG", "False"))

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

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

loss_dict = {'mse': MSELoss, "msece": MSECELoss, "msenll": MSENLLLoss}
