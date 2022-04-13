import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F

DEBUG = os.environ.get("DEBUG", False)

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
    
    def forward(self, inputs, mse_target, ce_target, weight=0.6):
        ce_target = ce_target.squeeze()
        # ? ingore background ce loss, should we sampling some negative rays ?
        loss = {}
        mse_wg = weight
        ce_wg = 1 - weight
        mse_loss = self.mse(inputs['rgb_coarse'], mse_target)
        ce_target = ce_target.to(torch.long)
        obj_mask = (ce_target != 0 ).to(dtype=torch.long, device=ce_target.device)

        if DEBUG:
            print(inputs['cls_coarse'].shape, ce_target.shape, "loss")
            pred_res = torch.max(inputs['cls_coarse'], axis=-1)
            print(pred_res , ce_target)
            print(obj_mask.sum())
        # ce_loss = self.ce(inputs['cls_coarse'][obj_mask], ce_target[obj_mask])
        # 加不加mask, cls_fine的结果基本都是 0 
        ce_loss = self.ce(inputs['cls_coarse'], ce_target)

        if "rgb_fine" in inputs:
            mse_loss += self.mse(inputs['rgb_fine'], mse_target)
            ce_loss += self.ce(inputs['cls_fine'], ce_target)
            print(list(set(torch.argmax(inputs['cls_fine'], dim=-1).detach().cpu().numpy().tolist())), 
                "loss")
            # ce_loss += self.ce(inputs['cls_fine'][obj_mask], ce_target[obj_mask])

        mse_loss *= mse_wg
        ce_loss *= ce_wg

        loss["sum"] = mse_loss + ce_loss
        loss["rgb"] = mse_loss
        loss["cls"] = ce_loss
        return loss


class MSENLLLoss(nn.Module):
    # need update render and nerf model
    def __init__(self):
        super(MSENLLLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, rgb_target, cls_target, weight=0.6):
        # print(inputs['cls_coarse'].shape, cls_target.shape)
        loss = {}
        cls_target = torch.squeeze(cls_target)
        cls_target = cls_target.to(torch.long)
        obj_mask = (cls_target != 0 ).to(dtype=torch.long, device=cls_target.device)

        cls_coarse = inputs['cls_coarse'].cuda()
        # ingore non-sample points
        
        rgb_loss = self.loss(inputs['rgb_coarse'], rgb_target)
        
        # ignore_mask = np.logical_and( (cls_target == -1).cpu().detach().numpy(), (cls_coarse==-1).cpu().detach().numpy())
        # ignore_mask = torch.Tensor(ignore_mask).cuda()
        # ignore_mask = cls_target == -1

        _print_mask = cls_target !=0
        print(torch.max(cls_coarse, dim=-1)[1][_print_mask], cls_target[_print_mask], "***")
        cls_loss = F.nll_loss(cls_coarse, cls_target)

        if 'rgb_fine' in inputs:
            rgb_loss += self.loss(inputs['rgb_fine'], rgb_target)
            cls_fine = inputs['cls_fine'].cuda()
            # cls_loss += F.nll_loss(cls_fine, cls_target)
            cls_loss += F.nll_loss(cls_fine[obj_mask], cls_target[obj_mask])

            print(torch.max(cls_fine, dim=-1)[1][_print_mask], cls_target[_print_mask], "***", cls_loss)

        
        loss["rgb"] = rgb_loss * weight
        loss["cls"] = cls_loss * (1-weight)
        loss["sum"] = loss["rgb"] + loss["cls"]

        return loss

loss_dict = {'mse': MSELoss, "msece": MSECELoss, "msenll": MSENLLLoss}
