import torch
from torch import nn
import os
import numpy as np
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

        # ce_loss = self.ce(inputs['cls_coarse'][obj_mask], ce_target[obj_mask])
        ce_loss = self.ce(inputs['cls_coarse'], ce_target)

        if "rgb_fine" in inputs:
            mse_loss += self.mse(inputs['rgb_fine'], mse_target)
            ce_loss += self.ce(inputs['cls_fine'], ce_target)

        mse_loss *= mse_wg 
        ce_loss *= ce_wg

        loss["sum"] = mse_loss + ce_loss
        loss["mse"] = mse_loss
        loss["ce"] = ce_loss
        return loss

loss_dict = {'mse': MSELoss, "msece": MSECELoss}