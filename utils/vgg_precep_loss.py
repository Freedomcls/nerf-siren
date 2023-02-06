""" refer from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    Diff with previous, use vgg16 for backbone.
"""


import torch
import torch.nn as nn
import torchvision

pretrain = "./pretrained_model/vgg16-397923af.pth"
class Vgg16GenFeats(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.vgg16()
        model.load_state_dict(torch.load(pretrain))
        blocks = []
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        

    def forward(self, images,  feature_layers=[0, 1, 2, 3], style_layers=[], norm=True, resize=False):
        """ forward image to get vgg feats

        Args:
            images (torch.Tensor): input images, NCHW
            feature_layers (list, optional): forward layers lvl. Defaults to [0, 1, 2, 3].
            style_layers (list, optional): _description_. Defaults to [].
            norm (bool, optional): _description_. Defaults to True.
            resize (bool, optional): _description_. Defaults to False.
        """
        # pre-process
        if images.shape[1] != 3:
            images = images.repeat(1, 3, 1, 1)
        if norm:
            images = (images - self.mean) / self.std
        if resize:
            images = self.transform(images, mode='bilinear', size=(224, 224), align_corners=False)
        assert images.shape[1] == 3, f"input channel need be 3 but get {images.shape}"
        # model forward
        x = images
        feats = []
        style_feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_layers:
                print(x.shape)
                feats.append(x)
            if i in style_layers:
                style_feats.append(x)
        return feats, style_feats
            

if __name__ == "__main__":
    class VGG16PercepLoss1(torch.nn.Module):
        def __init__(self, resize=False):
            super(VGG16PercepLoss1, self).__init__()
            blocks = []
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
            for bl in blocks:
                for p in bl.parameters():
                    p.requires_grad = False
            self.blocks = torch.nn.ModuleList(blocks)
            self.transform = torch.nn.functional.interpolate
            self.resize = resize
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
            if input.shape[1] != 3:
                input = input.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
            if self.resize:
                input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
                target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            loss = 0.0
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                if i in feature_layers:
                    loss += torch.nn.functional.l1_loss(x, y)
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.l1_loss(gram_x, gram_y)
            return loss

    class VGG16PercepLoss2(nn.Module):
        def __init__(self, features_layers=[0, 1, 2, 3], norm=True, resize=False):
            super().__init__()
            self.gen_feats = Vgg16GenFeats()
            self.norm = norm
            self.resize = resize
            self.feature_layers = features_layers

        def forward(self, input, target):
            inp_feats, inp_style = self.gen_feats(input, self.feature_layers, norm=self.norm, resize=self.resize)
            tar_feats, tar_style  = self.gen_feats(target, self.feature_layers, norm=self.norm, resize=self.resize)
            loss = 0.0

            for inp_f, tar_f in zip(inp_feats, tar_feats):
                loss += torch.nn.functional.l1_loss(inp_f, tar_f)
            for inp_f, tar_f in zip(inp_style, tar_style):
                act_inp = inp_f.reshape(inp_f.shape[0], inp_f.shape[1], -1)
                act_tar = tar_f.reshape(tar_f.shape[0], tar_f.shape[1], -1)
                gram_inp = act_inp @ act_inp.permute(0, 2, 1)
                gram_tar = act_tar @ act_tar.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_inp, gram_tar)

            return loss
    img1 = torch.rand(1, 3, 224, 224)
    img2 = torch.rand(1, 3, 224, 224)
    loss1 = VGG16PercepLoss1()(img1, img2)
    loss = VGG16PercepLoss2()(img1, img2)

    print(loss1, loss)


    
