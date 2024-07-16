#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/16 17:33
# @Author: ZhaoKe
# @File : iclr_model.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F


class Masker(nn.Module):
    # Masker(in_dim=512, num_classes=512, middle=4*512, k=308)
    def __init__(self, in_dim=2048, num_classes=2048, middle=8192, k=1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),

            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
        mask = self.bn(self.layers(f))
        z = torch.zeros_like(mask)
        for _ in range(self.k):
            mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
            z = torch.maximum(mask, z)
        return z


if __name__ == '__main__':
    from cirl_libs.ResNet import resnet18
    masker = Masker(512, 512, 4*512, k=308)
    resnet = resnet18(pretrained=True)
    x = torch.rand(size=(16, 3, 512, 512))
    print(resnet(x).shape)
    feat = torch.rand(size=(16, 512))
    print(masker(feat)[0])
