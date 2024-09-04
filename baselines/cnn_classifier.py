#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/18 10:23
# @Author: ZhaoKe
# @File : cnn_classifier.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.functional as F


class CNNCls(nn.Module):
    def __init__(self, input_channel, input_shape=(8, 33, 13), conv_out_dim=32, class_num=7):
        super().__init__()
        cc, hh, ww = input_shape
        self.conv1 = nn.Conv2d(input_channel, 16, 3, stride=2)  # (16, 6)
        hh, ww = (hh-3)//2+1, (ww-3)//2+1
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, conv_out_dim, 2, stride=(2, 1))  # (8, 5)
        hh, ww = (hh - 2) // 2 + 1, (ww - 2) // 1 + 1
        cc = conv_out_dim
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(cc*hh*ww, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, class_num)

    def forward(self, featmap):
        featmap = F.relu(self.conv1(featmap))
        featmap = F.relu(self.conv2(featmap))

        featmap = self.flatten(featmap)  # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(featmap))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, featmap


if __name__ == '__main__':
    import torch
    model = CNNCls(8,)
    x = torch.randn(size=(16, 8, 33, 13))
    out = model(x)
    print(out.shape)
