#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/16 18:32
# @Author: ZhaoKe
# @File : p2ammidr.py
# @Software: PyCharm
# adversarial maskers, mutual information disentangled representation
import os
# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from torch import optim
from torch.utils.data import DataLoader
from idbsr_libs.Dataset import MNIST_ROT
from cirl_libs.cirl_model import Masker


def get_datasets(dataset, train_batch_size, test_batch_size, cuda=False, root='Data'):
    print(f'Loading {dataset} dataset...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if dataset == 'mnist-rot':
        print("Load dataset MNIST_ROT")
        Dataset = MNIST_ROT
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_55_loader = DataLoader(dataset.test_55_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_65_loader = DataLoader(dataset.test_65_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print("Loading Datasets:")
        print(len(train_loader), len(test_loader))
        print(len(test_55_loader), len(test_65_loader))
        print('Done!\n')
        return train_loader, test_loader, test_55_loader, test_65_loader


# Encoder。alae的带有style模块，有点复杂；infogan没有；cirl的是resnet；
# IDB-SR的是VAEEncoder；
# 最终选择dstgib的Encoder
#


# Generator摘自同目录下的infogan_mnist.py
class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, class_num, code_dim, channels):
        super(Generator, self).__init__()
        input_dim = latent_dim + class_num + code_dim
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        # print(gen_input.shape)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Discriminator摘自同目录下的infogan_mnist.py
class Discriminator(nn.Module):
    def __init__(self, img_size, channels, n_classes, code_dim):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


if __name__ == '__main__':
    # from cirl_libs.ResNet import resnet18
    # masker = Masker(512, 512, 4 * 512, k=308)
    # resnet = resnet18(pretrained=True)
    # resnet = ConvNet()
    x = torch.rand(size=(16, 1, 28, 28))
    print(resnet(x).shape)

    feat = torch.rand(size=(16, 512))
    # print(feat[0])
    masked = masker(feat)[0]
    # print(masked)
    print(feat.shape, masked.shape)
