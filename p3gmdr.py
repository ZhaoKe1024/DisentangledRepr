#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/15 10:04
# @Author: ZhaoKe
# @File : p3gmdr.py
# @Software: PyCharm
# Gaussian Mixture Disentangled Representation
import os
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(data_dir='./data/mnist/', batch_size=128):
    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)

    X = torch.cat([train.data.float().view(-1, 784) / 255., test.data.float().view(-1, 784) / 255.], 0)
    Y = torch.cat([train.targets, test.targets], 0)

    dataset = dict()
    dataset['X'] = X
    dataset['Y'] = Y

    dataloader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, dataset


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers


class Encoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            *block(input_dim, inter_dims[0]),
            *block(inter_dims[0], inter_dims[1]),
            *block(inter_dims[1], inter_dims[2]),
        )

        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)

        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)

        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            *block(hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_pro = self.decoder(z)

        return x_pro


class VaDE(nn.Module):
    def __init__(self, args):
        super(VaDE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.pi_ = nn.Parameter(torch.FloatTensor(args.nClusters, ).fill_(1) / args.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)

        self.args = args


if __name__ == '__main__':
    img_size, channel = 32, 1
    class_num, batch_size = 10, 16
    latent_dim, code_dim = 36, 4

    x = torch.rand(size=(batch_size, channel, img_size, img_size))
    # noise =
    label = torch.randint(0, class_num, size=(batch_size,))
    code = torch.rand(size=(batch_size, code_dim))

    encoder = Encoder()
    decoder = Decoder()
