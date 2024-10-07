#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/10/7 11:59
# @Author: ZhaoKe
# @File : modules4mnist.py
# @Software: PyCharm
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor, transform=None):
        super(MyDataset, self).__init__()
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = torch.from_numpy(np.float32(data_tensor))
        self.target_tensor = torch.from_numpy(np.float32(np.reshape(target_tensor, target_tensor.shape[0])))
        self.sensitive_tensor = torch.from_numpy(np.float32(np.reshape(sensitive_tensor, sensitive_tensor.shape[0])))
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is None:
            return self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[index]
        else:
            return self.transform(self.data_tensor[index]), self.target_tensor[index], self.sensitive_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class MyDatasetWithoutSensitive(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        super(MyDatasetWithoutSensitive, self).__init__()
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = torch.from_numpy(np.float32(data_tensor))
        self.target_tensor = torch.from_numpy(np.float32(np.reshape(target_tensor, target_tensor.shape[0])))
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data_tensor[index]), self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class MNIST_ROT(Dataset):
    def __init__(self, dataset_path, transform):
        self.dataset_path = os.path.join(dataset_path, 'MNIST-ROT')
        self.train_dataset, self.test_dataset, self.test_55_dataset, self.test_65_dataset = self.load()
        self.transform = transform

    def load(self):
        train_data = np.load(os.path.join(self.dataset_path, 'train_data.npy'))  # / 255.0
        train_data = train_data.reshape(-1, 1, 28, 28)
        train_labels = np.load(os.path.join(self.dataset_path, 'train_labels.npy'))
        train_sensitive_labels = np.load(os.path.join(self.dataset_path, 'train_sensitive_labels.npy'))
        test_data = np.load(os.path.join(self.dataset_path, 'test_data.npy'))  # / 255.0
        test_data = test_data.reshape(-1, 1, 28, 28)
        test_labels = np.load(os.path.join(self.dataset_path, 'test_labels.npy'))
        test_sensitive_labels = np.load(os.path.join(self.dataset_path, 'test_sensitive_labels.npy'))

        test_55_data = np.load(os.path.join(self.dataset_path, 'test_55_data.npy'))  # / 255.0
        test_55_data = test_55_data.reshape(-1, 1, 28, 28)
        test_55_labels = np.load(os.path.join(self.dataset_path, 'test_55_labels.npy'))

        test_65_data = np.load(os.path.join(self.dataset_path, 'test_65_data.npy'))  # / 255.0
        test_65_data = test_65_data.reshape(-1, 1, 28, 28)
        test_65_labels = np.load(os.path.join(self.dataset_path, 'test_65_labels.npy'))

        train_dataset = MyDataset(train_data, train_labels, train_sensitive_labels, self.transform)
        test_dataset = MyDataset(test_data, test_labels, test_sensitive_labels, self.transform)
        test_55_dataset = MyDatasetWithoutSensitive(test_55_data, test_55_labels, self.transform)
        test_65_dataset = MyDatasetWithoutSensitive(test_65_data, test_65_labels, self.transform)

        return train_dataset, test_dataset, test_55_dataset, test_65_dataset


def get_datasets(dataset, train_batch_size, test_batch_size, cuda=False, root='Data', transform=None):
    print(f'Loading {dataset} dataset...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if dataset == 'mnist-rot':
        print("Load dataset MNIST_ROT")
        dataset_path = os.path.join(root, 'mnist')
        dataset = MNIST_ROT(dataset_path, transform)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_55_loader = DataLoader(dataset.test_55_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_65_loader = DataLoader(dataset.test_65_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print("Loading Datasets:")
        print(len(train_loader), len(test_loader))
        print(len(test_55_loader), len(test_65_loader))
        print('Done!\n')
        return train_loader, test_loader, test_55_loader, test_65_loader
    else:
        raise Exception("Unknown Dataset Name.")


def init_weights(layer):
    """
    Initialize weights.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None: layer.bias.data.zero_()


# Encoder。alae的带有style模块，有点复杂；infogan没有；cirl的是resnet；
# Idstgib的Encoder，是AutoEncoder；
# 最终选择DB-SR的是VAEEncoder


class EncoderMNIST(nn.Module):
    """
    Encoder Module.
    """

    def __init__(self, nz):
        super(EncoderMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.ReLU(inplace=True))
        # (2) FC
        # self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # self._fc = nn.Linear(in_features=576, out_features=nz, bias=True)

        self.mu_encoder = nn.Linear(in_features=576, out_features=nz)
        self.logvar_encoder = nn.Linear(in_features=576, out_features=nz)
        self.logpi_encoder = nn.Linear(in_features=576, out_features=nz)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        # print(x.shape)
        # ret = self._fc(x)
        ret_mu = self.mu_encoder(x)
        ret_logvar = self.logvar_encoder(x)
        ret_logpi = self.logpi_encoder(x)
        ret_gamma = torch.sigmoid(ret_logpi)
        # Return
        return ret_mu, ret_logvar, ret_logpi, ret_gamma


# Generator摘自同目录下的infogan_mnist.py
class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, class_dim, code_dim, channels):
        super(Generator, self).__init__()
        input_dim = latent_dim + class_dim + code_dim
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
        self.apply(init_weights)

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        # print(gen_input.shape)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Discriminator摘自同目录下的infogan_mnist.py
class Discriminator(nn.Module):
    def __init__(self, img_size, channels, n_classes, latent_dim):
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
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, latent_dim))
        self.apply(init_weights)

    def forward(self, img):
        out = self.conv_blocks(img)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# 参考IDB_SR的分类器
class VIB_Classifier(nn.Module):
    def __init__(self, dim_embedding, dim_hidden_classifier, num_target_class):
        super(VIB_Classifier, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(dim_embedding, dim_hidden_classifier),
            nn.BatchNorm1d(dim_hidden_classifier),
            nn.ReLU(),
            nn.Linear(dim_hidden_classifier, num_target_class)
        )

    def forward(self, input_data):
        classification_logit = self.decoder(input_data)
        return classification_logit
