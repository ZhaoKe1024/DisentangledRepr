#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/25 16:03
# @Author: ZhaoKe
# @File : conv_vae.py
# @Software: PyCharm
from collections import OrderedDict
import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, inp_shape=(1, 64, 128), latent_dim=128, feat_c=8, flat=True):
        super().__init__()
        self.flat = flat
        self.encoder = ConvEncoder(inp_shape=inp_shape, flat=flat)
        self.cc, hh, ww = self.encoder.cc, self.encoder.hh, self.encoder.ww
        if flat:
            self.calc_mean = MLP([self.cc * hh * ww, 128, 64, latent_dim], last_activation=False)
            self.calc_logvar = MLP([self.cc * hh * ww, 128, 64, latent_dim], last_activation=False)
        else:
            self.calc_mean = nn.Conv2d(128, feat_c, kernel_size=1, stride=1, padding=0,
                                       bias=False)
            self.calc_logvar = nn.Conv2d(128, feat_c, kernel_size=1, stride=1, padding=0,
                                         bias=False)
        self.decoder = ConvDecoder(inp_shape=(self.cc, hh, ww), flat=flat, latent_dim=latent_dim, feat_c=feat_c)

        # self.cls = nn.Sequential()
        # self.cls.append(nn.Linear(hidden_dim, 32))
        # self.cls.append(nn.BatchNorm1d(32))
        # self.cls.append(nn.ReLU(inplace=True))
        # self.cls.append(nn.Linear(32, class_num))

    @staticmethod
    def sampling(mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, dec=True):
        # encoder
        x_feat = self.encoder(x)
        # print("x_feat conv2:", x_feat.shape)
        # flatten

        mean_lant, logvar_lant = self.calc_mean(x_feat), self.calc_logvar(x_feat)
        if dec:
            z = self.sampling(mean_lant, logvar_lant, device=torch.device("cuda"))

            # print("recon:", z.shape)  # Size([32, 30])
            x_recon = self.decoder(inp_feat=z, shape_list=self.encoder.shapes)
            # x_pred = self.cls(x_feat)
            return x_recon, mean_lant, logvar_lant, z
        else:
            return mean_lant


class ConvEncoder(nn.Module):
    def __init__(self, inp_shape=(1, 64, 128), flat=True):
        super().__init__()
        c, h, w = inp_shape
        hh, ww = h, w
        self.shapes = [(hh, ww)]
        self.encoder_conv1 = nn.Sequential()

        self.encoder_conv1.append(nn.Conv2d(c, 16, kernel_size=4, stride=2, padding=1))
        self.encoder_conv1.append(nn.BatchNorm2d(16))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 4 + 2 * 1) / 2) + 1  # 32
        ww = int((ww - 4 + 2 * 1) / 2) + 1  # 64
        self.shapes.append((hh, ww))

        self.encoder_conv1.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        self.encoder_conv1.append(nn.BatchNorm2d(32))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 3 + 2 * 1) / 2) + 1  # 16
        ww = int((ww - 3 + 2 * 1) / 2) + 1  # 32
        self.shapes.append((hh, ww))

        # self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # hh, ww = hh // 2, ww // 2
        # self.shapes.append(())

        self.encoder_conv2 = nn.Sequential()
        self.encoder_conv2.append(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.encoder_conv2.append(nn.BatchNorm2d(64))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = int((hh - 4 + 2 * 1) / 2) + 1  # 8
        ww = int((ww - 4 + 2 * 1) / 2) + 1  # 16
        self.shapes.append((hh, ww))

        self.encoder_conv2.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0))
        self.encoder_conv2.append(nn.BatchNorm2d(128))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = (hh - 3 + 2 * 0) // 2 + 1  # 3
        ww = (ww - 3 + 2 * 0) // 2 + 1  # 7
        self.shapes.append((hh, ww))

        self.encoder_conv2.append(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0))
        self.encoder_conv2.append(nn.BatchNorm2d(128))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = (hh - 3 + 2 * 0) // 2 + 1  # 1
        ww = (ww - 3 + 2 * 0) // 2 + 1  # 3
        self.shapes.append((hh, ww))
        # ----------------- flatten (128, 5, 7) -> 128*5*7 ------------
        self.cc = 128
        self.hh, self.ww = self.shapes[-1]
        self.flat = flat
        if flat:
            self.flatten = nn.Flatten(start_dim=1)

    def forward(self, input_x):
        # encoder
        x_feat = self.encoder_conv2(self.encoder_conv1(input_x))
        # print("x_feat conv2:", x_feat.shape)
        # flatten
        if self.flat:
            x_feat = self.flatten(x_feat)
        return x_feat


class ConvDecoder(nn.Module):
    def __init__(self, inp_shape=(128, 1, 3), latent_dim=128, feat_c=8, flat=True):
        # ------------------------decoder---------------
        super().__init__()
        self.flat = flat
        self.decoder_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=0)
        self.decoder_norm1 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.decoder_norm2 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_norm3 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.decoder_conv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.decoder_norm4 = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        self.decoder_conv5 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        if flat:
            self.decoder_lin = MLP([latent_dim, 64, 128, inp_shape[0] * inp_shape[1] * inp_shape[2]])
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(inp_shape[0], inp_shape[1], inp_shape[2]))
        else:
            self.decoder_proj = nn.Conv2d(feat_c, 128, kernel_size=1, stride=1, padding=0,
                                          bias=False)

    def forward(self, inp_feat, shape_list):
        if self.flat:
            x_recon = self.unflatten(self.decoder_lin(inp_feat))
        else:
            x_recon = self.decoder_proj(inp_feat)
        # print(x_recon.shape)
        # decoder
        x_recon = self.decoder_conv1(x_recon, output_size=shape_list[-2])
        x_recon = self.decoder_norm1(x_recon)
        x_recon = self.decoder_conv2(x_recon, output_size=shape_list[-3])
        x_recon = self.decoder_norm2(x_recon)
        x_recon = self.decoder_conv3(x_recon, output_size=shape_list[-4])
        x_recon = self.decoder_norm3(x_recon)
        x_recon = self.decoder_conv4(x_recon, output_size=shape_list[-5])
        x_recon = self.decoder_norm4(x_recon)
        x_recon = self.decoder_conv5(x_recon, output_size=shape_list[-6])
        return x_recon


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and last_activation):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


MSE_loss = nn.MSELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight * KL_divergence


def kl_2normal(pmu, plogvar, qmu, qlogvar):
    """
    :param recon_x: generated image
    :param x: original image
    :param mu: latent mean of z
    :param logvar: latent log variance of z
    """
    return -0.5 * torch.sum(1 - qlogvar + plogvar - (torch.exp(plogvar) + (pmu - qmu) ** 2) / torch.exp(qlogvar))


def vae_loss_fn(recon_x, x, mean, log_var, kl_weight=0.00025):
    BCE = torch.nn.functional.mse_loss(
        recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    print(BCE.shape, KLD.shape)
    # kl_weight = 0.00025
    print(BCE, KLD)
    return (BCE + kl_weight * KLD) / x.size(0)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # m = ConvVAE(inp_shape=(1, 94, 128), flat=True).to(device)
    # x = torch.randn(size=(16, 1, 94, 128)).to(device)
    # x_recon, z, mean_lant, logvar_lant = m(x)
    # print(x_recon.shape, z.shape, mean_lant.shape, logvar_lant.shape)

    # m = ConvVAE(inp_shape=(1, 94, 128), flat=False).to(device)
    # x_recon, z, mean_lant, logvar_lant = m(x)
    # print(x_recon.shape, z.shape, mean_lant.shape, logvar_lant.shape)

    enc1 = ConvEncoder(inp_shape=(1, 64, 128), flat=True).to(device)
    enc2 = ConvEncoder(inp_shape=(1, 64, 128), flat=False).to(device)
    x = torch.randn(size=(16, 1, 64, 128)).to(device)
    x_feat1 = enc1(x)  # [16, 4480]
    x_feat2 = enc2(x)  # [16, 128, 5, 7]
    print(x_feat1.shape, x_feat2.shape)
    print(enc1.shapes)
    print(enc2.shapes)
    dec1 = ConvDecoder(inp_shape=(128, 1, 3), flat=True, latent_dim=128, feat_c=8).to(device)
    dec2 = ConvDecoder(inp_shape=(128, 1, 3), flat=False, latent_dim=128, feat_c=8).to(device)
    l_feat1 = torch.randn(size=(16, 128)).to(device)
    l_feat2 = torch.randn(size=(16, 8, 1, 3)).to(device)
    x_recon1 = dec1(l_feat1, enc1.shapes)
    x_recon2 = dec2(l_feat2, enc2.shapes)
    print(x_recon1.shape, x_recon2.shape)
