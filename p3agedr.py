#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/8/29 16:55
# @Author: ZhaoKe
# @File : p3gaedr.py
# @Software: PyCharm
import itertools
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# from torch import optim

from torchvision import transforms
from mylibs.conv_vae import ConvVAE
from audiokits.transforms import *


def normalize_data(train_df, test_df):
    # compute the mean and std (pixel-wise)
    mean = train_df['melspectrogram'].mean()
    std = np.std(np.stack(train_df['melspectrogram']), axis=0)

    # normalize train set
    train_spectrograms = (np.stack(train_df['melspectrogram']) - mean) / (std+1e-6)
    train_labels = train_df['label'].to_numpy()
    train_folds = train_df['fold'].to_numpy()
    train_df = pd.DataFrame(zip(train_spectrograms, train_labels, train_df["cough_type"], train_df["severity"], train_folds), columns=['melspectrogram', 'label', "cough_type", "severity", 'fold'])

    # normalize test set
    test_spectrograms = (np.stack(test_df['melspectrogram']) - mean) / (std+1e-6)
    test_labels = test_df['label'].to_numpy()
    test_folds = test_df['fold'].to_numpy()
    test_df = pd.DataFrame(zip(test_spectrograms, test_labels, train_df["cough_type"], train_df["severity"], test_folds), columns=['melspectrogram', 'label', "cough_type", "severity", 'fold'])

    return train_df, test_df



class CoughvidDataset(Dataset):
    def __init__(self, us8k_df, transform=None):
        assert isinstance(us8k_df, pd.DataFrame)
        assert len(us8k_df.columns) == 5

        self.us8k_df = us8k_df
        self.transform = transform

    def __len__(self):
        return len(self.us8k_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, label, cough_type, severity, fold = self.us8k_df.iloc[index]

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return {'spectrogram': spectrogram, 'label': label, "cough_type": cough_type, "severity": severity}


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


class AME(nn.Module):
    def __init__(self, em_dim, class_num=2, oup=16):
        super().__init__()
        self.mapping = nn.Embedding(num_embeddings=class_num, embedding_dim=oup)
        self.emb_lin_mu = nn.Linear(oup, em_dim)
        self.emb_lin_lv = nn.Linear(oup, em_dim)
    @staticmethod
    def sampling(mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    def forward(self, in_a):
        """

        :param in_a:
        :return: mu, logvar, z
        """
        mapd = self.mapping(in_a)
        res_mu = self.emb_lin_mu(mapd)
        res_logvar = self.emb_lin_lv(mapd)
        return res_mu, res_logvar, self.sampling(res_mu, res_logvar)


class Classifier(nn.Module):
    def __init__(self, dim_embedding, dim_hidden_classifier, num_target_class):
        super(Classifier, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(dim_embedding, dim_hidden_classifier),
            nn.BatchNorm1d(dim_hidden_classifier),
            nn.ReLU(),
            nn.Linear(dim_hidden_classifier, num_target_class)
        )

    def forward(self, input_data):
        classification_logit = self.decoder(input_data)
        return classification_logit


# Attribute Gaussian Embedding Disentangled Representation
class AGEDRTrainer(object):
    def __init__(self):
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.img_size, self.channel = (64, 128), 1
        self.class_num, self.batch_size = 2, 16
        self.latent_dim = 30
        self.configs = {
            "channels": 1,
            "class_num": 10,
            "code_dim": 2,
            "img_size": 32,
            "latent_dim": 62,
            "run_save_dir": "./run/infogan/",
            "sample_interval": 400,
            "fit": {
                "b1": 0.5,
                "b2": 0.999,
                "batch_size": 64,
                "epochs": 40,
                "learning_rate": 0.0002,
            }
        }
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __build_models(self):
        # vocab_size = 11 # 我们想要将每个单词映射到的向量维度
        self.embedding1 = nn.Embedding(num_embeddings=self.class_num, embedding_dim=6)
        self.embedding2 = nn.Embedding(num_embeddings=self.class_num, embedding_dim=8)
        self.vae = ConvVAE(inp_shape=(1, 64, 128), latent_dim=self.latent_dim, flat=True)
        self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                     num_target_class=self.class_num)

        self.optimizer_Em = torch.optim.Adam(
            itertools.chain(self.embedding1.parameters(), self.embedding2.parameters()),
            lr=self.configs["fit"]["learning_rate"],
            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=self.configs["fit"]["learning_rate"],
                                              betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_C = torch.optim.Adam(self.classifier.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))

        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.adversarial_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.MSELoss()

    def __build_dataloaders(self, batch_size=32):
        self.coughvid_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_split_specattri.pkl")
        self.coughvid_df = self.coughvid_df.iloc[:, [0, 1, 2, 8, 9]]
        neg_list = list(range(2076))
        pos_list = list(range(2076, 2850))
        random.shuffle(neg_list)
        random.shuffle(pos_list)
        valid_list = neg_list[:100] + pos_list[:100]
        train_list = neg_list[100:] + pos_list[100:]
        train_df = self.coughvid_df.iloc[train_list, :]
        valid_df = self.coughvid_df.iloc[valid_list, :]
        print(train_df.head())
        print(train_df.shape, valid_df.shape)
        # normalize the data
        train_df, valid_df = normalize_data(train_df, valid_df)
        self.train_transforms = transforms.Compose([MyRightShift(input_size=(128, 64),
                                                                 width_shift_range=7,
                                                                 shift_probability=0.9),
                                                    MyAddGaussNoise(input_size=(128, 64),
                                                                    add_noise_probability=0.55),
                                                    MyReshape(output_size=(1, 128, 64))])
        self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 64))])
        train_ds = CoughvidDataset(train_df, transform=self.train_transforms)
        self.train_loader = DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=0)
        # init test data loader
        valid_ds = CoughvidDataset(valid_df, transform=self.test_transforms)
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=0)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def __to_cuda(self):
        self.embedding1.to(self.device)
        self.embedding2.to(self.device)
        self.vae.to(self.device)
        self.classifier.to(self.device)

        self.adversarial_loss.to(self.device)
        self.categorical_loss.to(self.device)
        self.continuous_loss.to(self.device)

    def demo(self):
        device = torch.device("cuda")
        self.__build_dataloaders(batch_size=32)
        ame1 = AME(class_num=3, em_dim=6).to(device)
        ame2 = AME(class_num=4, em_dim=8).to(device)
        vae = ConvVAE(inp_shape=(1, 64, 128), latent_dim=self.latent_dim, flat=True).to(device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(device)
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(device)
            y_lab = batch["label"].to(device)
            ctype = batch["cough_type"].to(device)
            sevty = batch["severity"].to(device)
            print(x_mel.shape, y_lab.shape, ctype.shape, sevty.shape)
            mu_a_1, logvar_a_1, _ = ame1(ctype)
            print(mu_a_1.shape, logvar_a_1.shape)
            mu_a_2, logvar_a_2, _ = ame2(sevty)
            print(mu_a_2.shape, logvar_a_2.shape)
            x_recon, z_mu, z_logvar, z_latent = vae(x_mel)
            print(z_logvar.shape)
            y_pred = classifier(z_latent)
            print(y_pred.shape)
            break


if __name__ == '__main__':
    agedr = AGEDRTrainer()
    agedr.demo()
