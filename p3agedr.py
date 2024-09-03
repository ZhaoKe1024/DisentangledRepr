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
from mylibs.conv_vae import ConvVAE, kl_2normal
from audiokits.transforms import *


def pairwise_kl_loss(mu, log_sigma, batch_size):
    mu1 = mu.unsqueeze(dim=1).repeat(1, batch_size, 1)
    log_sigma1 = log_sigma.unsqueeze(dim=1).repeat(1, batch_size, 1)

    mu2 = mu.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    log_sigma2 = log_sigma.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    print(log_sigma2.shape, log_sigma1.shape)  # ([32, 16, 30]) torch.Size([16, 32, 30])
    kl_divergence1 = 0.5 * (log_sigma2 - log_sigma1)
    kl_divergence2 = 0.5 * torch.div(torch.exp(log_sigma1) + torch.square(mu1 - mu2), torch.exp(log_sigma2))
    kl_divergence_loss1 = kl_divergence1 + kl_divergence2 - 0.5

    pairwise_kl_divergence_loss = kl_divergence_loss1.sum(-1).sum(-1) / (batch_size - 1)

    return pairwise_kl_divergence_loss


def normalize_data(train_df, test_df):
    # compute the mean and std (pixel-wise)
    mean = train_df['melspectrogram'].mean()
    std = np.std(np.stack(train_df['melspectrogram']), axis=0)

    # normalize train set
    train_spectrograms = (np.stack(train_df['melspectrogram']) - mean) / (std + 1e-6)
    train_labels = train_df['label'].to_numpy()
    train_folds = train_df['fold'].to_numpy()
    train_df = pd.DataFrame(
        zip(train_spectrograms, train_labels, train_df["cough_type"], train_df["severity"], train_folds),
        columns=['melspectrogram', 'label', "cough_type", "severity", 'fold'])

    # normalize test set
    test_spectrograms = (np.stack(test_df['melspectrogram']) - mean) / (std + 1e-6)
    test_labels = test_df['label'].to_numpy()
    test_folds = test_df['fold'].to_numpy()
    test_df = pd.DataFrame(
        zip(test_spectrograms, test_labels, train_df["cough_type"], train_df["severity"], test_folds),
        columns=['melspectrogram', 'label', "cough_type", "severity", 'fold'])

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


# class AGESR(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
#         self.ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
#         self.vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
#         self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
#                                      num_target_class=self.class_num).to(self.device)
#
#     def forward(self, x_mel, ctype, sevty):
#         mu_a_1, logvar_a_1, _ = self.ame1(ctype)  # [32, 6] [32, 6]
#         mu_a_2, logvar_a_2, _ = self.ame2(sevty)  # [32, 8] [32, 8]
#         x_recon, z_mu, z_logvar, z_latent = self.vae(x_mel)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
#         y_pred = self.classifier(z_latent)  # torch.Size([32, 2])
#         return


# Attribute Gaussian Embedding Disentangled Representation
class AGEDRTrainer(object):
    def __init__(self):
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.img_size, self.channel = (64, 128), 1
        self.class_num, self.batch_size = 2, 16
        self.a1len, self.a2len = 6, 8
        self.latent_dim = 30
        self.blen = self.latent_dim - self.a1len - self.a2len
        self.configs = {
            "recon_weight": 0.01,
            "cls_weight": 1.,
            "kl_beta_alpha_weight": 0.01,
            "kl_c_weight": 0.075,
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
        self.ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
        self.ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
        self.vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                     num_target_class=self.class_num).to(self.device)

        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.recon_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()

        self.optimizer_Em = torch.optim.Adam(
            itertools.chain(self.ame1.parameters(), self.ame2.parameters()), lr=self.configs["fit"]["learning_rate"],
            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=self.configs["fit"]["learning_rate"],
                                              betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_cls = torch.optim.Adam(self.classifier.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))

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
        self.ame1.to(self.device)
        self.ame2.to(self.device)
        self.vae.to(self.device)
        self.classifier.to(self.device)

        self.recon_loss.to(self.device)
        self.categorical_loss.to(self.device)

    def train(self):
        data_root = "F:/DATAS/mnist/MNIST-ROT"
        self.__build_dataloaders(batch_size=32)
        print("dataloader {}".format(len(self.train_loader)))
        self.__build_models()
        recon_weight = 0.01
        cls_weight = 1.
        kl_attri_weight = 0.01  # noise
        kl_latent_weight = 0.075   # clean
        save_dir = "./runs/agedr/" + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        Loss_List_Epoch = []
        for epoch_id in range(100):
            Loss_List_Total = []
            Loss_List_disen = []
            Loss_List_attri = []
            Loss_List_recon = []
            Loss_List_cls = []
            x_Recon = None
            for jdx, batch in enumerate(self.train_loader):
                x_mel = batch["spectrogram"].to(self.device)
                y_lab = batch["label"].to(self.device)
                ctype = batch["cough_type"].to(self.device)
                sevty = batch["severity"].to(self.device)
                bs = len(x_mel)
                # print("batch_size:", bs)
                # print("shape of input, x_mel y_lab attris:", x_mel.shape, y_lab.shape, ctype.shape, sevty.shape)

                self.optimizer_Em.zero_grad()
                self.optimizer_vae.zero_grad()
                self.optimizer_cls.zero_grad()

                mu_a_1, logvar_a_1, _ = self.ame1(ctype)  # [32, 6] [32, 6]
                mu_a_2, logvar_a_2, _ = self.ame2(sevty)  # [32, 8] [32, 8]
                x_recon, z_mu, z_logvar, z_latent = self.vae(x_mel)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
                y_pred = self.classifier(z_latent)  # torch.Size([32, 2])
                # print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
                # print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
                # print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)
                # print("shape of y_pred:", y_pred.shape)

                mu1_latent = z_latent[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
                mu2_latent = z_latent[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
                lv1_latent = z_latent[:, self.blen:self.blen + self.a1len]  # Size([32, 8])
                lv2_latent = z_latent[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 8])
                # print("mu1:{}, lv1:{}, mu2:{}, lv2:{}".format(mu1_latent.shape, lv1_latent.shape, mu2_latent.shape,
                #                                               lv2_latent.shape))

                Loss_attri = kl_2normal(mu_a_1, logvar_a_1, mu1_latent, lv1_latent)
                Loss_attri += kl_2normal(mu_a_2, logvar_a_2, mu2_latent, lv2_latent)
                Loss_disen = kl_latent_weight*pairwise_kl_loss(z_mu[:, :self.blen], z_logvar[:, :self.blen], bs)
                Loss_disen += kl_attri_weight*pairwise_kl_loss(z_mu[:, self.blen:], z_logvar[:, self.blen:], bs)
                Loss_recon = recon_weight*self.recon_loss(x_recon, x_mel)
                Loss_disen += Loss_recon
                Loss_cls = cls_weight*self.categorical_loss(y_pred, y_lab)

                Loss_total = Loss_cls + Loss_attri + Loss_disen.sum(-1)

                Loss_total.backward()
                self.optimizer_cls.step()
                self.optimizer_vae.step()
                self.optimizer_Em.step()
                # print(L_attri.shape, L_disen.shape, L_cls.shape, L_total.shape)
                # print(L_attri, L_disen, L_cls, L_total)

                Loss_List_Total.append(Loss_total.item())
                Loss_List_disen.append(Loss_disen.item())
                Loss_List_attri.append(Loss_attri.item())
                Loss_List_recon.append(Loss_recon.item())
                Loss_List_cls.append(Loss_cls.item())
                if jdx % 500 == 0:
                    print("Epoch {}, Batch {}".format(epoch_id, jdx))
                    print([np.array(Loss_List_Total).mean(),
                           np.array(Loss_List_disen).mean(),
                           np.array(Loss_List_attri).mean(),
                           np.array(Loss_List_recon).mean(),
                           np.array(Loss_List_cls).mean()])
                if epoch_id == 0 and jdx == 0:
                    print("z_h:{}, a1.shape:{}, a2.sahpe:{}".format(z_latent.shape, mu1_latent.shape, mu2_latent.shape))
                    print("pred:{}; x_Recon:{}".format(y_pred.shape, x_recon.shape))
                    print("part[cls] cls loss:{};".format(Loss_cls))
                    print("part[disen] beta alpha kl loss:{};".format(Loss_disen))

                    print("part[attri] pdf loss:{};".format(Loss_attri))
                    print("part[recon] recon loss:{};".format(Loss_recon))
            Loss_List_Epoch.append([np.array(Loss_List_Total).mean(),
                                  np.array(Loss_List_disen).mean(),
                                  np.array(Loss_List_attri).mean(),
                                  np.array(Loss_List_cls).mean(),
                                  np.array(Loss_List_recon).mean()])
            print("Loss Parts:")
            print(Loss_List_Epoch)
            if epoch_id > 4:
                save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
                os.makedirs(save_dir_epoch, exist_ok=True)
                torch.save(self.ame1.state_dict(), save_dir_epoch + "epoch_{}_ame1.pth".format(epoch_id))
                torch.save(self.ame2.state_dict(), save_dir_epoch + "epoch_{}_ame2.pth".format(epoch_id))
                torch.save(self.vae.state_dict(), save_dir_epoch + "epoch_{}_vae.pth".format(epoch_id))
                torch.save(self.classifier.state_dict(), save_dir_epoch + "epoch_{}_classifier.pth".format(epoch_id))

                torch.save(self.optimizer_Em.state_dict(), save_dir_epoch + "epoch_{}_optimizer_Em".format(epoch_id))
                torch.save(self.optimizer_vae.state_dict(), save_dir_epoch + "epoch_{}_optimizer_vae".format(epoch_id))
                torch.save(self.optimizer_cls.state_dict(), save_dir_epoch + "epoch_{}_optimizer_cls".format(epoch_id))

            if epoch_id % 10 == 0:
                if epoch_id == 0:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                else:
                    with open(save_dir + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                        fin.write("total,cls,recon,klab,ib")
                        for epochlist in Loss_List_Epoch:
                            fin.write(",".join([str(it) for it in epochlist]))
                    plt.figure(0)
                    Loss_All_Lines = np.array(Loss_List_Epoch)
                    cs = ["black", "red", "green", "orange", "blue"]
                    for j in range(5):
                        dat_lin = Loss_All_Lines[:, j]
                        plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                    plt.savefig(save_dir + f'loss_iter_{epoch_id}.png')
                    plt.close(0)
            if x_Recon is not None:
                plt.figure(1)
                img_to_plot = x_Recon[:9].squeeze().data.cpu().numpy()
                for i in range(1, 4):
                    for j in range(1, 4):
                        plt.subplot(3, 3, (i - 1) * 3 + j)
                        plt.imshow(img_to_plot[(i - 1) * 3 + j - 1])
                plt.savefig(save_dir + "recon_epoch-{}.png".format(epoch_id), format="png")
                plt.close(1)
            else:
                raise Exception("x_Recon is None.")

    def demo(self):
        device = torch.device("cuda")
        self.__build_dataloaders(batch_size=32)
        ame1 = AME(class_num=3, em_dim=self.a1len).to(device)
        ame2 = AME(class_num=4, em_dim=self.a2len).to(device)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(device)
        recon_loss = nn.MSELoss()
        categorical_loss = nn.CrossEntropyLoss()
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(device)
            y_lab = batch["label"].to(device)
            ctype = batch["cough_type"].to(device)
            sevty = batch["severity"].to(device)
            bs = len(x_mel)
            print("batch_size:", bs)
            print("shape of input, x_mel y_lab attris:", x_mel.shape, y_lab.shape, ctype.shape, sevty.shape)
            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
            x_recon, z_mu, z_logvar, z_latent = vae(x_mel)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
            print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)
            y_pred = classifier(z_latent)  # torch.Size([32, 2])
            print("shape of y_pred:", y_pred.shape)

            mu1_latent = z_latent[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
            mu2_latent = z_latent[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
            lv1_latent = z_latent[:, self.blen:self.blen + self.a1len]  # Size([32, 8])
            lv2_latent = z_latent[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 8])
            print("mu1:{}, lv1:{}, mu2:{}, lv2:{}".format(mu1_latent.shape, lv1_latent.shape, mu2_latent.shape,
                                                          lv2_latent.shape))
            L_attri = kl_2normal(mu_a_1, logvar_a_1, mu1_latent, lv1_latent)
            L_attri += kl_2normal(mu_a_2, logvar_a_2, mu2_latent, lv2_latent)

            L_disen = pairwise_kl_loss(z_mu[:, :self.blen], z_logvar[:, :self.blen], bs)
            L_disen += pairwise_kl_loss(z_mu[:, self.blen:], z_logvar[:, self.blen:], bs)
            L_disen += recon_loss(x_recon, x_mel)

            L_cls = categorical_loss(y_pred, y_lab)

            L_total = L_cls + L_attri + L_disen.sum(-1)

            print(L_attri.shape, L_disen.shape, L_cls.shape, L_total.shape)
            print(L_attri, L_disen, L_cls, L_total)
            break


if __name__ == '__main__':
    agedr = AGEDRTrainer()
    agedr.demo()
