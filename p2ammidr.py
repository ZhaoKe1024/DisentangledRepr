#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/16 18:32
# @Author: ZhaoKe
# @File : p2ammidr.py
# @Software: PyCharm
# adversarial maskers, mutual information disentangled representation
import itertools
import os
import time
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from mylibs.conv_vae import ConvEncoder, ConvDecoder, MLP, vae_loss_fn
from audiokits.transforms import *
from mylibs.modules import *


class ConvVAE(nn.Module):
    def __init__(self, inp_shape=(1, 64, 128), vae_latent_dim=16, latent_dim=30, feat_c=8, flat=True):
        super().__init__()
        self.flat = flat
        self.encoder = ConvEncoder(inp_shape=inp_shape, flat=flat)
        self.cc, hh, ww = self.encoder.cc, self.encoder.hh, self.encoder.ww
        if flat:
            self.calc_mean = MLP([self.cc * hh * ww, 128, 64, vae_latent_dim], last_activation=False)
            self.calc_logvar = MLP([self.cc * hh * ww, 128, 64, vae_latent_dim], last_activation=False)
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

    def forward(self, x, mu1=None, lv1=None, mu2=None, lv2=None, dec=True):
        # encoder
        x_feat = self.encoder(x)
        # print("x_feat conv2:", x_feat.shape)
        # flatten

        mean_lant, logvar_lant = self.calc_mean(x_feat), self.calc_logvar(x_feat)
        if dec:
            z = self.sampling(mean_lant, logvar_lant, device=torch.device("cuda"))
            ah1 = self.sampling(mu1, lv1)
            ah2 = self.sampling(mu2, lv2)
            z = torch.concat((z, ah1, ah2), dim=-1)
            # print("recon:", z.shape)  # Size([32, 30])
            x_recon = self.decoder(inp_feat=z, shape_list=self.encoder.shapes)
            # x_pred = self.cls(x_feat)
            return x_recon, mean_lant, logvar_lant, z
        else:
            return mean_lant


class AMDR_CLS(nn.Module):
    def __init__(self, vae, ame1, ame2, cls):
        super().__init__()
        self.vae = vae
        self.ame1 = ame1
        self.ame2 = ame2
        self.cls = cls

    def load_state_from_path(self, state_dir, epoch_id):
        self.ame1.load_state_dict(torch.load("{}/epoch_{}_ame1.pth".format(state_dir, epoch_id)))
        self.ame2.load_state_dict(torch.load("{}/epoch_{}_ame2.pth".format(state_dir, epoch_id)))
        self.vae.load_state_dict(torch.load("{}/epoch_{}_vae.pth".format(state_dir, epoch_id)))
        self.ame1.eval()
        self.ame2.eval()
        self.vae.eval()

    def to_train(self):
        self.cls.train()

    def to_eval(self):
        self.cls.eval()

    def forward(self, x_input, a1, a2):
        with torch.no_grad():
            mu_a_1 = self.ame1(a1, mu_only=True)  # [32, 6] [32, 6]
            mu_a_2 = self.ame2(a2, mu_only=True)  # [32, 8] [32, 8]
            z_mu = self.vae(x_input, dec=False)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
        y_pred = self.cls(torch.concat((z_mu, mu_a_1, mu_a_2), dim=-1))  # torch.Size([32, 2])
        return y_pred


def pairwise_zc_kl_loss(mu, log_sigma, gamma, batch_size):
    eps = 1e-6
    gamma = torch.clamp(gamma, min=eps, max=1 - eps)

    mu1 = mu.unsqueeze(dim=1).repeat(1, batch_size, 1)
    log_sigma1 = log_sigma.unsqueeze(dim=1).repeat(1, batch_size, 1)
    gamma1 = gamma.unsqueeze(dim=1).repeat(1, batch_size, 1)

    mu2 = mu.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    log_sigma2 = log_sigma.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    gamma2 = gamma.unsqueeze(dim=0).repeat(batch_size, 1, 1)

    kl_divergence1 = 0.5 * (log_sigma2 - log_sigma1)
    kl_divergence2 = 0.5 * torch.div(torch.exp(log_sigma1) + torch.square(mu1 - mu2), torch.exp(log_sigma2))
    kl_divergence_loss1 = torch.mul(gamma1, kl_divergence1 + kl_divergence2 - 0.5)

    kl_divergence3 = (1 - gamma1).mul(torch.log(1 - gamma1) - torch.log(1 - gamma2))
    kl_divergence4 = gamma1.mul(torch.log(gamma1) - torch.log(gamma2))
    kl_divergence_loss2 = kl_divergence3 + kl_divergence4

    pairwise_kl_divergence_loss = (kl_divergence_loss1 + kl_divergence_loss2).sum(-1).sum(-1) / (batch_size - 1)

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


def bin_upsampling_balance(data_df):
    # print(data_df)
    df1 = data_df[data_df["label"] == 1]
    df2 = data_df[data_df["label"] == 0]
    res_df = None
    if len(df1) > len(df2):
        t = len(df1) // len(df2) - 1
        r = len(df1) - len(df2)
        # print("t r", t, r)
        for i in range(t):
            df1 = pd.concat((df1, df2))
        res_df = pd.concat((df1, df2.iloc[:r, :]))
    elif len(df2) > len(df1):
        t = len(df2) // len(df1)
        r = len(df2) % len(df1)
        # print("t r", t, r)
        for i in range(t):
            df2 = pd.concat((df2, df1))
        res_df = pd.concat((df2, df1.iloc[:r, :]))
    else:
        res_df = data_df
    return res_df


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


class AMMIDRTrainer(object):
    def __init__(self):
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.img_size, self.channel = (64, 128), 1
        self.class_num, self.batch_size = 2, 16
        self.vae_latent_dim = 16
        self.a1len, self.a2len = 6, 8
        self.latent_dim = self.vae_latent_dim + self.a1len + self.a2len
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

    def __build_models(self, mode="train"):
        # vocab_size = 11 # 我们想要将每个单词映射到的向量维度
        self.ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
        self.ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
        self.vae = ConvVAE(inp_shape=(1, 128, 64), vae_latent_dim=self.vae_latent_dim, latent_dim=self.latent_dim,
                           flat=True).to(self.device)
        self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                     num_target_class=self.class_num).to(self.device)

        # ame1 = AME(class_num=3, em_dim=self.a1len).to(device)
        # ame2 = AME(class_num=4, em_dim=self.a2len).to(device)
        # vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.vae_latent_dim, flat=True).to(device)
        # classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
        #                         num_target_class=self.class_num).to(device)

        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        self.cls_weight = 5  # orange
        self.vae_weight = 0.0001  # black

        self.align_weight = 0.0025
        self.kl_attri_weight = 0.01  # green
        self.kl_latent_weight = 0.0125  # green
        self.recon_weight = 1.2  # red

        self.recon_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(class_num=self.class_num)
        self.lr1, self.lr2, self.lr3 = 0.0004, 0.00001, 0.0005
        if mode == "train":
            self.optimizer_Em = torch.optim.Adam(
                itertools.chain(self.ame1.parameters(), self.ame2.parameters()), lr=self.lr1, betas=(0.5, 0.999))
            self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=self.lr2, betas=(0.5, 0.999))
            self.optimizer_cls = torch.optim.Adam(self.classifier.parameters(), lr=self.lr3, betas=(0.5, 0.999))

    def __build_df(self):
        self.coughvid_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_split_specattri.pkl")
        self.coughvid_df = self.coughvid_df.iloc[:, [0, 1, 2, 8, 9]]
        neg_list = list(range(2076))
        pos_list = list(range(2076, 2850))
        random.shuffle(neg_list)
        random.shuffle(pos_list)

        valid_list = neg_list[:100] + pos_list[:100]
        train_list = neg_list[100:] + pos_list[100:]
        train_df = bin_upsampling_balance(self.coughvid_df.iloc[train_list, :])
        valid_df = self.coughvid_df.iloc[valid_list, :]
        # print(train_df.head())
        print("train valid length:", train_df.shape, valid_df.shape)
        # normalize the data
        self.train_df, self.valid_df = normalize_data(train_df, valid_df)

    def __build_dataloaders(self, batch_size=32):
        self.__build_df()
        self.train_transforms = transforms.Compose([MyRightShift(input_size=(128, 64),
                                                                 width_shift_range=7,
                                                                 shift_probability=0.9),
                                                    MyAddGaussNoise(input_size=(128, 64),
                                                                    add_noise_probability=0.55),
                                                    MyReshape(output_size=(1, 128, 64))])
        self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 64))])
        train_ds = CoughvidDataset(self.train_df, transform=self.train_transforms)
        self.train_loader = DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=0)
        # init test data loader
        valid_ds = CoughvidDataset(self.valid_df, transform=self.test_transforms)
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

    def train(self):
        device = torch.device("cuda")
        self.__build_models(mode="train")
        self.__build_dataloaders(batch_size=32)

        recon_loss = nn.MSELoss()
        # categorical_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(class_num=self.class_num)

        Loss_List_Epoch = []

        x_mel = None
        x_recon = None
        epoch_id = 0

        save_dir = "./runs/ammidr/" + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + "settings.txt", 'w') as fout:
            fout.write(f"ame1 = AME(class_num=3, em_dim={self.a1len});")
            fout.write(f"ame2 = AME(class_num=4, em_dim={self.a2len})\n")
            fout.write(
                f"vae = ConvVAE(inp_shape=(1, 128, 64), vae_latent_dim={self.vae_latent_dim}, latent_dim={self.latent_dim}, flat=True)\n")
            fout.write(
                f"classifier = Classifier(dim_embedding={self.latent_dim}, dim_hidden_classifier=32, num_target_class={self.class_num})\n")
            fout.write(
                f"vae_weight={self.vae_weight}, kl_attr_weight={self.kl_attri_weight}, kl_latent_weight={self.kl_latent_weight}, recon_weight={self.recon_weight};")
            fout.write(f"cls_weight={self.cls_weight}\n")
            fout.write(f"optimizer_Em = torch.optim.Adam(self.vae.parameters(), lr={self.lr1}, betas=(0.5, 0.999))\n")
            fout.write(f"optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr={self.lr2}, betas=(0.5, 0.999))\n")
            fout.write(f"optimizer_cls = torch.optim.Adam(self.vae.parameters(), lr={self.lr3}, betas=(0.5, 0.999))\n")
        Loss_All_List = []
        for epoch_id in range(371):
            Loss_List_Total = []
            Loss_List_recon = []
            Loss_List_disen = []
            Loss_List_vae = []
            Loss_List_cls = []
            x_Recon = None
            for jdx, batch in tqdm(enumerate(self.train_loader), desc="Epoch[{}]".format(epoch_id)):
                x_mel = batch["spectrogram"].to(device)
                y_lab = batch["label"].to(device)
                ctype = batch["cough_type"].to(device)
                sevty = batch["severity"].to(device)

                self.optimizer_Em.zero_grad()
                self.optimizer_vae.zero_grad()
                self.optimizer_cls.zero_grad()

                bs = len(x_mel)
                # print("batch_size:", bs)
                # print("shape of input, x_mel y_lab attris:", x_mel.shape, y_lab.shape, ctype.shape, sevty.shape)

                mu_a_1, logvar_a_1, _ = self.ame1(ctype)  # [32, 6] [32, 6]
                mu_a_2, logvar_a_2, _ = self.ame2(sevty)  # [32, 8] [32, 8]
                x_recon, z_mu, z_logvar, _ = self.vae(x_mel, mu_a_1, logvar_a_1, mu_a_2,
                                                      logvar_a_2)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]

                # Loss_attri *= self.kl_attri_weight
                Loss_vae = self.vae_weight * vae_loss_fn(recon_x=x_recon, x=x_mel,
                                                         mean=torch.concat((z_mu, mu_a_1, mu_a_2), dim=-1),
                                                         log_var=torch.concat((z_logvar, logvar_a_1, logvar_a_2),
                                                                              dim=-1))
                # print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
                # print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
                # print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)

                Loss_akl = self.kl_latent_weight * pairwise_kl_loss(torch.concat((mu_a_1, mu_a_2), dim=-1),
                                                                    torch.concat((logvar_a_1, logvar_a_2), dim=-1), bs)
                # Loss_akl += kl_attri_weight * pairwise_kl_loss(mu_a_2, logvar_a_2, bs)
                Loss_akl += self.kl_attri_weight * pairwise_kl_loss(z_mu, z_logvar, bs)
                Loss_akl = Loss_akl.sum(-1)
                Loss_recon = recon_loss(x_recon, x_mel)
                # print("Loss recon", Loss_recon)
                Loss_recon *= self.recon_weight
                Loss_disen = Loss_akl + Loss_recon
                # print("Loss Disen", Loss_disen)

                y_pred = self.classifier(torch.concat((z_mu, mu_a_1, mu_a_2), dim=-1))  # torch.Size([32, 2])
                # Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
                Loss_cls = focal_loss(y_pred, y_lab)

                # print("shape of y_pred:", y_pred.shape, Loss_cls)
                Loss_cls *= self.cls_weight

                Loss_total = Loss_vae + Loss_disen + Loss_cls

                # print("part[1][2] cls loss:{}; recon loss:{};".format(Loss_cls, Loss_recon))
                # print("part[3] beta alpha kl loss:{};".format(Loss_akl))
                # print("part[5] ib2 (beta, alpha) c kl loss:{};".format(Loss_disen.shape))
                # print("Loss total: {}".format(Loss_total))

                Loss_total.backward()

                self.optimizer_Em.step()
                self.optimizer_vae.step()
                self.optimizer_cls.step()

                Loss_List_vae.append(Loss_vae.item())
                Loss_List_recon.append(Loss_recon.item())
                Loss_List_disen.append(Loss_disen.item())
                Loss_List_cls.append(Loss_cls.item())
                Loss_List_Total.append(Loss_total.item())

                # if jdx % 500 == 0:
                #     print("Epoch {}, Batch {}".format(0, jdx))
                #     print([np.array(Loss_List_vae).mean(),
                #            np.array(Loss_List_recon).mean(),
                #            np.array(Loss_List_disen).mean(),
                #            np.array(Loss_List_cls).mean(),
                #            np.array(Loss_List_Total).mean()])
                # if epoch_id == 0 and jdx == 0:
                #     print("pred:{}; x_Recon:{}".format(y_pred.shape, x_recon.shape))
                #     print("part[1][2] cls loss:{}; recon loss:{};".format(Loss_cls, Loss_vae))
                #     print("part[3] beta alpha kl loss:{};".format(Loss_disen))
            Loss_List_Epoch.append([np.array(Loss_List_vae).mean(),
                                    np.array(Loss_List_recon).mean(),
                                    np.array(Loss_List_disen).mean(),
                                    np.array(Loss_List_cls).mean(),
                                    np.array(Loss_List_Total).mean()])
            # if epoch_id % 9 == 0:
            #     print("Loss Parts:")
            #     print(Loss_List_Epoch)

            if epoch_id > 4 and epoch_id % 10 == 0:
                save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
                os.makedirs(save_dir_epoch, exist_ok=True)
                torch.save(self.ame1.state_dict(), save_dir_epoch + "epoch_{}_ame1.pth".format(epoch_id))
                torch.save(self.ame2.state_dict(), save_dir_epoch + "epoch_{}_ame2.pth".format(epoch_id))
                torch.save(self.vae.state_dict(), save_dir_epoch + "epoch_{}_vae.pth".format(epoch_id))
                torch.save(self.classifier.state_dict(), save_dir_epoch + "epoch_{}_cls.pth".format(epoch_id))

                torch.save(self.optimizer_Em.state_dict(), save_dir_epoch + "epoch_{}_optimizerEm".format(epoch_id))
                torch.save(self.optimizer_vae.state_dict(), save_dir_epoch + "epoch_{}_optimizerVae".format(epoch_id))
                torch.save(self.optimizer_cls.state_dict(), save_dir_epoch + "epoch_{}_optimizerCls".format(epoch_id))

            if epoch_id % 10 == 0:

                with open(save_dir + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                    fin.write("total,cls,recon,klab,ib\n")
                    for epochlist in Loss_List_Epoch:
                        fin.write(",".join([str(it) for it in epochlist]) + "\n")
                plt.figure(0)
                Loss_All_Lines = np.array(Loss_List_Epoch)
                # print(Loss_All_Lines.shape)
                #    [VAE       recon  disen    classi   total]
                cs = ["black", "red", "green", "orange", "blue"]
                for j in range(5):
                    dat_lin = Loss_All_Lines[:, j]
                    plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                plt.savefig(save_dir + f'loss_iter_{epoch_id}.png')
                plt.close(0)
            if x_recon is not None:
                plt.figure(1)
                img_to_plot = x_recon[:9].squeeze().data.cpu().numpy()
                for i in range(1, 4):
                    for j in range(1, 4):
                        plt.subplot(3, 3, (i - 1) * 3 + j)
                        plt.imshow(img_to_plot[(i - 1) * 3 + j - 1])
                plt.savefig(save_dir + "recon_epoch-{}.png".format(epoch_id), format="png")
                plt.close(1)
            else:
                raise Exception("x_Recon is None.")
        # x = torch.rand(size=(batch_size, channel, img_size, img_size))
        # label = torch.randint(0, class_num, size=(batch_size,))
        # code = torch.rand(size=(batch_size, code_dim))

    def train_demo(self):
        device = torch.device("cuda")
        # self.__build_models(mode="train")
        self.__build_dataloaders(batch_size=32)
        ame1 = AME(class_num=3, em_dim=self.a1len).to(device)
        ame2 = AME(class_num=4, em_dim=self.a2len).to(device)
        vae = ConvVAE(inp_shape=(1, 128, 64), vae_latent_dim=self.vae_latent_dim, latent_dim=self.latent_dim,
                      flat=True).to(device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(device)

        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        cls_weight = 2
        vae_weight = 0.3

        # align_weight = 0.0025
        kl_attri_weight = 0.01  # noise
        kl_latent_weight = 0.0125  # clean
        recon_weight = 0.05

        recon_loss = nn.MSELoss()
        # categorical_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(class_num=self.class_num)

        Loss_List_Epoch = []

        Loss_List_Total = []
        Loss_List_disen = []
        Loss_List_vae = []
        Loss_List_cls = []
        x_mel = None
        x_recon = None
        epoch_id = 0
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(device)
            y_lab = batch["label"].to(device)
            ctype = batch["cough_type"].to(device)
            sevty = batch["severity"].to(device)
            bs = len(x_mel)
            print("batch_size:", bs)
            print("shape of input, x_mel y_lab attris:", x_mel.shape, y_lab.shape, ctype.shape, sevty.shape)

            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            x_recon, z_mu, z_logvar, z_latent = vae(x_mel, mu_a_1, logvar_a_1, mu_a_2,
                                                    logvar_a_2)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
            # Loss_attri *= self.kl_attri_weight
            Loss_vae = 0.01 * vae_weight * vae_loss_fn(recon_x=x_recon, x=x_mel, mean=z_mu, log_var=z_logvar)
            print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
            print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
            print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)

            Loss_akl = kl_latent_weight * pairwise_kl_loss(torch.concat((mu_a_1, mu_a_2), dim=-1),
                                                           torch.concat((logvar_a_1, logvar_a_2), dim=-1), bs)
            # Loss_akl += kl_attri_weight * pairwise_kl_loss(mu_a_2, logvar_a_2, bs)
            Loss_akl += kl_attri_weight * pairwise_kl_loss(z_mu, z_logvar, bs)
            Loss_akl = Loss_akl.sum(-1)
            Loss_recon = recon_loss(x_recon, x_mel)
            print("Loss recon", Loss_recon)
            Loss_recon *= recon_weight
            Loss_disen = Loss_akl + Loss_recon
            print("Loss Disen", Loss_disen)

            y_pred = classifier(torch.concat((z_mu, mu_a_1, mu_a_2), dim=-1))  # torch.Size([32, 2])
            # Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
            Loss_cls = focal_loss(y_pred, y_lab)

            print("shape of y_pred:", y_pred.shape, Loss_cls)
            Loss_cls *= cls_weight

            Loss_total = Loss_vae + Loss_disen + Loss_cls

            print("part[1][2] cls loss:{}; recon loss:{};".format(Loss_cls, Loss_recon))
            print("part[3] beta alpha kl loss:{};".format(Loss_akl))
            print("part[5] ib2 (beta, alpha) c kl loss:{};".format(Loss_disen.shape))
            print("Loss total: {}".format(Loss_total))

            Loss_total.backward()

            Loss_List_vae.append(Loss_vae.item())
            Loss_List_disen.append(Loss_disen.item())
            Loss_List_cls.append(Loss_cls.item())
            Loss_List_Total.append(Loss_total.item())

            if jdx % 500 == 0:
                print("Epoch {}, Batch {}".format(0, jdx))
                print([np.array(Loss_List_vae).mean(),
                       np.array(Loss_List_disen).mean(),
                       np.array(Loss_List_cls).mean(),
                       np.array(Loss_List_Total).mean()])
            if 0 == 0 and jdx == 0:
                print("pred:{}; x_Recon:{}".format(y_pred.shape, x_recon.shape))
                print("part[1][2] cls loss:{}; recon loss:{};".format(Loss_cls, Loss_vae))
                print("part[3] beta alpha kl loss:{};".format(Loss_disen))
            if jdx == 10:
                break
        Loss_List_Epoch.append([np.array(Loss_List_vae).mean(),
                                np.array(Loss_List_disen).mean(),
                                np.array(Loss_List_cls).mean(),
                                np.array(Loss_List_Total).mean()])
        print("Loss Parts:")
        print(Loss_List_Epoch)
        save_dir = "./runs/ammidr/test_epoch_{}/".format(0)
        if epoch_id == 0:
            save_dir_epoch = save_dir
            os.makedirs(save_dir_epoch, exist_ok=True)
            torch.save(ame1.state_dict(), save_dir_epoch + "epoch_{}_ame1.pth".format(epoch_id))
            torch.save(ame2.state_dict(), save_dir_epoch + "epoch_{}_ame2.pth".format(epoch_id))
            torch.save(vae.state_dict(), save_dir_epoch + "epoch_{}_vae.pth".format(epoch_id))
            torch.save(classifier.state_dict(), save_dir_epoch + "epoch_{}_cls.pth".format(epoch_id))

            # torch.save(self.optimizer_E.state_dict(), save_dir_epoch + "epoch_{}_optimizerE".format(epoch_id))
            # torch.save(self.optimizer_C.state_dict(), save_dir_epoch + "epoch_{}_optimizerC".format(epoch_id))
            # torch.save(self.optimizer_G.state_dict(), save_dir_epoch + "epoch_{}_optimizerG".format(epoch_id))
            # torch.save(self.optimizer_D.state_dict(), save_dir_epoch + "epoch_{}_optimizerD".format(epoch_id))

        if epoch_id % 10 == 0:
            if epoch_id == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                with open(save_dir + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                    fin.write("total,cls,recon,klab,ib")
                    for epochlist in Loss_List_Epoch:
                        fin.write(",".join([str(it) for it in epochlist]))
                plt.figure(0)
                Loss_List_Epoch = np.array(Loss_List_Epoch)
                cs = ["red", "green", "orange", "blue"]
                for j in range(4):
                    dat_lin = Loss_List_Epoch[:, j]
                    plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                plt.savefig(save_dir + f'loss_iter_{epoch_id}.png')
                plt.close(0)
        if x_recon is not None:
            plt.figure(1)
            img_to_plot = x_recon[:9].squeeze().data.cpu().numpy()
            for i in range(1, 4):
                for j in range(1, 4):
                    plt.subplot(3, 3, (i - 1) * 3 + j)
                    plt.imshow(img_to_plot[(i - 1) * 3 + j - 1])
            plt.savefig(save_dir + "recon_epoch-{}.png".format(epoch_id), format="png")
            plt.close(1)
        else:
            raise Exception("x_Recon is None.")

    def train_cls(self, resume_dir=None, onlybeta=False):
        self.__build_dataloaders(batch_size=32)
        cls_latent_dim = 30
        ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
        ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
        vae = ConvVAE(inp_shape=(1, 128, 64), vae_latent_dim=self.vae_latent_dim, latent_dim=self.latent_dim,
                      flat=True).to(self.device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(self.device)

        AMDRcls = AMDR_CLS(vae=vae, ame1=ame1, ame2=ame2, cls=classifier)
        AMDRcls.load_state_from_path(state_dir=resume_dir, epoch_id=370)

        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
        resume_epoch = 370

        # categorical_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(class_num=self.class_num)
        Loss_List = []
        AMDRcls.to_train()
        epoch_id = None
        epoch_num = 81
        print("=======================Train on Train Set===========================")
        for epoch_id in tqdm(range(epoch_num), desc=f"Train[{epoch_id}]"):
            loss_epoch = 0.
            batch_num = 0
            for jdx, batch in enumerate(self.train_loader):
                optimizer_cls.zero_grad()
                x_mel = batch["spectrogram"].to(self.device)
                y_lab = batch["label"].to(self.device)
                ctype = batch["cough_type"].to(self.device)
                sevty = batch["severity"].to(self.device)

                y_pred = AMDRcls(x_input=x_mel, a1=ctype, a2=sevty)
                cls_loss = focal_loss(inputs=y_pred, targets=y_lab)
                cls_loss.backward()
                optimizer_cls.step()

                batch_num += 1
                loss_epoch += cls_loss.item()
            loss_avg = loss_epoch / batch_num
            Loss_List.append(loss_avg)
            if epoch_id == 0:
                print("save path: {}, epoch{}".format(resume_dir, resume_epoch))
                # os.makedirs(resume_dir + "epoch{}/".format(resume_epoch), exist_ok=True)
            elif epoch_id % 5 == 0:
                print("Epoch:", epoch_id)
                print(Loss_List)
            if epoch_id == epoch_num - 1:
                # if epoch_id == 0:
                #     os.makedirs(resume_dir + "epoch100_cls/", exist_ok=True)
                # else:
                #     torch.save(classifier.state_dict(), resume_dir + "epoch100_cls/epoch_{}_cls.pth".format(epoch_id))
                torch.save(AMDRcls.state_dict(),
                           "./runs/agedr/cls{}_ld{}_reoptim{}.pth".format(resume_epoch, cls_latent_dim, epoch_id))
        plt.figure(0)
        plt.plot(range(len(Loss_List)), Loss_List, c="black")
        plt.savefig("./runs/agedr/cls{}_ld{}_retrain{}_losslist.png".format(resume_epoch, cls_latent_dim, epoch_id),
                    dpi=300, format="png")
        plt.close(0)
        print("=======================Evaluate on Train Set===========================")
        AMDRcls.to_eval()
        # print()
        # print(y_preds.shape, y_labs.shape)
        # acc = calculate_correct(scores=y_preds, labels=y_labs)
        # print("train set, accuracy:", acc / len(self.train_loader.dataset))
        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            ctype = batch["cough_type"].to(self.device)
            sevty = batch["severity"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                y_pred = AMDRcls(x_input=x_mel, a1=ctype, a2=sevty)
                # if onlybeta:
                #     z_mu = z_mu[:, :latent_dim]
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        # print(y_labs)
        print(y_preds.shape, y_labs.shape)
        from sklearn import metrics
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print(precision, recall, acc)
        print("=======================Evaluate on Test Set===========================")
        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.valid_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            ctype = batch["cough_type"].to(self.device)
            sevty = batch["severity"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                y_pred = AMDRcls(x_input=x_mel, a1=ctype, a2=sevty)
                # if onlybeta:
                #     z_mu = z_mu[:, :latent_dim]
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        # print(y_labs)
        print(y_preds.shape, y_labs.shape)
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print(precision, recall, acc)
        # acc = calculate_correct(scores=y_preds, labels=y_labs)
        # print("valid set, accuracy:", acc / len(self.valid_loader.dataset))


if __name__ == '__main__':
    trainer = AMMIDRTrainer()
    # trainer.train()
    # trainer.demo()

    trainer.train_cls(resume_dir="./runs/ammidr/202410171554_reconwell/epoch370", onlybeta=False)

    # from cirl_libs.ResNet import resnet18
    # masker = Masker(512, 512, 4 * 512, k=308)
    # resnet = resnet18(pretrained=True)
    # resnet = ConvNet()

    # train_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT", 'train_data.npy'))
    # print(train_data.shape)

    # masked = masker(feat)[0]
    # # print(masked)
    # print(feat.shape, masked.shape)
