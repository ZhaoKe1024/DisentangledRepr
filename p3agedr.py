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
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mylibs.conv_vae import ConvVAE, kl_2normal, vae_loss_fn
from audiokits.transforms import *
from mylibs.modules import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


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
        self.latent_dim = 30
        self.a1len, self.a2len = 6, 8
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
        # self.recon_weight = 0.01
        # self.cls_weight = 1.5
        # self.vae_weight = 0.2
        # self.kl_attri_weight = 0.01  # noise
        # self.kl_latent_weight = 0.075  # clean
        # self.recon_loss = nn.MSELoss()
        # self.categorical_loss = nn.CrossEntropyLoss()

    def __build_models(self, mode="train"):
        # vocab_size = 11 # 我们想要将每个单词映射到的向量维度
        self.ame1 = AME(class_num=3, em_dim=self.a1len).to(self.device)
        self.ame2 = AME(class_num=4, em_dim=self.a2len).to(self.device)
        self.vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        self.classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                     num_target_class=self.class_num).to(self.device)
        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        self.cls_weight = 2
        self.vae_weight = 0.3

        self.align_weight = 0.0025
        self.kl_attri_weight = 0.01  # noise
        self.kl_latent_weight = 0.0125  # clean
        self.recon_weight = 0.05

        self.recon_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(class_num=self.class_num)
        if mode == "train":
            self.optimizer_Em = torch.optim.Adam(
                itertools.chain(self.ame1.parameters(), self.ame2.parameters()), lr=0.0003, betas=(0.5, 0.999))
            self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=0.0001, betas=(0.5, 0.999))
            self.optimizer_cls = torch.optim.Adam(self.classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

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

    def build_dataloaders(self, batch_size=32):
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

    def train(self, load_ckpt_path=None):
        self.build_dataloaders(batch_size=64)
        print("dataloader {}".format(len(self.train_loader)))
        self.__build_models(mode="train")
        epoch_start = 0
        save_dir = "./runs/agedr/" + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        if load_ckpt_path is not None:
            epoch_start = 300
            save_dir = load_ckpt_path
            # with open(resume_dir+f"epoch{epoch_start}/")
            self.ame1.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_ame1.pth".format(epoch_start, epoch_start)))
            self.ame2.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_ame2.pth".format(epoch_start, epoch_start)))
            self.vae.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_vae.pth".format(epoch_start, epoch_start)))
            self.classifier.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_classifier.pth".format(epoch_start, epoch_start)))
            self.optimizer_Em.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_optimizer_Em.pth".format(epoch_start, epoch_start)))
            self.optimizer_vae.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_optimizer_vae.pth".format(epoch_start, epoch_start)))
            self.optimizer_cls.load_state_dict(
                torch.load(load_ckpt_path + "epoch{}/epoch_{}_optimizer_cls.pth".format(epoch_start, epoch_start)))
            self.ame1.train()
            self.ame2.train()
            self.vae.train()
            self.classifier.train()
            epoch_start += 1
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + "setting.txt", 'w') as fout:
            fout.write(
                "self.align_weight={}\nself.recon_weight={}\nself.cls_weight={}\nself.vae_weight={}\nself.kl_attri_weight={}\nself.kl_latent_weight={}\n".format(
                    self.align_weight, self.recon_weight, self.cls_weight, self.vae_weight, self.kl_attri_weight,
                    self.kl_latent_weight))
            fout.write("self.optimizer_Em = torch.optim.Adam, lr=0.0003, betas=(0.5, 0.999)\n")
            fout.write("self.optimizer_vae = torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999)\n")
            fout.write("self.optimizer_cls = torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999))\n")
            fout.write("self.focal_loss = FocalLoss(class_num=2))\n")

        Loss_List_Epoch = []
        for epoch_id in tqdm(range(epoch_start, 371), desc="Epoch:"):
            Loss_List_Total = []
            Loss_List_disen = []
            Loss_List_attri = []
            Loss_List_vae = []
            Loss_List_cls = []
            x_mel = None
            x_recon = None
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
                # Loss_attri *= self.kl_attri_weight
                Loss_vae = 0.01 * self.vae_weight * vae_loss_fn(recon_x=x_recon, x=x_mel, mean=z_mu, log_var=z_logvar)
                # print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
                # print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
                # print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)
                # print("shape of y_pred:", y_pred.shape)

                mu1_latent = z_mu[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
                mu2_latent = z_mu[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
                lv1_latent = z_logvar[:, self.blen:self.blen + self.a1len]  # Size([32, 8])
                lv2_latent = z_logvar[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 8])
                # print("mu1:{}, lv1:{}, mu2:{}, lv2:{}".format(mu1_latent.shape, lv1_latent.shape, mu2_latent.shape,
                #                                               lv2_latent.shape))
                Loss_attri = kl_2normal(mu_a_1, logvar_a_1, mu1_latent, lv1_latent)
                Loss_attri += kl_2normal(mu_a_2, logvar_a_2, mu2_latent, lv2_latent)
                Loss_attri *= self.align_weight

                Loss_akl = self.kl_latent_weight * pairwise_kl_loss(z_mu[:, :self.blen], z_logvar[:, :self.blen], bs)
                Loss_akl += self.kl_attri_weight * pairwise_kl_loss(z_mu[:, self.blen:], z_logvar[:, self.blen:], bs)
                Loss_akl = Loss_akl.sum(-1)
                Loss_recon = self.recon_weight * self.recon_loss(x_recon, x_mel)
                Loss_disen = Loss_akl + Loss_recon

                y_pred = self.classifier(z_mu)  # torch.Size([32, 2])
                # Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
                Loss_cls = self.cls_weight * self.focal_loss(y_pred, y_lab)

                Loss_total = Loss_vae + Loss_attri + Loss_disen + Loss_cls

                Loss_total.backward()
                self.optimizer_cls.step()
                self.optimizer_vae.step()
                self.optimizer_Em.step()
                # print(L_attri.shape, L_disen.shape, L_cls.shape, L_total.shape)
                # print(L_attri, L_disen, L_cls, L_total)

                Loss_List_Total.append(Loss_total.item())
                Loss_List_disen.append(Loss_disen.item())
                Loss_List_attri.append(Loss_attri.item())
                Loss_List_vae.append(Loss_vae.item())
                Loss_List_cls.append(Loss_cls.item())
                # if jdx % 500 == 0:
                #     print("Epoch {}, Batch {}".format(epoch_id, jdx))
                #     print("Loss akl recon", Loss_akl, Loss_recon)
                #     print([np.array(Loss_List_Total).mean(),
                #            np.array(Loss_List_disen).mean(),
                #            np.array(Loss_List_attri).mean(),
                #            np.array(Loss_List_vae).mean(),
                #            np.array(Loss_List_cls).mean()])
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
                                    np.array(Loss_List_vae).mean(),
                                    np.array(Loss_List_cls).mean()])
            # print("Loss Parts:")
            ns = ["total", "disen", "attri", "vae", "cls"]
            # print([ns[j] + ":" + str(Loss_List_Epoch[-1][j]) for j in range(5)])
            # save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
            # os.makedirs(save_dir_epoch, exist_ok=True)
            # if epoch_id > 9:
            #     torch.save(self.ame1.state_dict(), save_dir_epoch + "epoch_{}_ame1.pth".format(epoch_id))
            #     torch.save(self.ame2.state_dict(), save_dir_epoch + "epoch_{}_ame2.pth".format(epoch_id))
            #     torch.save(self.vae.state_dict(), save_dir_epoch + "epoch_{}_vae.pth".format(epoch_id))
            #     torch.save(self.classifier.state_dict(), save_dir_epoch + "epoch_{}_classifier.pth".format(epoch_id))
            #
            #     torch.save(self.optimizer_Em.state_dict(), save_dir_epoch + "epoch_{}_optimizer_Em".format(epoch_id))
            #     torch.save(self.optimizer_vae.state_dict(), save_dir_epoch + "epoch_{}_optimizer_vae".format(epoch_id))
            #     torch.save(self.optimizer_cls.state_dict(), save_dir_epoch + "epoch_{}_optimizer_cls".format(epoch_id))

            if epoch_id % 10 == 0:
                if epoch_id == 0:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                else:
                    save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
                    os.makedirs(save_dir_epoch, exist_ok=True)
                    with open(save_dir_epoch + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                        fin.write("total,disen,attri,vae,cls\n")
                        for epochlist in Loss_List_Epoch:
                            fin.write(",".join([str(it) for it in epochlist]) + '\n')
                    Loss_All_Lines = np.array(Loss_List_Epoch)
                    cs = ["black", "red", "green", "orange", "blue"]
                    # ns = ["total", "disen", "attri", "vae", "cls"]
                    for j in range(5):
                        plt.figure(j)
                        dat_lin = Loss_All_Lines[:, j]
                        plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                        plt.title("Loss " + ns[j])
                        plt.savefig(save_dir_epoch + f'loss_{ns[j]}_iter_{epoch_id}.png')
                        plt.close(j)
                    torch.save(self.ame1.state_dict(), save_dir_epoch + "epoch_{}_ame1.pth".format(epoch_id))
                    torch.save(self.ame2.state_dict(), save_dir_epoch + "epoch_{}_ame2.pth".format(epoch_id))
                    torch.save(self.vae.state_dict(), save_dir_epoch + "epoch_{}_vae.pth".format(epoch_id))
                    torch.save(self.classifier.state_dict(),
                               save_dir_epoch + "epoch_{}_classifier.pth".format(epoch_id))

                    torch.save(self.optimizer_Em.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_Em.pth".format(epoch_id))
                    torch.save(self.optimizer_vae.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_vae.pth".format(epoch_id))
                    torch.save(self.optimizer_cls.state_dict(),
                               save_dir_epoch + "epoch_{}_optimizer_cls.pth".format(epoch_id))
                    with open(save_dir_epoch + "ckpt_info_{}.txt".format(epoch_id), 'w') as fin:
                        fin.write("epoch:{}\n".format(epoch_id))
                        fin.write("total,disen,attri,vae,cls\n")
                        fin.write(",".join([str(it) for it in Loss_List_Epoch[-1]]) + '\n')
                    if x_recon is not None:
                        plt.figure(1)
                        img_to_origin = x_mel[:3].squeeze().data.cpu().numpy()
                        img_to_plot = x_recon[:3].squeeze().data.cpu().numpy()
                        for i in range(1, 4):
                            plt.subplot(3, 2, (i - 1) * 2 + 1)
                            plt.imshow(img_to_origin[i - 1])

                            plt.subplot(3, 2, (i - 1) * 2 + 2)
                            plt.imshow(img_to_plot[i - 1])
                        plt.savefig(save_dir_epoch + "recon_epoch-{}.png".format(epoch_id), format="png")
                        plt.close(1)
                    else:
                        raise Exception("x_Recon is None.")

    def evaluate_cls(self, seed=89):
        # 446 0.6703296703296703 0.61 0.655
        # 12 0.717741935483871 0.89 0.77
        print("---------seed-{}-----------".format(seed))
        setup_seed(seed)
        self.build_dataloaders(batch_size=64)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(self.device)
        resume_dir = "./runs/agedr/202409061417_一层Linear/"
        # resume_epoch = 340
        resume_epoch = 370
        vae.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_vae.pth".format(resume_epoch, resume_epoch)))
        classifier.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_classifier.pth".format(
            resume_epoch, resume_epoch)))
        vae.eval()
        classifier.eval()
        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            _, z_mu, _, _ = vae(x_mel)
            y_pred = classifier(z_mu)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        print(y_preds.shape, y_labs.shape)
        from sklearn import metrics
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print("train set:", precision, recall, acc)
        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.valid_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            _, z_mu, _, _ = vae(x_mel)
            y_pred = classifier(z_mu)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        # print(y_labs)
        # print(y_preds.shape, y_labs.shape)
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.accuracy_score(y_labs, y_preds_label)
        print("test set results:", precision, recall, acc)

    def evaluate_cls_ml(self, seed=12):
        print("---------seed-{}-----------".format(seed))
        setup_seed(seed)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        resume_dir = "./runs/agedr/202409051036_二层Linear_提取特征/"
        # resume_epoch = 340
        resume_epoch = 370
        vae.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_vae.pth".format(resume_epoch, resume_epoch)))
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(self.device)
        classifier.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_classifier.pth".format(
            resume_epoch, resume_epoch)))
        from sklearn import svm
        from sklearn.metrics import precision_score, recall_score, roc_auc_score
        self.build_dataloaders(batch_size=64)
        vae.eval()
        classifier.eval()
        feats_tr = None
        y_labs = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            _, z_mu, _, _ = vae(x_mel)
            _, z_mu = classifier(z_mu, fe=True)
            if feats_tr is None:
                feats_tr = z_mu
            else:
                feats_tr = torch.concat((feats_tr, z_mu), dim=0)
        feats_va = None
        y_labs_va = None
        for jdx, batch in enumerate(self.valid_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs_va is None:
                y_labs_va = y_lab
            else:
                y_labs_va = torch.concat((y_labs_va, y_lab), dim=0)
            _, z_mu, _, _ = vae(x_mel)
            _, z_mu = classifier(z_mu, fe=True)
            if feats_va is None:
                feats_va = z_mu
            else:
                feats_va = torch.concat((feats_va, z_mu), dim=0)
        print(feats_tr.shape, y_labs.shape)
        print(feats_va.shape, y_labs_va.shape)

        # print(self.train_df.head)

        svm_model = svm.SVC(kernel='rbf', gamma='auto')
        svm_data_tr = feats_tr.data.cpu().numpy()
        svm_lab_tr = y_labs.data.cpu().numpy()
        svm_data_va = feats_tr.data.cpu().numpy()
        svm_lab_va = y_labs.data.cpu().numpy()
        svm_model.fit(svm_data_tr, svm_lab_tr)
        y_pref_tr = svm_model.predict(svm_data_tr)
        y_pref_te = svm_model.predict(svm_data_va)
        print("train precision:", precision_score(svm_lab_tr, y_pref_tr))
        print("test precision:", precision_score(svm_lab_va, y_pref_te))
        print("train recall:", recall_score(svm_lab_tr, y_pref_tr))
        print("test recall:", recall_score(svm_lab_va, y_pref_te))
        print("train acc:", roc_auc_score(svm_lab_tr, y_pref_tr))
        print("test acc:", roc_auc_score(svm_lab_va, y_pref_te))

    def evaluate_tsne(self):
        setup_seed(12)
        self.build_dataloaders(batch_size=32)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(self.device)
        resume_epoch = 370
        resume_dir = "./runs/agedr/202409051036_二层Linear_提取特征/".format(resume_epoch)
        vae.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_vae.pth".format(resume_epoch, resume_epoch)))
        classifier.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_classifier.pth".format(
            resume_epoch, resume_epoch)))
        vae.eval()
        classifier.eval()
        # a1_latents = None
        a1_labels = None
        # a2_latents = None
        a2_labels = None
        z_latents = None
        z_labels = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            ctype = batch["cough_type"].to(self.device)
            sevty = batch["severity"].to(self.device)
            x_recon, z_mu, z_logvar, z_latent = vae(x_mel)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
            # y_pred = classifier(z_latent)  # torch.Size([32, 2])
            mu1_latent = z_mu[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
            mu2_latent = z_mu[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
            _, z_mu = classifier(z_mu, fe=True)
            if z_latents is None:
                z_latents = z_mu
                z_labels = y_lab
            else:
                z_latents = torch.concat((z_latents, z_mu), dim=0)
                z_labels = torch.concat((z_labels, y_lab), dim=0)
            if a1_labels is None:
                # a1_latents = mu1_latent
                a1_labels = ctype
            else:
                # a1_latents = torch.concat((a1_latents, mu1_latent), dim=0)
                a1_labels = torch.concat((a1_labels, ctype), dim=0)
            if a2_labels is None:
                # a2_latents = mu2_latent
                a2_labels = sevty
            else:
                # a2_latents = torch.concat((a2_latents, mu2_latent), dim=0)
                a2_labels = torch.concat((a2_labels, sevty), dim=0)
        tsne_z_input = z_latents.data.cpu().numpy()
        # tsne_a1_input = a1_latents.data.cpu().numpy()
        # tsne_a2_input = a2_latents.data.cpu().numpy()
        print("tnse a1 shape:", tsne_z_input.shape)
        print("labels attributes shape:", z_labels.shape, a1_labels.shape, a2_labels.shape)
        # print("tnse a1 shape:", tsne_a1_input.shape)
        # print("tsne a2 shape:", tsne_a2_input.shape)

        from sklearn.manifold import TSNE
        from mylibs.figurekits import plot_embedding_2D
        # int2type = {0: "dry", 1: "wet", 2: "unknown"}
        # int2seve = {0: "mild", 1: "pseudocough", 2: "severe", 3: "unknown"}
        tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
        result2D = tsne_model.fit_transform(tsne_z_input)
        form = "svg"
        plot_embedding_2D(result2D, z_labels, "t-SNT for healthy or covid-19",
                          savepath=resume_dir + "epoch{}/tsnev_cls_label_{}.{}".format(resume_epoch, resume_epoch,
                                                                                       form),
                          names=["healthy", "covid19"], params={"format": form})
        # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
        # result2D = tsne_model.fit_transform(tsne_a1_input)
        plot_embedding_2D(result2D, a1_labels, "t-SNT for cough_type",
                          savepath=resume_dir + "epoch{}/tsnev_cls_coughtype_{}.{}".format(resume_epoch, resume_epoch,
                                                                                           form),
                          names=["dry", "wet", "unknown"], params={"format": form})
        # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
        # result2D = tsne_model.fit_transform(tsne_a1_input)
        plot_embedding_2D(result2D, a2_labels, "t-SNT for severity",
                          savepath=resume_dir + "epoch{}/epochv_cls_severity_{}.{}".format(resume_epoch, resume_epoch,
                                                                                           form),
                          names=["mild", "pseudocough", "severe", "unknown"], params={"format": form})
        print("TSNE finish.")

    def demo(self):
        device = torch.device("cuda")
        # self.__build_models(mode="train")
        self.build_dataloaders(batch_size=32)
        ame1 = AME(class_num=3, em_dim=self.a1len).to(device)
        ame2 = AME(class_num=4, em_dim=self.a2len).to(device)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(device)
        classifier = Classifier(dim_embedding=self.latent_dim, dim_hidden_classifier=32,
                                num_target_class=self.class_num).to(device)

        # self.classifier = nn.Linear(in_features=self.latent_dim, out_features=self.class_num).to(self.device)
        cls_weight = 2
        vae_weight = 0.3

        align_weight = 0.0025
        kl_attri_weight = 0.01  # noise
        kl_latent_weight = 0.0125  # clean
        recon_weight = 0.05

        recon_loss = nn.MSELoss()
        categorical_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(class_num=self.class_num)

        Loss_List_Total = []
        Loss_List_disen = []
        Loss_List_attri = []
        Loss_List_vae = []
        Loss_List_cls = []
        x_mel = None
        x_recon = None

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
            x_recon, z_mu, z_logvar, z_latent = vae(x_mel)  # [32, 1, 64, 128] [32, 30] [32, 30] [32, 30]
            # Loss_attri *= self.kl_attri_weight
            Loss_vae = 0.01 * vae_weight * vae_loss_fn(recon_x=x_recon, x=x_mel, mean=z_mu, log_var=z_logvar)
            print("shape of attri1 latent:", mu_a_1.shape, logvar_a_1.shape)
            print("shape of attri2 latent:", mu_a_2.shape, logvar_a_2.shape)
            print("shape of vae output:", x_recon.shape, z_mu.shape, z_logvar.shape, z_latent.shape)

            mu1_latent = z_mu[:, self.blen:self.blen + self.a1len]  # Size([32, 6])
            mu2_latent = z_mu[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 6])
            lv1_latent = z_logvar[:, self.blen:self.blen + self.a1len]  # Size([32, 8])
            lv2_latent = z_logvar[:, self.blen + self.a1len:self.blen + self.a1len + self.a2len]  # Size([32, 8])
            print("mu1:{}, lv1:{}, mu2:{}, lv2:{}".format(mu1_latent.shape, lv1_latent.shape, mu2_latent.shape,
                                                          lv2_latent.shape))
            Loss_attri = kl_2normal(mu_a_1, logvar_a_1, mu1_latent, lv1_latent)
            Loss_attri += kl_2normal(mu_a_2, logvar_a_2, mu2_latent, lv2_latent)
            print("Loss_attri:", Loss_attri)
            Loss_attri *= align_weight

            Loss_akl = kl_latent_weight * pairwise_kl_loss(z_mu[:, :self.blen], z_logvar[:, :self.blen], bs)
            Loss_akl += kl_attri_weight * pairwise_kl_loss(z_mu[:, self.blen:], z_logvar[:, self.blen:], bs)
            Loss_akl = Loss_akl.sum(-1)
            Loss_recon = recon_loss(x_recon, x_mel)
            print("Loss recon", Loss_recon)
            Loss_recon *= recon_weight
            Loss_disen = Loss_akl + Loss_recon
            print("Loss Disen", Loss_disen)
            y_pred = classifier(z_mu)  # torch.Size([32, 2])
            # Loss_cls = self.cls_weight * self.categorical_loss(y_pred, y_lab)
            Loss_cls = focal_loss(y_pred, y_lab)

            print("shape of y_pred:", y_pred.shape, Loss_cls)
            Loss_cls *= cls_weight

            Loss_total = Loss_vae + Loss_attri + Loss_disen + Loss_cls

            # print(L_attri.shape, L_disen.shape, L_cls.shape, L_total.shape)
            # print(L_attri, L_disen, L_cls, L_total)

            Loss_List_Total.append(Loss_total.item())
            Loss_List_disen.append(Loss_disen.item())
            Loss_List_attri.append(Loss_attri.item())
            Loss_List_vae.append(Loss_vae.item())
            Loss_List_cls.append(Loss_cls.item())
            # if jdx % 500 == 0:
            #     print("Epoch {}, Batch {}".format(epoch_id, jdx))
            #     print("Loss akl recon", Loss_akl, Loss_recon)
            #     print([np.array(Loss_List_Total).mean(),
            #            np.array(Loss_List_disen).mean(),
            #            np.array(Loss_List_attri).mean(),
            #            np.array(Loss_List_vae).mean(),
            #            np.array(Loss_List_cls).mean()])
            if jdx == 0:
                print("z_h:{}, a1.shape:{}, a2.sahpe:{}".format(z_latent.shape, mu1_latent.shape, mu2_latent.shape))
                print("pred:{}; x_Recon:{}".format(y_pred.shape, x_recon.shape))
                print("part[cls] cls loss:{};".format(Loss_cls))
                print("part[disen] beta alpha kl loss:{};".format(Loss_disen))

                print("part[attri] pdf loss:{};".format(Loss_attri))
                print("part[recon] recon loss:{};".format(Loss_recon))
            return
        # Loss_List_Epoch.append([np.array(Loss_List_Total).mean(),
        #                         np.array(Loss_List_disen).mean(),
        #                         np.array(Loss_List_attri).mean(),
        #                         np.array(Loss_List_vae).mean(),
        #                         np.array(Loss_List_cls).mean()])
        # # print("Loss Parts:")
        # ns = ["total", "disen", "attri", "vae", "cls"]
        # print([ns[j] + ":" + str(Loss_List_Epoch[-1][j]) for j in range(5)])
        # save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
        # os.makedirs(save_dir_epoch, exist_ok=True)

    def train_cls(self, latent_dim, onlybeta=False, seed=12, vaepath=None):
        setup_seed(seed)
        self.build_dataloaders(batch_size=32)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        vae.eval()
        classifier = Classifier(dim_embedding=latent_dim, dim_hidden_classifier=16,
                                num_target_class=self.class_num).to(self.device)
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
        resume_dir = vaepath
        resume_epoch = 370
        vae.load_state_dict(torch.load(resume_dir + "epoch{}/epoch_{}_vae.pth".format(resume_epoch, resume_epoch)))
        # categorical_loss = nn.CrossEntropyLoss()
        focal_loss = FocalLoss(class_num=self.class_num)
        Loss_List = []
        classifier.train()
        epoch_id = None
        epoch_num = 81
        for epoch_id in tqdm(range(epoch_num)):
            loss_epoch = 0.
            batch_num = 0
            for jdx, batch in enumerate(self.train_loader):
                optimizer_cls.zero_grad()
                x_mel = batch["spectrogram"].to(self.device)
                y_lab = batch["label"].to(self.device)
                with torch.no_grad():
                    _, z_mu, _, _ = vae(x_mel)
                    if onlybeta:
                        z_mu = z_mu[:, :latent_dim]
                y_pred = classifier(z_mu)
                cls_loss = focal_loss(inputs=y_pred, targets=y_lab)
                cls_loss.backward()
                optimizer_cls.step()
                batch_num += 1
                loss_epoch += cls_loss.item()
            loss_avg = loss_epoch / batch_num
            Loss_List.append(loss_avg)
            if epoch_id == 0:
                print("save path: ./runs/agedr/")
                # os.makedirs(resume_dir + "epoch{}/".format(resume_epoch), exist_ok=True)
            elif epoch_id % 5 == 0:
                print("Epoch:", epoch_id)
                print(Loss_List)
            if epoch_id == epoch_num - 1:
                # if epoch_id == 0:
                #     os.makedirs(resume_dir + "epoch100_cls/", exist_ok=True)
                # else:
                #     torch.save(classifier.state_dict(), resume_dir + "epoch100_cls/epoch_{}_cls.pth".format(epoch_id))
                torch.save(classifier.state_dict(),
                           "./runs/agedr/cls_vae{}_ld{}_retrain{}.pth".format(resume_epoch, latent_dim, epoch_id))
                torch.save(optimizer_cls.state_dict(),
                           "./runs/agedr/cls_vae{}_ld{}_reoptim{}.pth".format(resume_epoch, latent_dim, epoch_id))
        plt.figure(0)
        plt.plot(range(len(Loss_List)), Loss_List, c="black")
        plt.savefig("./runs/agedr/cls_vae{}_ld{}_retrain{}_losslist.png".format(resume_epoch, latent_dim, epoch_id),
                    dpi=300, format="png")
        plt.close(0)

        classifier.eval()
        # print()
        # print(y_preds.shape, y_labs.shape)
        # acc = calculate_correct(scores=y_preds, labels=y_labs)
        # print("train set, accuracy:", acc / len(self.train_loader.dataset))
        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                _, z_mu, _, _ = vae(x_mel)
                if onlybeta:
                    z_mu = z_mu[:, :latent_dim]
                y_pred = classifier(z_mu)
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

        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.valid_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                _, z_mu, _, _ = vae(x_mel)
                if onlybeta:
                    z_mu = z_mu[:, :latent_dim]
                y_pred = classifier(z_mu)
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

    def evaluate_retrain_cls(self, latent_dim, onlybeta=False, vaepath=None, clspath=None):
        setup_seed(12)
        self.build_dataloaders(batch_size=32)
        vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=self.latent_dim, flat=True).to(self.device)
        classifier = Classifier(dim_embedding=latent_dim, dim_hidden_classifier=16,
                                num_target_class=self.class_num).to(self.device)
        vae.load_state_dict(torch.load(vaepath))
        vae.eval()
        classifier.load_state_dict(torch.load(clspath))
        classifier.eval()

        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.train_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                _, z_mu, _, _ = vae(x_mel)
                if onlybeta:
                    z_mu = z_mu[:, :latent_dim]
                y_pred = classifier(z_mu)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        # print(y_labs)
        from sklearn import metrics
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.roc_auc_score(y_labs, y_preds_label)
        print("trainset results:", precision, recall, acc)

        y_preds = None
        y_labs = None
        for jdx, batch in enumerate(self.valid_loader):
            x_mel = batch["spectrogram"].to(self.device)
            y_lab = batch["label"].to(self.device)
            if y_labs is None:
                y_labs = y_lab
            else:
                y_labs = torch.concat((y_labs, y_lab), dim=0)
            # ctype = batch["cough_type"].to(self.device)
            # sevty = batch["severity"].to(self.device)
            # bs = len(x_mel)
            with torch.no_grad():
                _, z_mu, _, _ = vae(x_mel)
                if onlybeta:
                    z_mu = z_mu[:, :latent_dim]
                y_pred = classifier(z_mu)
            if y_preds is None:
                y_preds = y_pred
            else:
                y_preds = torch.concat((y_preds, y_pred), dim=0)
        y_labs = y_labs.data.cpu().numpy()
        y_preds = y_preds.data.cpu().numpy()
        y_preds_label = y_preds.argmax(-1)
        precision = metrics.precision_score(y_labs, y_preds_label)
        recall = metrics.recall_score(y_labs, y_preds_label)
        acc = metrics.roc_auc_score(y_labs, y_preds_label)
        print("validset results:", precision, recall, acc)


if __name__ == '__main__':
    agedr = AGEDRTrainer()
    agedr.evaluate_cls_ml(seed=12)
    # agedr.demo()
    # agedr.train()
    agedr.evaluate_cls(seed=12)
    # agedr.evaluate_tsne()
    # agedr.train_cls(latent_dim=30, onlybeta=False, seed=89, vaepath="./runs/agedr/202409061417_一层Linear/")
    # agedr.train_cls(latent_dim=16, onlybeta=True, seed=89, vaepath="./runs/agedr/202409061417_一层Linear/")
    # agedr.train(load_ckpt_path="./runs/agedr/202409041841/")
    # agedr.evaluate_retrain_cls(latent_dim=30, onlybeta=False,
    #                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
    #                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld30_retrain30.pth")
    # agedr.evaluate_retrain_cls(latent_dim=16, onlybeta=True,
    #                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
    #                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld16_retrain80.pth")
    # agedr.evaluate_retrain_cls(latent_dim=30, onlybeta=False,
    #                            vaepath="./runs/agedr/202409042044_一层Linear_分类失败/epoch370/epoch_370_vae.pth",
    #                            clspath="./runs/agedr/202409042044_一层Linear_分类失败_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld30_retrain30.pth")
    # agedr.evaluate_retrain_cls(latent_dim=16, onlybeta=True,
    #                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
    #                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld16_retrain80.pth")
