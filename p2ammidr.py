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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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


class AMMIDRTrainer(object):
    def __init__(self):
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.img_size, self.channel = 32, 1
        self.class_num, self.batch_size = 10, 16
        self.latent_dim, self.code_dim = 36, 4
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

    def __build_dataloaders(self):
        self.transform = transforms.Compose([transforms.Resize([self.img_size, self.img_size])])
        self.train_loader, self.test_loader, self.test_55_loader, self.test_65_loader = get_datasets(
            dataset="mnist-rot", train_batch_size=self.batch_size,
            test_batch_size=self.batch_size, cuda=True, root="F:/DATAS/mnist/MNIST-ROT", transform=self.transform)

    def __build_models(self):
        # vocab_size = 11 # 我们想要将每个单词映射到的向量维度
        embedding_dim = 4  # 创建一个Embedding层
        self.embedding = nn.Embedding(num_embeddings=self.class_num, embedding_dim=embedding_dim)
        self.encoder = EncoderMNIST(nz=self.latent_dim)
        self.generator = Generator(img_size=self.img_size, latent_dim=self.latent_dim, class_dim=embedding_dim,
                                   code_dim=self.code_dim,
                                   channels=self.channel)
        self.classifier = VIB_Classifier(dim_embedding=self.latent_dim + embedding_dim, dim_hidden_classifier=32,
                                         num_target_class=self.class_num)
        self.discriminator = Discriminator(img_size=self.img_size, channels=self.channel, n_classes=self.class_num,
                                           latent_dim=self.code_dim)

        self.optimizer_E = torch.optim.Adam(itertools.chain(self.embedding.parameters(), self.encoder.parameters()),
                                            lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_C = torch.optim.Adam(self.classifier.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.adversarial_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.MSELoss()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def __to_cuda(self):
        self.embedding.to(self.device)
        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)

        self.adversarial_loss.to(self.device)
        self.categorical_loss.to(self.device)
        self.continuous_loss.to(self.device)

    def train(self):
        cuda = True if torch.cuda.is_available() else False
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        data_root = "F:/DATAS/mnist/MNIST-ROT"
        train_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT", 'train_data.npy'))
        train_data = train_data.reshape(-1, 1, 28, 28)
        train_labels = np.load(os.path.join(data_root, 'train_labels.npy'))
        train_sensitive_labels = np.load(os.path.join(data_root, 'train_sensitive_labels.npy'))
        print(train_data.shape, train_labels.shape, train_sensitive_labels.shape)
        train_dataset = MyDataset(train_data, train_labels, train_sensitive_labels,
                                  transforms.Compose([transforms.Resize([self.img_size, self.img_size])]))
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        print("Dataset {}, loader {}".format(len(train_dataset), len(train_loader)))
        self.__build_models()
        self.__to_cuda()
        recon_weight = 0.01
        cls_weight = 1.
        kl_beta_alpha_weight = 0.01
        kl_c_weight = 0.075
        save_dir = "./runs/ammidr/" + time.strftime("%Y%m%d%H%M", time.localtime()) + '/'
        Loss_All_List = []
        for epoch_id in range(100):
            Loss_Total_Epoch = []
            Loss_List_cls = []
            Loss_List_recon = []
            Loss_List_klab = []
            Loss_List_ibcls = []
            x_Recon = None
            for jdx, (x_img, y_lab, a_sen) in tqdm(enumerate(train_loader), desc="Epoch[{}]".format(0)):
                # print("Batch[{}]".format(jdx))
                # print(x_img.shape, y_lab.shape, a_sen.shape)
                x_img = x_img.to(self.device)
                y_lab = y_lab.to(torch.long).to(self.device)
                a_sen = a_sen.to(torch.long).to(self.device)
                self.optimizer_E.zero_grad()
                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_C.zero_grad()

                z_mu, z_lv, z_logpi, z_gamma = self.encoder(x=x_img)
                z_h = self.reparameterize(mu=z_mu, logvar=z_lv)
                bs = z_h.shape[0]
                # the MI between the x and z
                pairwise_kl_loss_b = pairwise_zc_kl_loss(z_mu, z_lv, z_gamma, batch_size=bs)

                added_noise = torch.rand(size=(bs, self.code_dim), device=self.device)
                attri_vec = self.embedding(a_sen.to(torch.long))

                # MI beta alpha, the MI between the different parts of a latent vector.
                x_Recon = self.generator(noise=z_h, labels=attri_vec, code=added_noise)
                _, pred_label, pred_code = self.discriminator(x_Recon)
                kl_beta_alpha = (self.lambda_cat * self.categorical_loss(pred_label, y_lab)
                                 + self.lambda_con * self.continuous_loss(pred_code, attri_vec))

                pred = self.classifier(torch.concat((z_h, attri_vec), dim=-1))
                L_recon = self.adversarial_loss(x_Recon, x_img)
                L_cls = self.categorical_loss(pred, y_lab)
                ib_cls_kl_loss = kl_c_weight * pairwise_kl_loss_b.mean(-1) + L_cls
                Loss_total = (cls_weight * L_cls
                              + recon_weight * L_recon
                              + kl_beta_alpha_weight * kl_beta_alpha
                              + ib_cls_kl_loss)

                Loss_total.backward()
                self.optimizer_C.step()
                self.optimizer_D.step()
                self.optimizer_G.step()
                self.optimizer_E.step()
                # print("Loss total: {}".format(Loss_total))
                # optimizer.step()
                Loss_Total_Epoch.append(Loss_total.item())
                Loss_List_recon.append(L_recon.item())
                Loss_List_cls.append(L_cls.item())
                Loss_List_klab.append(kl_beta_alpha.item())
                Loss_List_ibcls.append(ib_cls_kl_loss.item())
                if jdx % 500 == 0:
                    print("Epoch {}, Batch {}".format(epoch_id, jdx))
                    print([np.array(Loss_Total_Epoch).mean(),
                           np.array(Loss_List_cls).mean(),
                           np.array(Loss_List_recon).mean(),
                           np.array(Loss_List_klab).mean(),
                           np.array(Loss_List_ibcls).mean()])
                if epoch_id == 0 and jdx == 0:
                    print("KL(Zc||x):{}".format(pairwise_kl_loss_b.shape))
                    print("z_h:{}, label_vec:{}, code:{}".format(z_h.shape, attri_vec.shape, added_noise.shape))
                    print("pred:{}; x_Recon:{}".format(pred.shape, x_Recon.shape))
                    print("part[1][2] cls loss:{}; recon loss:{};".format(L_cls, L_recon))
                    print("part[3] beta alpha kl loss:{};".format(kl_beta_alpha))

                    print("part[4] ib1 x beta loss:{};".format(pairwise_kl_loss_b.shape))
                    print("part[5] ib2 (beta, alpha) c kl loss:{};".format(L_cls))
            Loss_All_List.append([np.array(Loss_Total_Epoch).mean(),
                                  np.array(Loss_List_cls).mean(),
                                  np.array(Loss_List_recon).mean(),
                                  np.array(Loss_List_klab).mean(),
                                  np.array(Loss_List_ibcls).mean()])
            print("Loss Parts:")
            print(Loss_All_List)
            if epoch_id > 4:
                save_dir_epoch = save_dir + "epoch{}/".format(epoch_id)
                os.makedirs(save_dir_epoch, exist_ok=True)
                torch.save(self.embedding.state_dict(), save_dir_epoch + "epoch_{}_embedding.pth".format(epoch_id))
                torch.save(self.encoder.state_dict(), save_dir_epoch + "epoch_{}_encoder.pth".format(epoch_id))
                torch.save(self.generator.state_dict(), save_dir_epoch + "epoch_{}_generator.pth".format(epoch_id))
                torch.save(self.classifier.state_dict(), save_dir_epoch + "epoch_{}_classifier.pth".format(epoch_id))
                torch.save(self.discriminator.state_dict(),
                           save_dir_epoch + "epoch_{}_discriminator.pth".format(epoch_id))

                torch.save(self.optimizer_E.state_dict(), save_dir_epoch + "epoch_{}_optimizerE".format(epoch_id))
                torch.save(self.optimizer_C.state_dict(), save_dir_epoch + "epoch_{}_optimizerC".format(epoch_id))
                torch.save(self.optimizer_G.state_dict(), save_dir_epoch + "epoch_{}_optimizerG".format(epoch_id))
                torch.save(self.optimizer_D.state_dict(), save_dir_epoch + "epoch_{}_optimizerD".format(epoch_id))

            if epoch_id % 10 == 0:
                if epoch_id == 0:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                else:
                    with open(save_dir + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                        fin.write("total,cls,recon,klab,ib")
                        for epochlist in Loss_All_List:
                            fin.write(",".join([str(it) for it in epochlist]))
                    plt.figure(0)
                    Loss_All_Lines = np.array(Loss_All_List)
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
                        plt.subplot(3, 3, (i-1) * 3 + j)
                        plt.imshow(img_to_plot[(i-1) * 3 + j-1])
                plt.savefig(save_dir + "recon_epoch-{}.png".format(epoch_id), format="png")
                plt.close(1)
            else:
                raise Exception("x_Recon is None.")
        # x = torch.rand(size=(batch_size, channel, img_size, img_size))
        # label = torch.randint(0, class_num, size=(batch_size,))
        # code = torch.rand(size=(batch_size, code_dim))

    def demo(self):
        data_root = "F:/DATAS/mnist/MNIST-ROT"
        train_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT", 'train_data.npy'))
        train_data = train_data.reshape(-1, 1, 28, 28)
        train_labels = np.load(os.path.join(data_root, 'train_labels.npy'))
        train_sensitive_labels = np.load(os.path.join(data_root, 'train_sensitive_labels.npy'))
        print(train_data.shape, train_labels.shape, train_sensitive_labels.shape)
        train_dataset = MyDataset(train_data, train_labels, train_sensitive_labels,
                                  transforms.Compose([transforms.Resize([self.img_size, self.img_size])]))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.__build_models()
        recon_weight = 0.01
        cls_weight = 1.
        kl_beta_alpha_weight = 0.01
        kl_c_weight = 0.075
        # kl_a_weight = 0.01
        epoch_id = 0
        Loss_All_List = []
        Loss_Total_Epoch = []
        Loss_List_cls = []
        Loss_List_recon = []
        Loss_List_klab = []
        Loss_List_ibcls = []
        x_Recon = None
        for jdx, (x_img, y_lab, a_sen) in tqdm(enumerate(train_loader), desc="Epoch[{}]".format(0)):
            print("Batch[{}]".format(jdx))
            print(x_img.shape, y_lab.shape, a_sen.shape)
            y_lab = y_lab.to(torch.long)
            a_sen = a_sen.to(torch.long)
            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()
            self.optimizer_D.zero_grad()
            self.optimizer_C.zero_grad()

            z_mu, z_lv, z_logpi, z_gamma = self.encoder(x=x_img)
            z_h = self.reparameterize(mu=z_mu, logvar=z_lv)
            bs = z_h.shape[0]
            # the MI between the x and z
            pairwise_kl_loss_b = pairwise_zc_kl_loss(z_mu, z_lv, z_gamma, batch_size=bs)
            print("KL(Zc||x):{}".format(pairwise_kl_loss_b.shape))

            added_noise = torch.rand(size=(bs, self.code_dim))
            attri_vec = self.embedding(a_sen.to(torch.long))
            print("z_h:{}, label_vec:{}, code:{}".format(z_h.shape, attri_vec.shape, added_noise.shape))

            # MI beta alpha, the MI between the different parts of a latent vector.
            x_Recon = self.generator(noise=z_h, labels=attri_vec, code=added_noise)

            _, pred_label, pred_code = self.discriminator(x_Recon)
            kl_beta_alpha = self.lambda_cat * self.categorical_loss(pred_label,
                                                                    y_lab) + self.lambda_con * self.continuous_loss(
                pred_code, attri_vec)

            pred = self.classifier(torch.concat((z_h, attri_vec), dim=-1))

            print("pred:{}; x_Recon:{}".format(pred.shape, x_Recon.shape))

            L_recon = self.adversarial_loss(x_Recon, x_img)
            L_cls = self.categorical_loss(pred, y_lab)
            print("part[1][2] cls loss:{}; recon loss:{};".format(L_cls, L_recon))
            print("part[3] beta alpha kl loss:{};".format(kl_beta_alpha))

            print("part[4] ib1 x beta loss:{};".format(pairwise_kl_loss_b.shape))
            print("part[5] ib2 (beta, alpha) c kl loss:{};".format(pred.shape))
            ib_cls_kl_loss = kl_c_weight * pairwise_kl_loss_b.mean(-1) + L_cls
            Loss_total = (cls_weight * L_cls
                          + recon_weight * L_recon
                          + kl_beta_alpha_weight * kl_beta_alpha
                          + ib_cls_kl_loss)
            Loss_total.backward()
            self.optimizer_C.step()
            self.optimizer_D.step()
            self.optimizer_G.step()
            self.optimizer_E.step()
            print("Loss total: {}".format(Loss_total))
            Loss_Total_Epoch.append(Loss_total.item())
            Loss_List_recon.append(L_recon.item())
            Loss_List_cls.append(L_cls.item())
            Loss_List_klab.append(kl_beta_alpha.item())
            Loss_List_ibcls.append(ib_cls_kl_loss.item())
            if jdx % 500 == 0:
                print("Epoch {}, Batch {}".format(0, jdx))
                print([np.array(Loss_Total_Epoch).mean(),
                       np.array(Loss_List_cls).mean(),
                       np.array(Loss_List_recon).mean(),
                       np.array(Loss_List_klab).mean(),
                       np.array(Loss_List_ibcls).mean()])
            if 0 == 0 and jdx == 0:
                print("KL(Zc||x):{}".format(pairwise_kl_loss_b.shape))
                print("z_h:{}, label_vec:{}, code:{}".format(z_h.shape, attri_vec.shape, added_noise.shape))
                print("pred:{}; x_Recon:{}".format(pred.shape, x_Recon.shape))
                print("part[1][2] cls loss:{}; recon loss:{};".format(L_cls, L_recon))
                print("part[3] beta alpha kl loss:{};".format(kl_beta_alpha))

                print("part[4] ib1 x beta loss:{};".format(pairwise_kl_loss_b.shape))
                print("part[5] ib2 (beta, alpha) c kl loss:{};".format(L_cls))
            if jdx == 10:
                break
        Loss_All_List.append([np.array(Loss_Total_Epoch).mean(),
                              np.array(Loss_List_cls).mean(),
                              np.array(Loss_List_recon).mean(),
                              np.array(Loss_List_klab).mean(),
                              np.array(Loss_List_ibcls).mean()])
        print("Loss Parts:")
        print(Loss_All_List)
        save_dir = "./runs/ammidr/test_epoch_{}/".format(epoch_id)
        if epoch_id > 4:
            save_dir_epoch = save_dir
            os.makedirs(save_dir_epoch, exist_ok=True)
            torch.save(self.embedding.state_dict(), save_dir_epoch + "epoch_{}_embedding.pth".format(epoch_id))
            torch.save(self.encoder.state_dict(), save_dir_epoch + "epoch_{}_encoder.pth".format(epoch_id))
            torch.save(self.generator.state_dict(), save_dir_epoch + "epoch_{}_generator.pth".format(epoch_id))
            torch.save(self.classifier.state_dict(), save_dir_epoch + "epoch_{}_classifier.pth".format(epoch_id))
            torch.save(self.discriminator.state_dict(),
                       save_dir_epoch + "epoch_{}_discriminator.pth".format(epoch_id))

            torch.save(self.optimizer_E.state_dict(), save_dir_epoch + "epoch_{}_optimizerE".format(epoch_id))
            torch.save(self.optimizer_C.state_dict(), save_dir_epoch + "epoch_{}_optimizerC".format(epoch_id))
            torch.save(self.optimizer_G.state_dict(), save_dir_epoch + "epoch_{}_optimizerG".format(epoch_id))
            torch.save(self.optimizer_D.state_dict(), save_dir_epoch + "epoch_{}_optimizerD".format(epoch_id))

        if epoch_id % 10 == 0:
            if epoch_id == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                with open(save_dir + "losslist_epoch_{}.csv".format(epoch_id), 'w') as fin:
                    fin.write("total,cls,recon,klab,ib")
                    for epochlist in Loss_All_List:
                        fin.write(",".join([str(it) for it in epochlist]))
                plt.figure(0)
                Loss_All_List = np.array(Loss_All_List)
                cs = ["black", "red", "green", "orange", "blue"]
                for j in range(5):
                    dat_lin = Loss_All_List[:, j]
                    plt.plot(range(len(dat_lin)), dat_lin, c=cs[j], alpha=0.7)
                plt.savefig(save_dir + f'loss_iter_{epoch_id}.png')
                plt.close(0)
        if x_Recon is not None:
            plt.figure(1)
            img_to_plot = x_Recon[:9].squeeze().data.cpu().numpy()
            for i in range(1, 4):
                for j in range(1, 4):
                    plt.subplot(3, 3, (i-1) * 3 + j)
                    plt.imshow(img_to_plot[(i-1) * 3 + j-1])
            plt.savefig(save_dir + "recon_epoch-{}.png".format(epoch_id), format="png")
            plt.close(1)
        else:
            raise Exception("x_Recon is None.")



if __name__ == '__main__':
    trainer = AMMIDRTrainer()
    trainer.train()
    # trainer.demo()

    # from cirl_libs.ResNet import resnet18
    # masker = Masker(512, 512, 4 * 512, k=308)
    # resnet = resnet18(pretrained=True)
    # resnet = ConvNet()

    # train_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT", 'train_data.npy'))
    # print(train_data.shape)

    # masked = masker(feat)[0]
    # # print(masked)
    # print(feat.shape, masked.shape)
