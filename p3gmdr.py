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
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torch import optim
from torchvision import transforms
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset, DataLoader
from p2ammidr import MyDataset, get_datasets


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers


class Encoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=None, hid_dim=10):
        super(Encoder, self).__init__()

        if inter_dims is None:
            inter_dims = [500, 500, 2000]
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
    def __init__(self, input_dim=784, inter_dims=None, hid_dim=10):
        super(Decoder, self).__init__()

        if inter_dims is None:
            inter_dims = [500, 500, 2000]
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


class VaDE(nn.Module):
    def __init__(self, device=torch.device("cuda"), args=None):
        super(VaDE, self).__init__()
        self.encoder = Encoder(hid_dim=args["z_dim"]+args["a_dim"])
        self.decoder = Decoder(hid_dim=args["z_dim"]+args["a_dim"])
        self.pi_prior = nn.Parameter(torch.FloatTensor(args["nClusters"], ).fill_(1) / args["nClusters"],
                                     requires_grad=True)
        self.mu_prior = nn.Parameter(torch.FloatTensor(args["nClusters"], args["a_dim"]).fill_(0), requires_grad=True)
        self.log_var_prior = nn.Parameter(torch.FloatTensor(args["nClusters"], args["a_dim"]).fill_(0),
                                          requires_grad=True)
        self.a_st = args["z_dim"]
        self.args = args
        self.device = device

    def compute_gamma(self, z, p_c):
        print("shape pi_c：{} mu_c：{} logvar_c:{}".format(self.pi_prior.shape, self.mu_prior.shape, self.log_var_prior.shape))
        h = (z.unsqueeze(1) - self.mu_prior).pow(2) / self.log_var_prior.exp()
        h += self.log_var_prior
        h += torch.Tensor([np.log(np.pi * 2)]).to(self.device)
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-9
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return gamma

    def compute_loss(self, x, x_hat, mu, log_var, z):
        p_c = self.pi_prior
        gamma = self.compute_gamma(z[:, :, self.a_st:], p_c)
        log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum')
        h = log_var[:, :, self.a_st:].exp().unsqueeze(1) + (mu[:, :, self.a_st:].unsqueeze(1) - self.mu_prior).pow(2)
        h = torch.sum(self.log_var_prior + h / self.log_var_prior.exp(), dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h)
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var[:, :, self.a_st:])
        loss = log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
        loss /= x.size(0)
        print("loss:{}".format(loss))
        return loss

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        print("x:", x.shape)
        mu, log_var = self.encoder(x=x)
        print("mu:{}, logvar:{}".format(mu.shape, log_var.shape))
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z=z)
        print("z:{}, x_hat:{}".format(z.shape, x_hat.shape))
        return x_hat, mu, log_var, z


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
        self.encoder = Encoder(input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10)
        self.generator = Decoder(input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10)
        self.classifier = VIB_Classifier(dim_embedding=10 + embedding_dim, dim_hidden_classifier=32,
                                         num_target_class=self.class_num)
        # self.discriminator = Discriminator(img_size=self.img_size, channels=self.channel, n_classes=self.class_num,
        #                                    code_dim=self.code_dim)

        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        # self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.configs["fit"]["learning_rate"],
        #                                     betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))

        self.adversarial_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.MSELoss()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def train(self):
        cuda = True if torch.cuda.is_available() else False
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        for epoch_id in range(100):
            for jdx, (x_img, y_lab, a_sen) in tqdm(enumerate(self.train_loader), desc="Epoch[{}]".format(epoch_id)):
                print("Batch[{}]".format(jdx))
                bs = x_img.shape[0]

                self.optimizer_E.zero_grad()
                latent_mu, latent_logvar = self.encoder(x=x_img)
                latent_vec = self.reparameterize(latent_mu, latent_logvar)

                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (bs, self.code_dim))))
                one_hot = torch.zeros((bs, self.class_num))
                one_hot = one_hot.scatter_(1, y_lab.unsqueeze(1).long(), 1)
                recon = self.generator(noise=latent_vec, labels=one_hot, code=code_input)

                validity, _, _ = self.discriminator(img=recon)

                if epoch_id == 0 and jdx == 0:
                    print(x_img.shape, y_lab.shape, a_sen.shape)
                    print(latent_vec.shape)
                    print(recon.shape)
                    print(validity.shape)

                return
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
        train_dataset = MyDataset(train_data, train_labels, train_sensitive_labels, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.__build_models()
        for jdx, (x_img, y_lab, a_sen) in tqdm(enumerate(train_loader), desc="Epoch[{}]".format(0)):
            print("Batch[{}]".format(jdx))
            bs = x_img.shape[0]
            print(x_img.shape, x_img.view(bs, -1).shape, y_lab.shape)
            z_mu, z_lv = self.encoder(x=x_img.view(bs, -1))
            z_h = self.reparameterize(mu=z_mu, logvar=z_lv)
            bs = z_h.shape[0]
            code = torch.rand(size=(bs, self.code_dim))
            label_vec = self.embedding(y_lab.to(torch.long))
            print("z_h:{}, label_vec:{}, code:{}".format(z_h.shape, label_vec.shape, code.shape))
            x_Recon = self.generator(z=z_h)

            print("x_recon:", x_Recon.view(bs, 1, 28, 28).shape)
            pred = self.classifier(torch.concat((z_h, label_vec), dim=-1))
            print("pred:", pred.shape)

            return

    def vade_demo(self, epoch):
        args = {"nClusters": 5, "z_dim": 6, "a_dim": 4}
        self.__build_dataloaders()
        self.VaDE = VaDE(args=args)

        self.VaDE.train()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.configs["fit"]["learning_rate"],
                                          betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        total_loss = 0
        for jdx, (x_img, y_lab, a_sen) in tqdm(enumerate(self.train_loader), desc="Epoch[{}]".format(0)):
            print("Batch[{}]".format(jdx))
            bs = x_img.shape[0]
            print(x_img.shape, x_img.view(bs, -1).shape, y_lab.shape)

            bs = x_img.shape[0]
            self.optimizer.zero_grad()
            x_img = x_img.to(self.device)
            print(x_img.shape, x_img.view(bs, -1).shape, y_lab.shape)
            label_vec = self.embedding(a_sen.to(torch.long))

            aem_hat, mu, log_var, aem_z = self.VaDE(label_vec)
            # print('Before backward: {}'.format(self.VaDE.pi_prior))
            loss_gmm = self.VaDE.compute_loss(label_vec, aem_hat, mu, log_var, aem_z)

            z_mu, z_lv = self.encoder(x=x_img.view(bs, -1))
            z_h = self.reparameterize(mu=z_mu, logvar=z_lv)
            bs = z_h.shape[0]
            code = torch.rand(size=(bs, self.code_dim))
            print("z_h:{}, label_vec:{}, code:{}".format(z_h.shape, label_vec.shape, code.shape))
            x_recon = self.generator(z=z_h)

            print("x_recon:", x_recon.view(bs, 1, 28, 28).shape)
            pred = self.classifier(torch.concat((z_h, label_vec), dim=-1))
            print("pred:", pred.shape)
            

            loss_gmm.backward()
            self.optimizer.step()
            total_loss += loss_gmm.item()
            # print('After backward: {}'.format(self.VaDE.pi_prior))
        print('Training VaDE... Epoch: {}, Loss: {}'.format(epoch, total_loss))


if __name__ == '__main__':
    # loaders, sets = get_mnist()
    # print(len(sets["X"]), len(sets["Y"]))
    # trainer = AMMIDRTrainer()
    # trainer.demo()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_size, channel = 28, 1
    class_num, batch_size = 10, 16
    latent_dim, code_dim = 36, 4
    embedding_dim = 4
    args = {"nClusters": 5, "z_dim": 10, "a_dim": 5}
    vade = VaDE(args=args).to(device)
    embedding = nn.Embedding(num_embeddings=class_num, embedding_dim=embedding_dim)
    x_img = torch.rand(size=(batch_size, channel, img_size * img_size), device=device)
    y_lab = torch.randint(low=0, high=10, size=(batch_size,), device=device)
    a_sen = torch.randint(low=0, high=5, size=(batch_size,), device=device)
    label_vec = embedding(a_sen.to(torch.long))

    x_hat, mu, logvar, z = vade(x=x_img)
    loss_gmm = vade.compute_loss(x_img, x_hat, mu, logvar, z)




    loss_gmm.backward()


    # label = torch.randint(0, class_num, size=(batch_size,))
    # code = torch.rand(size=(batch_size, code_dim))
    #
    # encoder = Encoder()
    # decoder = Decoder()
