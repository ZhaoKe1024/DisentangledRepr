#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/10 9:10
# @Author: ZhaoKe
# @File : infogan_mnist.py
# @Software: PyCharm
# highlight: the calculation of mutual information
import os
import itertools
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image


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


iscuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if iscuda else torch.FloatTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class TrainerInfoGAN(object):
    def __init__(self, configs, istrain=True):
        self.configs = configs
        setup_seed(3407)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.is_train = istrain
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + '/'

            os.makedirs(self.run_save_dir + 'static/', exist_ok=True)
            os.makedirs(self.run_save_dir + 'varying_c1/', exist_ok=True)
            os.makedirs(self.run_save_dir + 'varying_c2/', exist_ok=True)

    def __build_models(self):
        self.adversarial_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.MSELoss()
        self.lambda_cat = 1.
        self.lambda_con = 0.1
        self.generator = Generator(img_size=self.configs["img_size"], latent_dim=self.configs["latent_dim"],
                                   class_num=self.configs["class_num"],  code_dim=self.configs["code_dim"],
                                   channels=self.configs["channels"])
        self.discriminator = Discriminator(img_size=self.configs["img_size"], n_classes=self.configs["class_num"],
                                           code_dim=self.configs["code_dim"], channels=self.configs["channels"])
        # lr设置为0.0002
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.configs["fit"]["learning_rate"],
                                            betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"]))
        self.optimizer_info = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()),
            lr=self.configs["fit"]["learning_rate"], betas=(self.configs["fit"]["b1"], self.configs["fit"]["b2"])
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.adversarial_loss.to(self.device)  # gan loss

        self.categorical_loss.to(self.device)  # info loss
        self.continuous_loss.to(self.device)  # info loss

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def __build_data(self):
        self.train_dataset = MNIST("F:/DATAS/mnist", train=True, download=True, transform=transforms.Compose([
            transforms.Resize(self.configs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]))  # 之前按照网上说的设置为：(0.1307,), (0.3081,)，结果生成效果巨差。
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.configs["fit"]["batch_size"], shuffle=True)

    def train(self):
        self.__build_data()
        self.__build_models()
        cuda = True if torch.cuda.is_available() else False
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # Static generator inputs for sampling
        static_z = Variable(FloatTensor(np.zeros((self.configs["class_num"] ** 2, self.configs["latent_dim"]))))
        static_label = to_categorical(
            np.array([num for _ in range(self.configs["class_num"]) for num in range(self.configs["class_num"])]),
            num_columns=self.configs["class_num"]
        )
        static_code = Variable(FloatTensor(np.zeros((self.configs["class_num"] ** 2, self.configs["code_dim"]))))

        for epoch_id in range(self.configs["fit"]["epochs"]):
            for i, (x_imgs, labels) in enumerate(self.train_loader):
                batch_size = x_imgs.shape[0]
                # 真伪标签
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
                # 输入数据
                real_imgs = Variable(x_imgs.type(FloatTensor))
                labels = to_categorical(labels.numpy(), num_columns=self.configs["class_num"])
                # print(f"shape of input: \n\tx_imgs {real_imgs.shape}, \n\tlabels {labels.shape}")

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                # 通过随机噪声生成伪造数据
                # Sample noise and labels as generator input
                # (64, 62), (64, 10), (64, 2)
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.configs["latent_dim"]))))
                label_input = to_categorical(np.random.randint(0, self.configs["class_num"], batch_size),
                                             num_columns=self.configs["class_num"])
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.configs["code_dim"]))))
                # print(f"train generator: \n\tz {z.shape}, \n\tlabel_input {label_input.shape}, \n\tcode_input {code_input.shape}")
                # Generate a batch of images
                gen_imgs = self.generator(z, label_input, code_input)
                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(gen_imgs)
                # print(f"train generator: \n\tgen_imgs {gen_imgs.shape}, \n\tvalidity {validity.shape}")
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                # Loss for real images
                real_pred, _, _ = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_pred, valid)
                # Loss for fake images
                fake_pred, _, _ = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # ------------------
                # Information Loss
                # ------------------
                # 首先随机采样标签、噪声和隐向量，然后生成伪图像
                # 互信息的计算，就是预测标签和预测隐向量与真实值的损失
                self.optimizer_info.zero_grad()
                # Sample labels
                sampled_labels = np.random.randint(0, self.configs["class_num"], batch_size)
                # Ground truth labels
                gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)
                # Sample noise, labels and code as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.configs["latent_dim"]))))
                label_input = to_categorical(sampled_labels, num_columns=self.configs["class_num"])
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.configs["code_dim"]))))

                gen_imgs = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_imgs)

                info_loss = self.lambda_cat * self.categorical_loss(pred_label, gt_labels) + self.lambda_con * self.continuous_loss(
                    pred_code, code_input
                )
                info_loss.backward()
                self.optimizer_info.step()

                # --------------
                # Log Progress
                # --------------
                batches_done = epoch_id * len(self.train_loader) + i
                if i > 1 and i % 200 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                        % (epoch_id, self.configs["fit"]["epochs"], i, len(self.train_loader), d_loss.item(), g_loss.item(), info_loss.item())
                    )
                if batches_done > 1 and batches_done % self.configs["sample_interval"] == 0:
                    n_row = 10
                    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, self.configs["latent_dim"]))))
                    static_sample = self.generator(z, static_label, static_code)
                    save_image(static_sample.data, self.run_save_dir+"static/%d.png" % batches_done, nrow=n_row, normalize=True)

                    # Get varied c1 and c2
                    zeros = np.zeros((n_row ** 2, 1))
                    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
                    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
                    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
                    sample1 = self.generator(static_z, static_label, c1)
                    sample2 = self.generator(static_z, static_label, c2)
                    save_image(sample1.data, self.run_save_dir+"varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
                    save_image(sample2.data, self.run_save_dir+"varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)
                # return


if __name__ == '__main__':
    configs = {
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
    # # ========----Model Test----======
    # device = torch.device("cuda") if iscuda else "cpu"
    # model_g = Generator(img_size=32, latent_dim=62, code_dim=2, class_num=10, channels=1).to(device)
    # model_d = Discriminator(img_size=32, n_classes=10, code_dim=2, channels=1).to(device)
    # # Input:
    # z = Variable(FloatTensor(np.random.normal(0, 1, (64, 62))))
    # label_input = to_categorical(np.random.randint(0, 10, 64), num_columns=10)
    # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (64, 2))))
    #
    # gen_imgs = model_g(z, label_input, code_input)
    # print(gen_imgs.shape)
    # fake_pred, cls_pred, latent_code = model_d(gen_imgs.detach())
    # print(fake_pred.shape, cls_pred.shape, latent_code.shape)

    # ===========----Train----===========
    infogan_trainer = TrainerInfoGAN(configs, istrain=True)
    infogan_trainer.train()
