#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/10/7 12:04
# @Author: ZhaoKe
# @File : modules.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pairwise_kl_loss(mu, log_sigma, batch_size):
    mu1 = mu.unsqueeze(dim=1).repeat(1, batch_size, 1)
    log_sigma1 = log_sigma.unsqueeze(dim=1).repeat(1, batch_size, 1)

    mu2 = mu.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    log_sigma2 = log_sigma.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    # print(log_sigma2.shape, log_sigma1.shape)  # ([32, 16, 30]) torch.Size([16, 32, 30])
    kl_divergence1 = 0.5 * (log_sigma2 - log_sigma1)
    kl_divergence2 = 0.5 * torch.div(torch.exp(log_sigma1) + torch.square(mu1 - mu2), torch.exp(log_sigma2))
    kl_divergence_loss1 = kl_divergence1 + kl_divergence2 - 0.5

    pairwise_kl_divergence_loss = kl_divergence_loss1.sum(-1).sum(-1) / (batch_size - 1)
    # print("Pair_kl_loss:", pairwise_kl_divergence_loss)
    return pairwise_kl_divergence_loss


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
        layers = []
        layers.extend([nn.Embedding(num_embeddings=class_num, embedding_dim=oup)])
        layers.extend([nn.Linear(in_features=oup, out_features=oup), nn.LeakyReLU()])
        layers.extend([nn.Linear(in_features=oup, out_features=oup), nn.LeakyReLU()])
        self.net = nn.Sequential(*layers)
        self.emb_lin_mu = nn.Linear(oup, em_dim)
        self.emb_lin_lv = nn.Linear(oup, em_dim)

    @staticmethod
    def sampling(mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, in_a, mu_only=False):
        """

        :param mu_only:
        :param in_a:
        :return: mu, logvar, z
        """
        mapd = self.net(in_a)
        res_mu = self.emb_lin_mu(mapd)
        if mu_only:
            return res_mu
        else:
            res_logvar = self.emb_lin_lv(mapd)
            return res_mu, res_logvar, self.sampling(res_mu, res_logvar)


class Classifier(nn.Module):
    def __init__(self, dim_embedding, dim_hidden_classifier, num_target_class):
        super(Classifier, self).__init__()
        self.ext = nn.Sequential(
            nn.Linear(dim_embedding, dim_hidden_classifier),
            nn.BatchNorm1d(dim_hidden_classifier),
            nn.ReLU(),
            # nn.Linear(dim_hidden_classifier, dim_hidden_classifier),
            # nn.BatchNorm1d(dim_hidden_classifier),
            # nn.ReLU(),
        )
        self.cls = nn.Linear(dim_hidden_classifier, num_target_class)

    def forward(self, input_data, fe=False):
        feat = self.ext(input_data)
        if fe:
            return self.cls(feat), feat
        else:
            return self.cls(feat)


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)  # prob pred
        # print("pred:", P)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        # print(ids)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        # print("alpha:", alpha)
        probs = (P * class_mask).sum(1).view(-1, 1)
        # print("probs:")
        # print(probs)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
