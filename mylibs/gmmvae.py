#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/8/17 16:09
# @Author: ZhaoKe
# @File : gmmvae.py
# @Software: PyCharm
# reference: https://github.com/baohq1595/vae-dec/blob/master/model/dec.py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    # log_pdf = -0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    # return torch.sum(log_pdf, dim=-1)
    return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu) ** 2 / torch.exp(log_var), 1))


activations_mapping = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'linear': None
}


class Stochastic(nn.Module):
    def reparameterize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        z = mu + torch.exp(log_var / 2) * epsilon
        return z


class GaussianSampling(Stochastic):
    def __init__(self, in_features, out_features):
        super(GaussianSampling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(self.in_features, self.out_features, 'mu_sampl')
        self.log_var = nn.Linear(self.in_features, self.out_features, 'log_var_sampl')

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var


class Encoder(nn.Module):
    """
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).

    :param dims: dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """

    def __init__(self, dimensions, **kwargs):
        super(Encoder, self).__init__()

        # unpack dimension of vae
        self.hidden_dims = dimensions[:-1]
        self.latent_dim = dimensions[-1]
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        # Constructing layers
        linear_layers = [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1], 'enc_hid_{}'.format(i))
                         for i, _ in enumerate(self.hidden_dims[:-1])]

        self.bn_layers = [nn.BatchNorm1d(self.hidden_dims[i]).to(self.device) for i in range(1, len(self.hidden_dims))]

        self.linear_layers = nn.ModuleList(linear_layers)
        self.sampling = GaussianSampling(self.hidden_dims[-1], self.latent_dim)

    def forward(self, x):
        for linear_layer, bn_layer in zip(self.linear_layers, self.bn_layers):
            x = linear_layer(x)
            # x = bn_layer(x)
            x = F.relu(x)

        return self.sampling(x)


class Decoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims: dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """

    def __init__(self, dimensions, **kwargs):
        super(Decoder, self).__init__()
        # unpack dimension of vae
        self.embedding_dim = dimensions[-1]
        self.hidden_dims = dimensions[:-1]
        self.dec_final_act = kwargs.get('decoder_final_activation', '')
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        linear_layers = [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1], 'dec_hid_{}'.format(i))
                         for i, _ in enumerate(self.hidden_dims[:-1])]

        self.bn_layers = [nn.BatchNorm1d(self.hidden_dims[i]).to(self.device) for i in range(1, len(self.hidden_dims))]

        self.linear_layers = nn.ModuleList(linear_layers)
        self.reconstruction_layer = nn.Linear(self.hidden_dims[-1], self.embedding_dim, 'recons_layer')
        self.final_act = nn.Sigmoid()

        # Output activation func depends on hyperparameters
        if self.dec_final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif self.dec_final_act == 'tanh':
            self.final_act = nn.Tanh()
        elif self.dec_final_act == 'relu':
            self.final_act = nn.ReLU()

    def forward(self, x):
        for linear_layer, bn_layer in zip(self.linear_layers, self.bn_layers):
            x = linear_layer(x)
            # x = bn_layer(x)
            x = F.relu(x)
        x = self.reconstruction_layer(x)

        if not self.dec_final_act == '':
            x = self.final_act(x)

        return x


class VAE(nn.Module):
    '''
    Variational autoencoder module for deep embedding clustering.
    Args:
        dimensions: list of layers' dimensions for each block encoder/decoder
        For ex: [784, 500, 500, 2000, 10], encoder will have input layer dim 784,
        500-500-2000 is hidden layers' dimension, 10 is latent's dimension.
        Decoder will have reversed order, which is [10, 2000, 500, 500, 784].
        **kwargs: other params dict, contains:
        kwargs['decoder_final_activation'] is final activation function for decoder last
        layer (ex. mnist data use sigmoid, reuters data uses linear,...).
    '''

    def __init__(self, dimensions, **kwargs):
        super(VAE, self).__init__()
        assert len(dimensions) > 1, "Number of dimension must larger than 1 layer"

        self.encoder = Encoder(dimensions, **kwargs)
        self.decoder = Decoder(list(reversed(dimensions)), **kwargs)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_mu = self.decoder(z)

        return x_mu, z, mu, log_var

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (q_mu, q_log_var) = q_param
        qz = log_gaussian(z, q_mu, q_log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (p_mu, p_log_var) = p_param
            pz = log_gaussian(z, p_mu, p_log_var)

        kl = qz - pz

        return kl

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def criterion(self, x, x_decoded, z, z_mean, z_log_var):
        return -self.resconstruction_loss(x_decoded, x) - self._kld(z, (z_mean, z_log_var))


class VAE_Scratch(nn.Module):
    '''
    Variational autoencoder module for deep embedding clustering.
    Args:
        dimensions: list of layers' dimensions for each block encoder/decoder
        For ex: [784, 500, 500, 2000, 10], encoder will have input layer dim 784,
        500-500-2000 is hidden layers' dimension, 10 is latent's dimension.
        Decoder will have reversed order, which is [10, 2000, 500, 500, 784].
        **kwargs: other params dict, contains:
        kwargs['decoder_final_activation'] is final activation function for decoder last
        layer (ex. mnist data use sigmoid, reuters data uses linear,...).
    '''

    def __init__(self, dimensions, **kwargs):
        super(VAE_Scratch, self).__init__()
        assert len(dimensions) > 1

        # unpack dimension of vae
        self.embedding_dim = dimensions[0]
        self.hidden_dims = dimensions[1:-1]
        self.latent_dim = dimensions[-1]
        self.dec_final_act = kwargs['decoder_final_activation']
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.is_logits = kwargs.get('logits', False)

        if self.is_logits:
            self.resconstruction_loss = nn.modules.loss.MSELoss()
        else:
            self.resconstruction_loss = self.binary_cross_entropy

        # Construct layers for encoder and decoder block
        # Encoder block
        self.enc_hidden_layers = nn.Sequential()
        self.enc_hidden_layers.add_module('hidden_layer_0',
                                          nn.Linear(self.embedding_dim, self.hidden_dims[0]))
        self.enc_hidden_layers.add_module('h_layer_act_0', nn.ReLU())

        for i, _ in enumerate(self.hidden_dims[:-1]):
            self.enc_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                                              nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])),
            self.enc_hidden_layers.add_module('h_layer_act_{}'.format(i + 1), nn.ReLU())

        # define mean and log variance of vae
        self.z_mean = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.z_log_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        # ~Encoder block

        # Decoder block
        dec_hidden_layers = nn.Sequential()
        dec_hidden_layers.add_module('hidden_layer_0',
                                     nn.Linear(self.latent_dim, self.hidden_dims[-1]))
        dec_hidden_layers.add_module('h_layer_act_0',
                                     nn.ReLU())
        reversed_hidden_dims = list(reversed(self.hidden_dims))
        for i, _ in enumerate(reversed(self.hidden_dims)):
            if i == (len(reversed_hidden_dims) - 1):
                dec_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                                             nn.Linear(reversed_hidden_dims[i], self.embedding_dim)),
            else:
                dec_hidden_layers.add_module('hidden_layer_{}'.format(i + 1),
                                             nn.Linear(reversed_hidden_dims[i], reversed_hidden_dims[i + 1])),
                dec_hidden_layers.add_module('h_layer_act_{}'.format(i + 1), nn.ReLU())

        # Final activation function of decoder depends on data
        if self.dec_final_act is not None:
            if self.dec_final_act == 'sigmoid':
                dec_hidden_layers.add_module('dec_final_act', nn.Sigmoid())
            elif self.dec_final_act == 'tanh':
                dec_hidden_layers.add_module('dec_final_act', nn.Tanh())
            elif self.dec_final_act == 'relu':
                dec_hidden_layers.add_module('dec_final_act', nn.ReLU())
            else:
                pass
        self.decoder = dec_hidden_layers
        # ~ Decoder block

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.to(self.device)

    def binary_cross_entropy(self, r, x):
        return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

    def encode(self, x):
        '''
        Encoding step. Receives data input tensor and produce is gaussian distribution
        params, which is mean and log-variance.
        Args:
            x: data input
        '''
        hidden_embedding = self.enc_hidden_layers(x)
        z_mean = self.z_mean(hidden_embedding)
        z_log_var = self.z_log_var(hidden_embedding)

        latent = self.sampling(z_mean, z_log_var)

        return latent, z_mean, z_log_var

    def decode(self, z):
        '''
        Decoding step. Receives latent vector encoded from encoder, and decode
        it back to observed data distribution.
        Args:
            z: latent vector.
        '''
        return self.decoder(z)

    def sampling(self, z_mean, z_log_var):
        '''
        Sampling function, which produce latent vector from params mean
        and log variance using formula as:
            z = z_mean + z_var*epsilon
        Args:
            z_mean: mean value of a gaussian.
            z_log_var: log of variance of a gaussian.
        '''
        epsilon = torch.randn(z_mean.size()).to(self.device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (q_mu, q_log_var) = q_param
        qz = log_gaussian(z, q_mu, q_log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (p_mu, p_log_var) = p_param
            pz = log_gaussian(z, p_mu, p_log_var)

        kl = qz - pz

        return kl

    def criterion(self, x, x_decoded, z, z_mean, z_log_var):
        return -self.resconstruction_loss(x_decoded, x) - self._kld(z, (z_mean, z_log_var))

    def forward(self, x):
        latent, z_mean, z_log_var = self.encode(x)
        x_hat = self.decode(latent)

        return x_hat, latent, z_mean, z_log_var


class ClusteringBasedVAE(nn.Module):
    def __init__(self, n_clusters, dimensions, alpha, **kwargs):
        super(ClusteringBasedVAE, self).__init__()
        # self.vae = VAE(dimensions, **kwargs)
        self.encoder = Encoder(dimensions, **kwargs)
        self.decoder = Decoder(list(reversed(dimensions)), **kwargs)
        self.embedding_dim = dimensions[0]
        self.latent_dim = dimensions[-1]
        self.is_logits = kwargs.get('logits', False)
        self.alpha = alpha

        self.n_centroids = n_clusters
        self.pi = nn.Parameter(torch.ones(self.n_centroids, dtype=torch.float32) / self.n_centroids)
        self.mu_c = nn.Parameter(torch.zeros((self.n_centroids, self.latent_dim), dtype=torch.float32))
        self.log_sigma_c = nn.Parameter(torch.ones((self.n_centroids, self.latent_dim), dtype=torch.float32))
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        if self.is_logits:
            self.resconstruction_loss = nn.modules.loss.MSELoss()
        else:
            self.resconstruction_loss = self.binary_cross_entropy

    @staticmethod
    def binary_cross_entropy(x, r):
        return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

    def forward(self, x):
        latent, _, _ = self.encoder(x)

        pi = self.pi
        log_sigmac_c = self.log_sigma_c
        mu_c = self.mu_c
        pzc = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(latent, mu_c, log_sigmac_c))

        return pzc

    def elbo_loss(self, x, L=1):
        det = 1e-10
        res_loss = 0.0

        _, mu, logvar = self.encoder(x)
        for l in range(L):
            z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu

            x_decoded = self.decoder(z)

            if self.is_logits:
                res_loss += F.mse_loss(x_decoded, x)
            else:
                # print(x_decoded.max(), x_decoded.min())
                # print(x.max(), x.min())
                res_loss += F.binary_cross_entropy(x_decoded/x_decoded.max(), x)
                # res_loss += F.binary_cross_entropy(x, x_decoded)

        res_loss /= L
        loss = self.alpha * res_loss * x.size(1)
        pi = self.pi
        log_sigma2_c = self.log_sigma_c
        mu_c = self.mu_c

        z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu
        pcz = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        pcz = pcz / (pcz.sum(1).view(-1, 1))  # batch_size*clusters

        loss += 0.5 * torch.mean(torch.sum(pcz * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                           torch.exp(logvar.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                           (mu.unsqueeze(1) - mu_c.unsqueeze(0)) ** 2 / torch.exp(
            log_sigma2_c.unsqueeze(0)), 2), 1))

        loss -= torch.mean(torch.sum(pcz * torch.log(pi.unsqueeze(0) / (pcz)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + logvar, 1))

        return loss

    def log_gaussians(self, x, mus, logvars):
        G = []
        for c in range(self.n_centroids):
            G.append(log_gaussian(x, mus[c:c + 1, :], logvars[c:c + 1, :]).view(-1, 1))

        return torch.cat(G, 1)

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.n_centroids):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))


if __name__ == '__main__':
    vae = ClusteringBasedVAE(n_clusters=10, dimensions=[784, 256, 128, 32], alpha=1)
    input_x = torch.rand(size=(16, 784))
    y = vae(x=input_x)
    print(y.shape)
