#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/8/17 16:22
# @Author: ZhaoKe
# @File : vade_mnist.py
# @Software: PyCharm
import os
from time import strftime, gmtime
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from mylibs.gmmvae import ClusteringBasedVAE
import matplotlib.pyplot as plt

pretrained_save_path = './runs/gmmvae/model/pretrained/model_10.pt'


def cluster_accuracy(predicted: np.array, target: np.array):
    assert predicted.size == target.size, ''.join('Different size between predicted\
        and target, {} and {}').format(predicted.size, target.size)

    D = max(predicted.max(), target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(predicted.size):
        w[predicted[i], target[i]] += 1

    ind_1, ind_2 = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_1, ind_2)]) * 1.0 / predicted.size, w


def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    if os.path.exists(pretrained_save_path):
        print("Model exists, Loading Model...")
        model.load_state_dict(torch.load(pretrained_save_path))
        return
    else:
        os.makedirs(os.path.dirname(pretrained_save_path), exist_ok=True)

    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', './runs/gmmvae')
    dataset_name = params.get('dataset_name', '')
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    res_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                                 model.decoder.parameters()), lr=0.002)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Pretrains VAE using only reconstruction loss...')
    for pre_epoch in range(num_pretrained_epoch):
        total_loss = 0.0
        iters = 0
        for i, data in enumerate(train_dataloader):
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)

            x = x.float()
            x = x.to(device)
            # Forward pass
            _, z_mu, _ = model.encoder(x)
            x_decoded = model.decoder(z_mu)
            loss = res_loss(x_decoded, x)
            total_loss += loss.detach().cpu().numpy()

            # Calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1

        print('VAE resconstruction loss: ', total_loss / iters)
        steplr.step()

    model.encoder.sampling.log_var.load_state_dict(model.encoder.sampling.mu.state_dict())

    Z = []
    Y = []
    with torch.no_grad():
        for x, y in train_dataloader:
            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)

            x = x.float()
            x = x.to(device)
            z, mu, log_var = model.encoder(x)
            assert F.mse_loss(mu, log_var) == 0
            Z.append(mu)
            Y.append(y)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()
    gmm = GaussianMixture(n_components=model.n_centroids, covariance_type='diag')
    predict = gmm.fit_predict(Z)

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_c.data = torch.from_numpy(gmm.means_).to(device).float()
    model.log_sigma_c.data = torch.log(torch.from_numpy(gmm.covariances_).to(device).float())
    model.pi.data = torch.from_numpy(gmm.weights_).to(device).float()

    torch.save(model.state_dict(), pretrained_save_path)
    print("Pretrain End, ckpt saved at:", pretrained_save_path)


def train(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, eps=1e-4)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = params.get('epochs', 80)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epoch = 0
    for epoch in range(num_epochs):
        train_iters = 0
        total_loss = 0.0
        for i, data in enumerate(train_dataloader):
            steplr.step()
            model.zero_grad()

            # Get only data, ignore label (data[1])
            x = data[0]

            # Flatten 28x28 to 1x784 on mnist dataset
            if dataset_name == 'mnist':
                x = x.view(x.size()[0], -1)
            # model = model.to(torch.device("cpu"))
            # x = x.to(torch.device("cpu"))
            x = x.to(model.device)
            # print(model.device, x.device)
            # Acquire the loss
            loss = model.elbo_loss(x, 1)

            # Calculate gradients
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Update models
            optimizer.step()

            train_iters += 1
            if i % 250 == 0:
                print("Train Loss:", loss.item())
            total_loss += loss.detach().cpu().numpy()

        print('Training loss: ', total_loss / train_iters)

        gtruth = []
        predicted = []
        # For each epoch, log the p_c_z accuracy
        with torch.no_grad():
            mean_accuracy = 0.0
            iters = 0
            for i, data in enumerate(val_dataloader):
                # Get z value
                x = data[0].to(model.device)
                labels = data[1].cpu().detach().numpy()
                if dataset_name == 'mnist':
                    x = x.view(x.size()[0], -1)

                # x_decoded, latent, z_mean, z_log_var, gamma = model(x)
                gamma = model(x)

                gtruth.append(labels)

                # Cluster latent space
                sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
                predicted.append(sample)
                # mean_accuracy += cluster_accuracy(sample, labels)[0]
                iters += 1

            gtruth = np.concatenate(gtruth, 0)
            predicted = np.concatenate(predicted, 0)
            print('accuracy p(c|z): {:0.4f}'.format(cluster_accuracy(predicted, gtruth)[0] * 100))
        if epoch % 7 == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
                                                        .format(epoch)))
    torch.save(model.state_dict(), os.path.join(save_path, 'vae-dec-model-{}-{}'
                                                .format(epoch, strftime("%Y-%m-%d-%H-%M", gmtime())
                                                        )))


# if __name__ == '__main__':
#     D = np.array([[2, 4, 5], [3, 5, 7], [7, 5, 3]])
#     print([D[i, j] for i, j in [[1, 2], [2, 0]]])

if __name__ == '__main__':
    dimensions = [784, 256, 64, 8]
    save_dir = './runs/gmmvae/'
    model_params = {
        'decoder_final_activation': 'relu',
        'pretrained_epochs': 10,
        'epochs': 80,
        'save_path': save_dir,
        'dataset_name': 'mnist',
        'logits': True
    }
    dec_cluster = ClusteringBasedVAE(2, dimensions, 1, **model_params)
    mnist_path = "F:/DATAS/mnist"
    train_dataloader = DataLoader(MNIST(mnist_path, train=True,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.1307,), (0.3081,)),
                                            # transforms.Normalize((0,), (1,))
                                        ])),
                                  batch_size=32,
                                  shuffle=True)
    val_dataloader = DataLoader(MNIST(mnist_path, train=False,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          # transforms.Normalize((0.1307,), (0.3081,)),
                                          # transforms.Normalize((0,), (1,))
                                      ])),
                                batch_size=32,
                                shuffle=True)
    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()
    else:
        print('No GPU')
    pretrain(dec_cluster, train_dataloader, None, **model_params)
    train(dec_cluster, train_dataloader, val_dataloader, **model_params)
