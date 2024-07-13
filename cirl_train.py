#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/13 17:11
# @Author: ZhaoKe
# @File : cirl_train.py
# @Software: PyCharm
import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from cirl_libs.ResNet import resnet18


digits_datset = ["mnist", "mnist_m", "svhn", "syn"]
pacs_dataset = ["art_painting", "cartoon", "photo", "sketch"]
officehome_dataset = ['Art', 'Clipart', 'Product', 'RealWorld']
available_datasets = pacs_dataset + officehome_dataset + digits_datset
batch_size = 16
epoch = 50
warmup_epoch = 5
warmup_type = "sigmoid"
lr = 0.001
lr_decay_rate = 0.1
lam_const = 5.0    # loss weight for factorization loss
T = 10.0
k = 308
encoder_optimizer = {
    "optim_type": "sgd",
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}

classifier_optimizer = {
    "optim_type": "sgd",
    "lr": 10*lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}

def get_optim_and_scheduler(network, optimizer_config):
    params = network.parameters()

    if optimizer_config["optim_type"] == 'sgd':
        optimizer = optim.SGD(params,
                              weight_decay=optimizer_config["weight_decay"],
                              momentum=optimizer_config["momentum"],
                              nesterov=optimizer_config["nesterov"],
                              lr=optimizer_config["lr"])
    elif optimizer_config["optim_type"] == 'adam':
        optimizer = optim.Adam(params,
                               weight_decay=optimizer_config["weight_decay"],
                               lr=optimizer_config["lr"])
    else:
        raise ValueError("Optimizer not implemented")

    if optimizer_config["sched_type"] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=optimizer_config["lr_decay_step"],
                                              gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["sched_type"] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=optimizer_config["lr_decay_step"],
                                                   gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["sched_type"] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=optimizer_config["lr_decay_rate"])
    else:
        raise ValueError("Scheduler not implemented")

    return optimizer, scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default="./cirl_libs/ResNet18", help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores


class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(mask,z)
       return z


class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = resnet18(pretrained=True).to(device)
        self.classifier = Classifier(in_dim=512, num_classes=10).to(device)
        self.classifier_ad = Classifier(in_dim=512, num_classes=10).to(device)
        dim = self.config["networks"]["classifier"]["in_dim"]
        self.masker = Masker(in_dim=dim, num_classes=dim, middle=4 * dim, k=self.config["k"]).to(device)

        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_ad_optim, self.classifier_ad_sched = \
            get_optim_and_scheduler(self.classifier_ad, self.config["optimizer"]["classifier_optimizer"])
        self.masker_optim, self.masker_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}



def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
