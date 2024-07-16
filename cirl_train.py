#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/13 17:11
# @Author: ZhaoKe
# @File : cirl_train.py
# @Software: PyCharm
import argparse
import ast
import numpy as np
import torch
from torch import nn
from torch import optim
from cirl_libs.cirl_model import Masker
from cirl_libs.ResNet import resnet18
from cirl_libs.cirl_datasets import get_fourier_train_dataloader, get_val_dataloader, get_test_loader

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
lam_const = 5.0  # loss weight for factorization loss
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
    "lr": 10 * lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": True,
    "sched_type": "step",
    "lr_decay_step": int(epoch * 0.8),
    "lr_decay_rate": lr_decay_rate
}


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)


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
    parser.add_argument("--config", default="cirl_libs/ResNet18", help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True,
                        help="If true will save tensorboard compatible logs")
    args = parser.parse_args()
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(args.config.replace('/', '.'), fromlist=[""]).config
    print(config)
    for item in config:
        print(item, '\t', config[item])
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
        # Masker(in_dim=512, num_classes=512, middle=4*512, k=308)
        self.masker = Masker(in_dim=dim, num_classes=dim, middle=4 * dim, k=self.config["k"]).to(device)
        # print("=====------encoder------=====")
        # print(self.encoder)
        # print("=====------classifier------=====")
        # print(self.classifier)
        # print("=====------classifier ad------=====")
        # print(self.classifier_ad)
        # print("=====------masker------=====")
        # print(self.masker)
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

    def _do_epoch(self, current_epoch):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        self.encoder.train()
        self.classifier.train()
        self.classifier_ad.train()
        self.masker.train()

        for it, (batch, label, domain) in enumerate(self.train_loader):

            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            labels = torch.cat(label, dim=0).to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            self.classifier_ad_optim.zero_grad()
            self.masker_optim.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            ## --------------------------step 1 : update G and C -----------------------------------
            features = self.encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            if current_epoch <= 5:
                masks_sup = torch.ones_like(features.detach())
                masks_inf = torch.ones_like(features.detach())
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            features_ori, features_aug = torch.split(features, split_idx)
            assert features_ori.size(0) == features_aug.size(0)

            # classification loss for sup feature
            loss_cls_sup = criterion(scores_sup, labels)
            loss_dict["sup"] = loss_cls_sup.item()
            correct_dict["sup"] = calculate_correct(scores_sup, labels)
            num_samples_dict["sup"] = int(scores_sup.size(0))

            # classification loss for inf feature
            loss_cls_inf = criterion(scores_inf, labels)
            loss_dict["inf"] = loss_cls_inf.item()
            correct_dict["inf"] = calculate_correct(scores_inf, labels)
            num_samples_dict["inf"] = int(scores_inf.size(0))

            # factorization loss for features between ori and aug
            loss_fac = factorization_loss(features_ori, features_aug)
            loss_dict["fac"] = loss_fac.item()

            # get consistency weight
            const_weight = get_current_consistency_weight(epoch=current_epoch,
                                                          weight=self.config["lam_const"],
                                                          rampup_length=self.config["warmup_epoch"],
                                                          rampup_type=self.config["warmup_type"])

            # calculate total loss
            total_loss = 0.5 * loss_cls_sup + 0.5 * loss_cls_inf + const_weight * loss_fac
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.classifier_ad_optim.step()

            ## ---------------------------------- step2: update masker------------------------------
            self.masker_optim.zero_grad()
            features = self.encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            loss_cls_sup = criterion(scores_sup, labels)
            loss_cls_inf = criterion(scores_inf, labels)
            total_loss = 0.5 * loss_cls_sup - 0.5 * loss_cls_inf
            total_loss.backward()
            self.masker_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        self.masker.eval()
        self.classifier_ad.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][current_epoch] = class_acc

            # save from best model
            if self.results['test'][current_epoch] >= self.best_acc:
                self.best_acc = self.results['test'][current_epoch]
                self.best_epoch = current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_acc)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct

    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_acc = 0
        self.best_epoch = 0

        for current_epoch in range(self.epochs):
            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch(current_epoch)
            self.logger.finish_epoch()

        # save from best model
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_acc, self.best_epoch - 1)

        return self.logger


def main():
    args, config = get_args()
    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    # trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
