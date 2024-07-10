#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/10 17:56
# @Author: ZhaoKe
# @File : dstgib_mnist.py
# @Software: PyCharm
# reference: https://github.com/PanZiqiAI/disentangled-information-bottleneck
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from dstgib_libs.logger import Logger
from dstgib_libs.dataloader_mnist import DataCycle, ImageMNIST
from dstgib_libs.model_build import IterativeBaseModel, fet_d, ValidContainer
from dstgib_libs.custom_operations import BaseCriterion, TensorWrapper, set_requires_grad
from dstgib_libs.basic_metrics import FreqCounter, TriggerLambda, TriggerPeriod


# ----------------------------------------------------------------------------------------------------------------------
# Criterion
# ----------------------------------------------------------------------------------------------------------------------


class CrossEntropyLoss(BaseCriterion):
    """
    Classification loss.
    """

    def __init__(self, lmd=None):
        super(CrossEntropyLoss, self).__init__(lmd)
        # Config
        self._loss = nn.CrossEntropyLoss()

    def _call_method(self, output, label):
        return self._loss(output, label)


class RecLoss(BaseCriterion):
    """
    Reconstruction Loss.
    """

    def _call_method(self, ipt, target):
        loss_rec = torch.sum((ipt - target).pow(2)) / ipt.data.nelement()
        # Return
        return loss_rec


class EstLoss(BaseCriterion):
    """
    Estimator objective.
    """

    def __init__(self, radius, lmd=None):
        super(EstLoss, self).__init__(lmd=lmd)
        # Config
        self._radius = radius

    def _call_method(self, mode, **kwargs):
        assert mode in ['main', 'est']
        # 1. Calculate for main
        if mode == 'main':
            # (1) Density estimation
            loss_est = -kwargs['output'].mean()
            # (2) Making embedding located in [-radius, radius].
            emb = torch.cat(kwargs['emb'], dim=0)
            loss_wall = torch.relu(torch.abs(emb) - self._radius).square().mean()
            # Return
            return {'loss_est': loss_est, 'loss_wall': loss_wall}, -loss_est
        # 2. Calculate for estimator
        else:
            # (1) Real & fake losses
            loss_real = torch.mean((1.0 - kwargs['output_real']) ** 2)
            loss_fake = torch.mean((1.0 + kwargs['output_fake']) ** 2)
            # (2) Making outputs of the estimator to be zero-centric
            outputs = torch.cat([kwargs['output_real'], kwargs['output_fake']], dim=0)
            loss_zc = torch.mean(outputs).square()
            # Return
            return {'loss_real': loss_real, 'loss_fake': loss_fake, 'loss_zc': loss_zc}, \
                (kwargs['output_real'].mean(), kwargs['output_fake'].mean())

    def __call__(self, mode, **kwargs):
        ret = super(EstLoss, self).__call__(mode, **kwargs)
        # 1. For main
        if mode == 'main':
            losses, est = ret if isinstance(ret, tuple) else (ret, TensorWrapper(None))
            losses.update({'est': est})
            # Return
            return losses
        # 2. For estimator
        else:
            losses, (est_real, est_fake) = ret if isinstance(ret, tuple) else (
                ret, TensorWrapper(None), TensorWrapper(None))
            losses.update({'est_real': est_real, 'est_fake': est_fake})
            # Return
            return losses


# ----------------------------------------------------------------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------------------------------------------------------------

class GANLoss(BaseCriterion):
    """
    GAN objectives.
    """

    def __init__(self, lmd=None):
        """
        Adversarial loss.
        """
        super(GANLoss, self).__init__(lmd)
        # Set loss
        self.__loss = nn.CrossEntropyLoss()

    def _call_method(self, pred, target_is_real):
        target_tensor = torch.tensor(1 if target_is_real else 0, dtype=torch.long).to(pred.device)
        loss = self.__loss(pred, target_tensor.expand(pred.size(0), ))
        # Return
        return loss, torch.max(pred, dim=1)[1]

    def __call__(self, prediction, target_is_real, **kwargs):
        # Get result
        ret = super(GANLoss, self).__call__(prediction, target_is_real, **kwargs)
        loss, pred = ret if isinstance(ret, tuple) else (ret, TensorWrapper(None))
        # Return
        return {'loss': loss, 'pred': pred}


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


class Decoder(nn.Module):
    """
    Decoder module.
    """

    def __init__(self, class_dim, num_classes):
        super(Decoder, self).__init__()
        # 1. Architecture
        self._fc = nn.Linear(in_features=class_dim, out_features=num_classes, bias=False)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, emb):
        return self._fc(emb)


class DensityEstimator(nn.Module):
    """
    Estimating probability density.
    """

    def __init__(self, style_dim, class_dim):
        super(DensityEstimator, self).__init__()
        # 1. Architecture
        # (1) Pre-fc
        self._fc_style = nn.Linear(in_features=style_dim, out_features=128, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=128, bias=True)
        # (2) FC blocks
        self._fc_blocks = nn.Sequential(
            # Layer 1
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Linear(in_features=256, out_features=1, bias=True))
        # 2. Init weights
        self.apply(init_weights)

    def _call_method(self, style_emb, class_emb):
        style_emb = self._fc_style(style_emb)
        class_emb = self._fc_class(class_emb)
        return self._fc_blocks(torch.cat([style_emb, class_emb], dim=1))

    def forward(self, style_emb, class_emb, mode):
        assert mode in ['orig', 'perm']
        # 1. q(s, t)
        if mode == 'orig':
            return self._call_method(style_emb, class_emb)
        # 2. q(s)q(t)
        else:
            # Permutation
            style_emb_permed = style_emb[torch.randperm(style_emb.size(0)).to(style_emb.device)]
            class_emb_permed = class_emb[torch.randperm(class_emb.size(0)).to(class_emb.device)]
            return self._call_method(style_emb_permed, class_emb_permed)


# ----------------------------------------------------------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------------------------------------------------------

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
        self._fc = nn.Linear(in_features=256, out_features=nz, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        x = self._conv_blocks(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        ret = self._fc(x)
        # Return
        return ret


class ReconstructorMNIST(nn.Module):
    """
    Decoder Module.
    """

    def __init__(self, style_dim, class_dim, num_classes):
        super(ReconstructorMNIST, self).__init__()
        # 1. Architecture
        self.register_parameter('word_dict', torch.nn.Parameter(torch.randn(size=(num_classes, class_dim))))
        # (1) FC
        self._fc_style = nn.Linear(in_features=style_dim, out_features=256, bias=True)
        self._fc_class = nn.Linear(in_features=class_dim, out_features=256, bias=True)
        # (2) Convolution
        self._deconv_blocks = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(num_features=16, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid())
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, style_emb, class_label):
        # Get class dim
        class_emb = torch.index_select(self.word_dict, dim=0, index=class_label)
        # 1. FC
        style_emb = F.leaky_relu_(self._fc_style(style_emb), negative_slope=0.2)
        class_emb = F.leaky_relu_(self._fc_class(class_emb), negative_slope=0.2)
        # 2. Convolution
        x = torch.cat((style_emb, class_emb), dim=1)
        x = x.view(x.size(0), 128, 2, 2)
        x = self._deconv_blocks(x)
        # Return
        return x


class DiscriminatorMNIST(nn.Module):
    """
    Discriminator Module.
    """

    def __init__(self):
        super(DiscriminatorMNIST, self).__init__()
        # 1. Architecture
        # (1) Convolution
        self._conv_blocks = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=32, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=64, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # (2) FC
        self._fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # 2. Init weights
        self.apply(init_weights)

    def forward(self, x):
        # 1. Convolution
        x = self._conv_blocks(x)
        # 2. FC
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self._fc(x)
        # Return
        return x


# ----------------------------------------------------------------------------------------------------------------------
# operations
# ----------------------------------------------------------------------------------------------------------------------

def resampling(mu, std, **kwargs):
    """
    Resampling trick.
    """
    # Multi sampling. (batch*n_samples, nz)
    if 'n_samples' in kwargs.keys():
        if kwargs['n_samples'] > 0:
            eps = torch.randn(mu.size(0), kwargs['n_samples'], mu.size(1), device=mu.device)
            ret = eps.mul(std.unsqueeze(1) if isinstance(std, torch.Tensor) else std).add(mu.unsqueeze(1))
            return ret.reshape(-1, ret.size(2))
        else:
            return mu
    # Single sampling. (batch, nz)
    else:
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)


class LossWrapper(TensorWrapper):
    """
    Loss wrapper.
    """

    def __init__(self, _lmd, loss_tensor):
        super(LossWrapper, self).__init__(loss_tensor)
        # Lambda
        self._lmd = _lmd

    def loss_backprop(self):
        if self._lmd.hyper_param > 0.0 and self._tensor is not None:
            return self._lmd(self._tensor) * self._lmd.hyper_param
        else:
            return None


def summarize_losses_and_backward(*args, **kwargs):
    """
    Each arg should either be instance of
        - None
        - Tensor
        - LossWrapper
        - LossWrapperContainer
    """
    # 1. Init
    ret = 0.0
    # 2. Summarize to result
    for arg in args:
        if arg is None:
            continue
        elif isinstance(arg, LossWrapper):
            loss_backprop = arg.loss_backprop()
            if loss_backprop is not None: ret += loss_backprop
        elif isinstance(arg, torch.Tensor):
            ret += arg
        else:
            raise NotImplementedError
    # 3. Backward
    if isinstance(ret, torch.Tensor):
        ret.backward(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Visualizing disentangling
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def vis_grid_disentangling(batch_data, func_style, func_rec, gap_size, save_path):
    """
    Visualizing disentangling in grid.
    """
    images, class_label = batch_data
    # 1. Calculate reconstructions
    # (1) Encoded & get shape. (batch, style_dim)
    style_mu = func_style(images)
    batch, style_dim = style_mu.size()
    # (2) Mesh grid. (batch*batch, style_dim) & (batch*batch, class_dim)
    style_mu = style_mu.unsqueeze(1).expand(batch, batch, style_dim).reshape(-1, style_dim)
    class_label = class_label.unsqueeze(0).expand(batch, batch).reshape(-1, )
    # (3) Decode. (batch*batch, ...)
    recon = func_rec(style_mu, class_label)
    # 2. Get result
    recon = torch.reshape(recon, shape=(batch, batch, *recon.size()[1:]))
    recon = torch.cat([_.squeeze(1) for _ in torch.split(recon, split_size_or_sections=1, dim=1)], dim=3)
    recon = torch.cat([_.squeeze(0) for _ in torch.split(recon, split_size_or_sections=1, dim=0)], dim=1)
    # 1> Right
    hor_images = torch.cat([_.squeeze(0) for _ in torch.split(images, split_size_or_sections=1, dim=0)], dim=2)
    hor_gap = torch.ones(size=(hor_images.size(0), gap_size, hor_images.size(2)), device=hor_images.device)
    ret = torch.cat([hor_images, hor_gap, recon], dim=1)
    # 2> Left
    ver_images = torch.cat([_.squeeze(0) for _ in torch.split(images, split_size_or_sections=1, dim=0)], dim=1)
    ver_images = torch.cat([
        torch.zeros(size=(ver_images.size(0), images.size(2), images.size(3)), device=ver_images.device),
        torch.ones(size=(ver_images.size(0), gap_size, images.size(3)), device=ver_images.device),
        ver_images], dim=1)
    ver_gap = torch.ones(size=(ver_images.size(0), ver_images.size(1), gap_size), device=ver_images.device)
    ver_images = torch.cat([ver_images, ver_gap], dim=2)
    # Result
    ret = torch.cat([ver_images, ret], dim=2)
    # 3. Save
    save_image(ret.unsqueeze(0), save_path)


class DisenIB(IterativeBaseModel):
    """
    Disentangled IB model.
    """
    def _build_architectures(self, **modules):
        super(DisenIB, self)._build_architectures(
            # Encoder, decoder, reconstructor, estimator
            Enc_style=EncoderMNIST(self._cfg.args.style_dim), Enc_class=EncoderMNIST(self._cfg.args.class_dim),
            Dec=Decoder(self._cfg.args.class_dim, self._cfg.args.num_classes),
            Rec=ReconstructorMNIST(self._cfg.args.style_dim, self._cfg.args.class_dim, self._cfg.args.num_classes),
            Est=DensityEstimator(self._cfg.args.style_dim, self._cfg.args.class_dim),
            # Discriminator for improving generated quality
            Disc=DiscriminatorMNIST())

    def _set_criterions(self):
        self._criterions['dec'] = CrossEntropyLoss(lmd=self._cfg.args.lambda_dec)
        self._criterions['rec'] = RecLoss(lmd=self._cfg.args.lambda_rec)
        self._criterions['est'] = EstLoss(radius=self._cfg.args.emb_radius)
        # Discriminator
        self._criterions['disc'] = GANLoss()

    def _set_optimizers(self):
        self._optimizers['main'] = torch.optim.Adam(
            list(self._Enc_style.parameters()) + list(self._Enc_class.parameters()) +
            list(self._Dec.parameters()) + list(self._Rec.parameters()),
            lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))
        self._optimizers['est'] = torch.optim.Adam(
            self._Est.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))
        # Discriminator
        self._optimizers['disc'] = torch.optim.Adam(
            self._Disc.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.999))

    def _set_meters(self, **kwargs):
        super(DisenIB, self)._set_meters()
        self._meters['counter_eval'] = FreqCounter(self._cfg.args.freq_step_eval)
        self._meters['trigger_est'] = TriggerLambda(lambda n: n >= self._cfg.args.est_thr)
        self._meters['trigger_est_style_optimize'] = TriggerPeriod(
            period=self._cfg.args.est_style_optimize + 1, area=self._cfg.args.est_style_optimize)
        self._meters['trigger_disc'] = TriggerLambda(lambda n: n >= self._cfg.args.disc_thr)

    # ------------------------------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------------------------------

    def _deploy_batch_data(self, batch_data):
        image, label = map(lambda x: x.to(self._cfg.args.device), batch_data)
        return image.size(0), (image, label)

    def _train_step(self, packs):
        ################################################################################################################
        # Main
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_main):
            images, label = self._fetch_batch_data()
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=True)
            set_requires_grad([self._Disc, self._Est], requires_grad=False)
            self._optimizers['main'].zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            # Decoding & reconstruction
            # ----------------------------------------------------------------------------------------------------------
            # 1. Decoding
            style_emb, class_emb = self._Enc_style(images), self._Enc_class(images)
            dec_output = self._Dec(resampling(class_emb, self._cfg.args.class_std))
            loss_dec = self._criterions['dec'](dec_output, label)
            # 2. Reconstruction
            rec_output = self._Rec(resampling(style_emb, self._cfg.args.style_std), label)
            loss_rec = self._criterions['rec'](rec_output, images)
            # Backward
            summarize_losses_and_backward(loss_dec, loss_rec, retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Estimator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate output (batch*n_samples, ) & loss (1, ).
            est_output = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output=est_output, emb=(style_emb, class_emb), mode='main',
                lmd={'loss_est': self._cfg.args.lambda_est, 'loss_wall': self._cfg.args.lambda_wall})
            # Backward
            # 1> Density estimation
            if self._meters['trigger_est'].check(self._meters['i']['step']):
                if self._meters['trigger_est_style_optimize'].check():
                    set_requires_grad(self._Enc_class, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_class, requires_grad=True)
                else:
                    set_requires_grad(self._Enc_style, requires_grad=False)
                    summarize_losses_and_backward(crit_est['loss_est'], retain_graph=True)
                    set_requires_grad(self._Enc_style, requires_grad=True)
            # 2> Embedding wall
            summarize_losses_and_backward(crit_est['loss_wall'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Discriminator
            # ----------------------------------------------------------------------------------------------------------
            # Calculate loss
            disc_output = self._Disc(rec_output)
            crit_gen = self._criterions['disc'](disc_output, True, lmd=self._cfg.args.lambda_disc)
            # Backward
            if self._meters['trigger_disc'].check(self._meters['i']['step']):
                summarize_losses_and_backward(crit_gen['loss'], retain_graph=True)
            # ----------------------------------------------------------------------------------------------------------
            # Update
            self._optimizers['main'].step()
            """ Saving """
            packs['log'].update({
                # Decoding & reconstruction
                'loss_dec': loss_dec.item(), 'loss_rec': loss_rec.item(),
                # Estimator
                'loss_est_NO_DISPLAY': crit_est['loss_est'].item(), 'est': crit_est['est'].item()
            })
        ################################################################################################################
        # Density Estimator
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_est):
            with self._meters['timers']('io'):
                images, label = map(lambda _x: _x.to(self._cfg.args.device), next(self._data['train_est']))
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad([self._Est], requires_grad=True)
            self._optimizers['est'].zero_grad()
            # 1. Get embedding
            style_emb, class_emb = self._Enc_style(images).detach(), self._Enc_class(images).detach()
            # 2. Get output (batch*n_samples, ) & loss (1, ).
            est_output_real = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='perm')
            est_output_fake = self._Est(
                resampling(style_emb, self._cfg.args.est_style_std),
                resampling(class_emb, self._cfg.args.est_class_std), mode='orig')
            crit_est = self._criterions['est'](
                output_fake=est_output_fake, output_real=est_output_real, mode='est',
                lmd={'loss_real': 1.0, 'loss_fake': 1.0, 'loss_zc': self._cfg.args.lambda_est_zc})
            # Backward
            summarize_losses_and_backward(crit_est['loss_real'], crit_est['loss_fake'], crit_est['loss_zc'])
            # Update
            self._optimizers['est'].step()
            """ Saving """
            packs['log'].update({
                # Anchor
                'loss_est_real_NO_DISPLAY': crit_est['loss_real'].item(), 'est_real': crit_est['est_real'].item(),
                'loss_est_fake_NO_DISPLAY': crit_est['loss_fake'].item(), 'est_fake': crit_est['est_fake'].item()})
        ################################################################################################################
        # Discriminator
        ################################################################################################################
        for _ in range(self._cfg.args.n_times_disc):
            images, label = self._fetch_batch_data()
            # Clear grad
            set_requires_grad([self._Enc_style, self._Enc_class, self._Dec, self._Rec], requires_grad=False)
            set_requires_grad([self._Disc], requires_grad=True)
            self._optimizers['disc'].zero_grad()
            # 1. Get disc_output
            disc_output_real = self._Disc(images)
            style_emb = resampling(self._Enc_style(images), self._cfg.args.style_std)
            disc_output_fake = self._Disc(self._Rec(style_emb, label).detach())
            # 2. Calculate loss
            crit_disc_real = self._criterions['disc'](disc_output_real, True, lmd=1.0)
            crit_disc_fake = self._criterions['disc'](disc_output_fake, False, lmd=1.0)
            # Backward & save
            disc_acc = torch.cat([crit_disc_real['pred'] == 1, crit_disc_fake['pred'] == 0], dim=0).sum().item() / (
                    images.size(0) * 2)
            if disc_acc < self._cfg.args.disc_limit_acc:
                summarize_losses_and_backward(crit_disc_real['loss'], crit_disc_fake['loss'])
                self._optimizers['disc'].step()
            packs['log'].update({
                'loss_disc_real': crit_disc_real['loss'].item(), 'loss_disc_fake': crit_disc_fake['loss'].item(),
                'disc_acc': disc_acc})

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging
        self._process_log_after_step(packs)
        # 2. Evaluation
        if self._meters['counter_eval'].check(self._meters['i']['step']):
            vis_grid_disentangling(
                batch_data=map(lambda x: x[:self._cfg.args.eval_dis_n_samples], self._fetch_batch_data(no_record=True)),
                func_style=self._Enc_style, func_rec=self._Rec, gap_size=3,
                save_path=os.path.join(self._cfg.args.eval_dis_dir, 'step[%d].png' % self._meters['i']['step']))
        # 3. Chkpt
        self._process_chkpt_and_lr_after_step()
        # Clear packs
        packs['log'] = ValidContainer()

    def _process_log_after_step(self, packs, **kwargs):

        def _lmd_generate_log():
            r_tfboard = {
                'train/losses': fet_d(packs['log'], prefix='loss_', remove=('loss_', '_NO_DISPLAY')),
                'train/est': fet_d(packs['log'], prefix='est_')
            }
            packs['log'] = packs['log'].dict
            packs['tfboard'] = r_tfboard

        super(DisenIB, self)._process_log_after_step(
            packs, lmd_generate_log=_lmd_generate_log, lmd_process_log=Logger.reform_no_display_items)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    ################################################################################################################
    # Datasets
    ################################################################################################################
    parser.add_argument("--dataset_shuffle", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dataset_num_threads", type=int, default=0)
    parser.add_argument("--dataset_drop_last", type=bool, default=True)
    ################################################################################################################
    # Others
    ################################################################################################################
    parser.add_argument("--style_dim", type=int, default=16)
    parser.add_argument("--class_dim", type=int, default=16)
    parser.add_argument("--style_std", type=float, default=0.1)
    parser.add_argument("--class_std", type=float, default=1.0)
    parser.add_argument("--emb_radius", type=float, default=3.0)
    # Optimization & Lambda
    parser.add_argument("--n_times_main", type=int, default=10)
    parser.add_argument("--n_times_est", type=int, default=1)
    parser.add_argument("--n_times_disc", type=int, default=1)
    parser.add_argument("--disc_thr", type=int, default=1000)
    parser.add_argument("--disc_limit_acc", type=float, default=0.8)
    parser.add_argument("--est_thr", type=int, default=3000)
    parser.add_argument("--est_batch_size", type=int, default=64)
    parser.add_argument("--est_style_std", type=float, default=0.1)
    parser.add_argument("--est_class_std", type=float, default=0.1)
    parser.add_argument("--est_style_optimize", type=int, default=4)
    parser.add_argument("--lambda_dec", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=10.0)
    parser.add_argument("--lambda_est", type=float, default=0.5)
    parser.add_argument("--lambda_est_zc", type=float, default=0.05)
    parser.add_argument("--lambda_wall", type=float, default=10.0)
    parser.add_argument("--lambda_disc", type=float, default=0.1)
    # Evaluating args
    parser.add_argument("--freq_step_eval", type=int, default=500)
    parser.add_argument("--eval_dis_n_samples", type=int, default=10)

    # Epochs & batch size
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    # Learning rate
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    # Frequency
    parser.add_argument("--freq_iter_log", type=int, default=4096)
    parser.add_argument("--freq_step_chkpt", type=int, default=1000)

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 1. Generate config
    cfg = get_args()
    # 2. Generate model & dataloader
    dataloader = {
        "train_data": DataCycle(DataLoader(dataset=ImageMNIST('train', cfg.dataset, transforms=ToTensor()))),
        "train_est_data": DataCycle(DataLoader(dataset=ImageMNIST('train', cfg.dataset, transforms=ToTensor()),
                                               batch_size=cfg.args.est_batch_size))
    }
    model = DisenIB(cfg=cfg)
    # 3. Train
    model.train_parameters(**dataloader)
