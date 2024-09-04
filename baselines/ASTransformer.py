#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/22 15:25
# @Author: ZhaoKe
# @File : ASTransformer.py
# @Software: PyCharm
import numpy as np
import yaml
import collections
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import DropPath


class AST(nn.Module):
    """ Create an Audio Spectrogram model from building blocks.
    From AST paper: https://arxiv.org/pdf/2104.01778.pdf

    First, the input audio waveform of t seconds is converted into a sequence of 128-dimensional log Mel filterbank (fbank)
    features computed with a 25ms Hamming window every 10ms. This results in a 128×100t spectrogram as input to the AST.
    We then split the spectrogram into a sequence of N 16×16 patches with an overlap of 6 in both time and frequency dimension,
    where N = 12[(100t − 16)/10] is the number of patches and the effective input sequence length for the Transformer.
    We flatten each 16×16 patch to a 1D patch embedding of size 768 using a linear projection layer.

    Following the paper, we:
      * Take our input spectrograms
      * Perform patch split with overlap
      * Project linearly and encode
      * Each patch embedding added with learnable position embedding
      * CLS classification token prepended to sequence
      * Output of CLS token used for classification with linear layer
      * Transformer encoder: multiple attention heads, variable depth
      * Final output of model.
    """

    def __init__(self, input_fdim=128, input_tdim=1000, patch_size=16, embed_dim=768, drop_rate=0.1, fstride=10,
                 tstride=10,
                 n_classes=50, config=None):
        super(AST, self).__init__()
        self.config = config
        if config.debug:
            print('In super init')
        # Patch embeddings based on
        # https://github.com/rwightman/pytorch-image-models/blob/fa8c84eede55b36861460cc8ee6ac201c068df4d/timm/models/layers/patch_embed.py#L15

        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=(fstride, tstride))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.num_patches = num_patches
        if config.debug:
            print('self num patches here', self.num_patches)

        # positional embedding
        self.embed_dim = embed_dim
        embed_len = self.num_patches

        # Classifier token definition
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        if config.debug:
            print('self.pos_embed in init', np.shape(self.pos_embed))
        self.original_embedding_dim = self.pos_embed.shape[2]
        if config.debug:
            print('original embedding dim', self.original_embedding_dim)

        if config.debug:
            print('frequncy stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        # new_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, self.original_embedding_dim))
        new_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.original_embedding_dim))
        self.pos_embed = new_pos_embed
        # nn.init.trunc_normal_(new_pos_embed, std=.02)
        if config.debug:
            print('drop rate', drop_rate)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder blocks:
        self.transformer = TransformerBlocks(config=config)
        # Final linear layer
        self.FinalLinear = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                         nn.Linear(self.original_embedding_dim, config.n_classes))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = self.proj
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):

        x = x.unsqueeze(1)
        if self.config.debug:
            print('x unsqueezed', np.shape(x))
        x = x.transpose(2, 3)
        if self.config.debug:
            print('x after transpose', np.shape(x))
        B = x.shape[0]  # batch

        x = self.proj(x).flatten(2).transpose(1, 2)  # Linear projection of 1D patch embedding
        if self.config.debug:
            print('x shape after linear proj', np.shape(x))
            print('Shape for token', np.shape(self.cls_token))
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.config.debug:
            print('shape tokens', np.shape(cls_tokens))
            print('value of cls_tokens', cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.config.debug:
            print('x after torch cat', np.shape(x))
            print('self.pos_embed dims', np.shape(self.pos_embed))
        x = x + self.pos_embed
        if self.config.debug:
            print('Pos embed value', self.pos_embed)
        x = self.pos_drop(x)

        # Transformer encoder here
        for block in self.transformer.blocks:
            x = block(x)

        x = (x[:, 0] + x[:, 1]) / 2

        # Final linear layer

        x = F.relu(self.FinalLinear(x))

        return x


# Helper modules:

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _ntuple(2)(bias)
        drop_probs = _ntuple(2)(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#
#     def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep
#
#     def forward(self, x):
#         # return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
#         return drop_path(x, self.drop_prob, self.training)
#
#     def extra_repr(self):
#         return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print('In forward of attention module')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=None,
            attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TransformerBlocks(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(
            dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=config.dropout, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for _ in range(config.depth)])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# class Dict(dict):
#     __setattr__ = dict.__setitem__
#     __getattr__ = dict.__getitem__
#
#
# def dict_to_object(dict_obj):
#     if not isinstance(dict_obj, dict):
#         return dict_obj
#     inst = Dict()
#     for k, v in dict_obj.items():
#         inst[k] = dict_to_object(v)
#     return inst
#
#
# if __name__ == '__main__':
#     with open("../../configs/astransformer.yaml", 'r') as jsf:
#         cfg = yaml.load(jsf.read(), Loader=yaml.FullLoader)
#         cfg = dict_to_object(cfg)
#     trans = AST(config=cfg)
#     print(trans)
#     x = torch.randn(2, 64, 64)
#     print(trans(x))
