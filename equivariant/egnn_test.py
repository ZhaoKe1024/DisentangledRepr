#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/10/18 16:23
# @Author: ZhaoKe
# @File : egnn_test.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit
from torch.autograd import Variable



def check_equivariance(im, layers, input_array, output_array, point_group):
    # Transform the image
    f = input_array(im)
    g = point_group.rand()
    gf = g * f
    im1 = gf.v

    # Apply layers to both images
    im = Variable(torch.from_numpy(im))
    im1 = Variable(torch.from_numpy(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        print(layer)
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(fmap1.data.numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = fmap.data.numpy()
    assert np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)
