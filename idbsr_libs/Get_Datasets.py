#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/11 15:08
# @Author: ZhaoKe
# @File : Get_Datasets.py
# @Software: PyCharm
import os
from idbsr_libs.Dataset import MNIST_ROT, MNIST_DIL, MNIST_ROT_VIS
from torch.utils.data import DataLoader


def get_datasets(dataset, train_batch_size, test_batch_size, cuda=False, root='Data'):
    print(f'Loading {dataset} dataset...')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if dataset == 'mnist-rot':
        Dataset = MNIST_ROT
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        train_loader = DataLoader(dataset.train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_55_loader = DataLoader(dataset.test_55_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_65_loader = DataLoader(dataset.test_65_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return train_loader, test_loader, test_55_loader, test_65_loader

    elif dataset == 'mnist-dil':
        Dataset = MNIST_DIL
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        test_erode_2_loader = DataLoader(dataset.test_erode_2_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_2_loader = DataLoader(dataset.test_dilate_2_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_3_loader = DataLoader(dataset.test_dilate_3_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        test_dilate_4_loader = DataLoader(dataset.test_dilate_4_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return test_erode_2_loader, test_dilate_2_loader, test_dilate_3_loader, test_dilate_4_loader

    elif dataset == 'mnist-rot-vis':
        Dataset = MNIST_ROT_VIS
        dataset_path = os.path.join(root, 'mnist')
        dataset = Dataset(dataset_path)
        test_loader = DataLoader(dataset.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        print('Done!\n')
        return test_loader

    else:
        raise ValueError('Dataset not supported')
