#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/16 12:24
# @Author: ZhaoKe
# @File : cirl_datasets.py
# @Software: PyCharm
import os
import bisect
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cirl_libs.data_utils import *


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


default_input_dir = 'path/to/datalists/'


def get_datalists_folder(args=None):
    datalists_folder = default_input_dir
    if args is not None:
        if args.input_dir is not None:
            datalists_folder = args.input_dir
    return datalists_folder


class FourierDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None, from_domain=None, alpha=1.0):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        self.alpha = alpha

        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s, label_s, domain_s = self.sample_image(domain)
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [label, label_s, label, label_s]
        domain = [domain, domain_s, domain, domain_s]
        return img, label, domain

    def sample_image(self, domain):
        if self.from_domain == 'all':
            domain_idx = random.randint(0, len(self.names) - 1)
        elif self.from_domain == 'inter':
            domains = list(range(len(self.names)))
            domains.remove(domain)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra':
            domain_idx = domain
        else:
            raise ValueError("Not implemented")
        img_idx = random.randint(0, len(self.names[domain_idx]) - 1)
        img_name_sampled = self.names[domain_idx][img_idx]
        img_name_sampled = os.path.join(self.args.input_dir, img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx


def get_fourier_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        from_domain = config["from_domain"]
        alpha = config["alpha"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return FourierDGDataset(args, names, labels, img_transform, from_domain, alpha)


def get_fourier_train_dataloader(source_list=None, batch_size=64, image_size=224, crop=False,
        jitter=0, args=None, from_domain='all', alpha=1.0, config=None
):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)

    paths = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        paths.append(path)
    dataset = get_fourier_dataset(args=args,
                                  path=paths,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader


class DGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return img, label


def get_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return DGDataset(args, names, labels, img_transform)


def get_val_dataloader(source_list=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_val.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader


def get_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_test.txt' % target)
    test_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
    dataset = ConcatDataset([test_dataset])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader
