#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/10 18:19
# @Author: ZhaoKe
# @File : dataloader_mnist.py
# @Software: PyCharm
import os
import struct
import numpy as np
import torchvision
from torch.utils.data import Dataset

########################################################################################################################
# Data
########################################################################################################################
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    :param idx3_ubyte_file:
    :return:
    """
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # File header
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # Dataset
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    :param idx1_ubyte_file:
    :return:
    """
    # Load bin data
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # File header
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # Dataset
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


class DataCycle(object):
    """
    Data cycle infinitely. Using next(self) to fetch batch data.
    """

    def __init__(self, dataloader):
        # Dataloader
        self._dataloader = dataloader
        # Iterator
        self._data_iterator = iter(self._dataloader)

    @property
    def num_samples(self):
        return len(self._dataloader.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self._dataloader)
            return next(self._data_iterator)


class BaseClassification(Dataset):
    """
    Base class for dataset for classification.
    """

    @property
    def classes(self):
        """
        :return: A list, whose i-th element is the name of the i-th category.
        """
        raise NotImplementedError

    @property
    def class_to_idx(self):
        """
        :return: A dict, where dict[key] is the category index corresponding to the category 'key'.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        :return: A integer indicating number of categories.
        """
        raise NotImplementedError

    @property
    def class_counter(self):
        """
        :return: A list, whose i-th element equals to the total sample number of the i-th category.
        """
        raise NotImplementedError

    @property
    def sample_indices(self):
        """
        :return: A list, whose i-th element is a numpy.array containing sample indices of the i-th category.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Should return x, y, where y is the class label.
        :param index:
        :return:
        """
        raise NotImplementedError


class MNIST(BaseClassification):
    """
    MNIST dataset.
    """

    def __init__(self, images_path, labels_path, transforms=None):
        # Member variables
        self._transforms = transforms
        # Load from file
        # (1) Data & label
        self._dataset = decode_idx3_ubyte(images_path).astype('float32')[:, :, :, np.newaxis] / 255.0
        self._label = decode_idx1_ubyte(labels_path).astype('int64')
        # (2) Samples per category
        self._sample_indices = [np.argwhere(self._label == label)[:, 0].tolist() for label in range(self.num_classes)]
        # (3) Class counter
        self._class_counter = [len(samples) for samples in self._sample_indices]

    @property
    def num_classes(self):
        return len(set(self._label))

    @property
    def class_counter(self):
        return self._class_counter

    @property
    def sample_indices(self):
        return self._sample_indices

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        # 1. Get image & label
        image, label = self._dataset[index], self._label[index]
        # 2. Transform
        if self._transforms:
            image = self._transforms(image)
        # Return
        return image, label


__DATA_ROOT__ = "F:/DATAS/"


def mnist_paths(name):
    assert name in ['mnist', 'fashion_mnist']
    return {
        'train': (os.path.join(__DATA_ROOT__, "%s/train-images.idx3-ubyte" % name),
                  os.path.join(__DATA_ROOT__, "%s/train-labels.idx1-ubyte" % name)),
        'test': (os.path.join(__DATA_ROOT__, "%s/t10k-images.idx3-ubyte" % name),
                 os.path.join(__DATA_ROOT__, "%s/t10k-labels.idx1-ubyte" % name))}


class To32x32(object):
    """
    From 28x28 -> 32x32.
    """

    def __call__(self, x):
        assert x.shape == (28, 28, 1)
        x = np.concatenate([np.zeros(shape=(28, 2, 1), dtype=np.float32), x,
                            np.zeros(shape=(28, 2, 1), dtype=np.float32)], axis=1)
        x = np.concatenate([np.zeros(shape=(2, 32, 1), dtype=np.float32), x,
                            np.zeros(shape=(2, 32, 1), dtype=np.float32)], axis=0)
        return x


class ImageMNIST(MNIST):
    """
    Flatten MNIST dataset.
    """

    def __init__(self, phase, name='mnist', **kwargs):
        assert phase in ['train', 'test']
        # Get transforms
        if 'transforms' in kwargs.keys():
            transforms = kwargs['transforms']
        else:
            transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
            if 'to32x32' in kwargs.keys() and kwargs['to32x32']: transforms = [To32x32()] + transforms
            transforms = torchvision.transforms.Compose(transforms)
        # Init
        super(ImageMNIST, self).__init__(*(mnist_paths(name)[phase]), transforms=transforms)