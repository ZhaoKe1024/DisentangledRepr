#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/11 15:58
# @Author: ZhaoKe
# @File : mnist_rot_create.py
# @Software: PyCharm
# The processing method is referred to: https://github.com/ChaitanyaBaweja/RotNIST
import gzip
import os
import numpy as np
from scipy import ndimage

# Params for MNIST

VALIDATION_SIZE = 5000  # Size of the validation set.

'''
Extract images from given file path into a 4D tensor [image index, y, x, channels].
Values are rescaled from [0, 255] down to [-0.5, 0.5].
filename: filepath to images
num: number of images
60000 in case of training
10000 in case of testing
Returns numpy vector
'''


def extract_data(filename, num):
    print('Extracting', filename)
    # unzip data
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num * 1)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (255 / 2.0)) / 255  # rescaling value to [-0.5,0.5]
        data = data.reshape(num, 28, 28, 1)  # reshape into tensor
        data = np.reshape(data, [num, -1])
    return data


'''
Extract the labels into a vector of int64 label IDs.
filename: filepath to labels
num: number of labels
60000 in case of training
10000 in case of testing
Returns numpy vector
'''


def extract_labels(filename, num):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data, 10))
        one_hot_encoding[np.arange(num_labels_data), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
    return one_hot_encoding


'''
Augment training data with rotated digits
images: training images
labels: training labels
'''


def expand_training_data(images, labels, mode='train'):
    expanded_images = []
    expanded_labels = []
    expanded_attris = []

    # directory = os.path.dirname("data/New")
    # if not tf.gfile.Exists("data/New"):
    #     tf.gfile.MakeDirs("data/New")
    k = 0  # counter
    angles = [-45.0, -22.5, 22.5, 45.0]
    angle_label = [4, 2, 1, 3]
    for x, y in zip(images, labels):
        k = k + 1
        if k % 100 == 0:
            print('expanding data : %03d / %03d' % (k, np.size(images, 0)))
        y = np.argmax(y)
        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)
        expanded_attris.append(0)

        bg_value = -0.5  # this is regarded as background's value black

        image = np.reshape(x, (-1, 28))

        for i in range(len(angles)):
            # rotate the image with random degree
            new_img = ndimage.rotate(image, angle=angles[i], reshape=False, cval=bg_value)

            '''
            if k<50:
                NAME1 = DATA_DIRECTORY+"/New"+"/"+str(k)+"_0.jpeg"
                im = Image.fromarray(image1)
                im.convert('RGB').save(NAME1)
                im = Image.fromarray(new_img1)
                NAME = DATA_DIRECTORY+"/New"+"/"+str(k)+"_"+str(i+1)+".jpeg"
                im.convert('RGB').save(NAME)
            '''

            # register new training data
            expanded_images.append(np.reshape(new_img, 784))
            expanded_labels.append(y)
            expanded_attris.append(angle_label[i])

    if mode == 'train':
        np.save("F:/DATAS/mnist/MNIST-ROT/train_data.npy", expanded_images)
        np.save("F:/DATAS/mnist/MNIST-ROT/train_labels.npy", expanded_labels)
        np.save("F:/DATAS/mnist/MNIST-ROT/train_sensitive_labels.npy", expanded_attris)
        print("Save successfully train data!")
    else:
        np.save("F:/DATAS/mnist/MNIST-ROT/test_data.npy", expanded_images)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_labels.npy", expanded_labels)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_sensitive_labels.npy", expanded_attris)
        print("Save successfully test data!")

        expanded_images_55 = []
        expanded_labels_55 = []
        expanded_attris_55 = []

        expanded_images_65 = []
        expanded_labels_65 = []
        expanded_attris_65 = []
        # directory = os.path.dirname("data/New")
        # if not tf.gfile.Exists("data/New"):
        #     tf.gfile.MakeDirs("data/New")
        k = 0  # counter
        for x, y in zip(images, labels):
            k = k + 1
            if k % 100 == 0:
                print('expanding data : %03d / %03d' % (k, np.size(images, 0)))

            y = np.argmax(y)
            bg_value = -0.5  # this is regarded as background's value black

            image = np.reshape(x, (-1, 28))
            new_img = ndimage.rotate(image, angle=55, reshape=False, cval=bg_value)

            expanded_images_55.append(np.reshape(new_img, 784))
            expanded_labels_55.append(y)
            expanded_attris_55.append(5)

            new_img = ndimage.rotate(image, angle=-55, reshape=False, cval=bg_value)
            expanded_images_55.append(np.reshape(new_img, 784))
            expanded_labels_55.append(y)
            expanded_attris_55.append(6)

            new_img = ndimage.rotate(image, angle=65, reshape=False, cval=bg_value)
            expanded_images_65.append(np.reshape(new_img, 784))
            expanded_labels_65.append(y)
            expanded_attris_65.append(7)

            new_img = ndimage.rotate(image, angle=-65, reshape=False, cval=bg_value)
            expanded_images_65.append(np.reshape(new_img, 784))
            expanded_labels_65.append(y)
            expanded_attris_65.append(8)

        # images and labels are concatenated for random-shuffle at each epoch
        # notice that pair of image and label should not be broken
        # expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
        # np.random.shuffle(expanded_train_total_data)

        np.save("F:/DATAS/mnist/MNIST-ROT/test_55_data.npy", expanded_images_55)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_55_labels.npy", expanded_labels_55)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_55_sensitive_labels.npy", expanded_attris_55)
        print("Save successfully train 55 data!")

        np.save("F:/DATAS/mnist/MNIST-ROT/test_65_data.npy", expanded_images_65)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_65_labels.npy", expanded_labels_65)
        np.save("F:/DATAS/mnist/MNIST-ROT/test_65_sensitive_labels.npy", expanded_attris_65)
        print("Save successfully train 65 data!")
    # return expanded_images, expanded_labels, expanded_attris


'''
Main function to prepare the entire data
'''


def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    # train_data_filename = download('train-images-idx3-ubyte.gz')
    # train_labels_filename = download('train-labels-idx1-ubyte.gz')
    # test_data_filename = download('t10k-images-idx3-ubyte.gz')
    # test_labels_filename = download('t10k-labels-idx1-ubyte.gz')
    train_data_filename = os.path.join("F:/DATAS/mnist", 'train-images-idx3-ubyte.gz')
    train_labels_filename = os.path.join("F:/DATAS/mnist", 'train-labels-idx1-ubyte.gz')

    test_data_filename = os.path.join("F:/DATAS/mnist", 't10k-images-idx3-ubyte.gz')
    test_labels_filename = os.path.join("F:/DATAS/mnist", 't10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)

    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    if use_data_augmentation:
        expand_training_data(train_data, train_labels, 'train')
        expand_training_data(test_data, test_labels, 'test')
    else:
        np.concatenate((train_data, train_labels), axis=1)


#     if use_data_augmentation:
#         train_data, train_label, train_attri = expand_training_data(train_data, train_labels)
#         test_data, test_labels, test_attri = expand_training_data(test_data, test_labels)
#     else:
#         train_total_data = np.concatenate((train_data, train_labels), axis=1)
#
#     # Generate a validation set.
#     print(train_data.shape)
#     train_data = train_data[VALIDATION_SIZE:, :]
#     train_labels = train_labels[VALIDATION_SIZE:, :]
#     validation_data = train_data[:VALIDATION_SIZE, :]
#     validation_labels = train_labels[:VALIDATION_SIZE, :]
#
#     # Concatenate train_data & train_labels for random shuffle
#
#     train_size = train_total_data.shape[0]
#
#     return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
#
#
if __name__ == '__main__':
    # train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = prepare_MNIST_data(True)
    # prepare_MNIST_data(True)
    train_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'train_data.npy'))
    train_label = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'train_labels.npy'))
    train_angle = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'train_sensitive_labels.npy'))
    print(train_data.shape, train_label.shape, train_angle.shape)
    test_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_55_data.npy'))
    test_55_label = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_55_labels.npy'))
    test_55_attri = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_55_sensitive_labels.npy'))
    print(test_data.shape, test_55_label.shape, test_55_attri.shape)

    test_65_data = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_65_data.npy'))
    test_65_label = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_65_labels.npy'))
    test_65_attri = np.load(os.path.join("F:/DATAS/mnist/MNIST-ROT/", 'test_65_sensitive_labels.npy'))
    print(test_65_data.shape, test_65_label.shape, test_65_attri.shape)
    # print(train_data[0])
