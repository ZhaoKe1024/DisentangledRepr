#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/8/29 9:40
# @Author: ZhaoKe
# @File : signalprocess.py
# @Software: PyCharm
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import librosa
import torch
from audiokits.featurizer import Wave2Mel


def get_signal_spec(wav_path, w2m):
    y, sr = librosa.load(wav_path, sr=44010)
    print(y.shape, sr)
    mel = w2m(torch.from_numpy(y))
    print(mel.transpose(0, 1).shape)
    return y, mel.transpose(0, 1)


def show_mel():
    # wav_path = "../datasets/sound0000_de543d13-541c-4ad7-bb3c-c5c302de3aaf.wav"
    wav2mel = Wave2Mel(sr=22050)
    data_root = "F:/DATAS/COUGHVID-public_dataset_v3/coughvid_20211012_fine/"
    plt.figure(0)
    for j, file_item in enumerate(os.listdir(data_root)[:9]):
        file_path = os.path.join(data_root, file_item)
        _, mel = get_signal_spec(file_path, wav2mel)
        plt.subplot(3, 3, j + 1)
        plt.imshow(mel.data.numpy().astype(np.uint8))
    plt.show()


def show_coughvid_pkl():
    coughvid_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_split_specdf.pkl")
    cough_mels = coughvid_df["melspectrogram"]
    print("number of data:", cough_mels.shape)
    print("shape of data:", cough_mels[0].shape)
    print()
    # plt.figure(0)
    # for i in range(9):
    #     plt.subplot(3, 3, i + 1)
    #     new_mel = np.zeros(shape=(128, 128))
    #     new_mel[:, :64] = cough_mels[i]
    #     new_mel[:, 64:] = cough_mels[i]
    #     plt.imshow(new_mel.astype(np.uint8))
    # plt.show()


def show_signal_spec():
    # filepath = "F:/DATAS/COUGHVID-public_dataset_v3/coughvid_20211012/ff958d96-b7eb-42f2-8a02-59eebf35668b.wav"
    # # filepath = "F:/DATAS/COUGHVID-public_dataset_v3/coughvid_20211012/ffdc1fbe-ae22-4488-ad01-307edd2912ed.webm"
    filepath = "../datasets/sound0000_de543d13-541c-4ad7-bb3c-c5c302de3aaf.wav"
    wav2mel = Wave2Mel(sr=44010)
    sig, spec = get_signal_spec(filepath, wav2mel)
    print(sig.shape)
    print(spec.shape)
    # plt.figure(1)
    # plt.plot(range(len(sig)), sig, c="black")
    # plt.show()
    # plt.figure(0)
    # plt.imshow(spec.transpose(0, 1).data.numpy().astype(np.uint8))
    # plt.show()


if __name__ == '__main__':
    # show_coughvid_pkl()
    show_signal_spec()
