#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/9/30 21:15
# @Author: ZhaoKe
# @File : neuctmdata.py
# @Software: PyCharm
import os
import numpy as np
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
import torch
from audiokits.featurizer import Wave2Mel
from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms
# import seaborn as sns
# import pyecharts

neuctm_path = "F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/neucough_metainfo.txt"


def min2sec(t: str):
    parts = t.split(':')
    return int(parts[0]) * 60 + float(parts[1])


def build_csv():
    filename_list = []
    st_list = []
    en_list = []
    with open(neuctm_path) as fin:
        fin.readline()
        line = fin.readline()
        while line:
            line = line.strip().split(',')
            if len(line) > 2:
                # print(line)
                # print("{}_audiodata_元音字母a.wav".format(line[0]))
                filename_list.append("F:/DATAS/NEUCOUGHDATA_FULL/{}_audiodata_元音字母a.wav".format(line[0]))
                # print(min2sec(line[1]), min2sec(line[2]))
                st_list.append(min2sec(line[1]))
                en_list.append(min2sec(line[2]))
            line = fin.readline()
            # print("continue.")
    return filename_list, st_list, en_list


def read_audio(filepath: str, w2m, st=None, en=None):
    # print(y.shape, sr)
    sr = 22050
    if (st is not None) and (en is not None):
        # st, en = int(st * sr), int(en * sr)
        print("st, en:", st, en)
        y, sr = librosa.load(filepath, offset=st, duration=en - st)
        print("y, sr:", len(y), sr)
    else:
        y, sr = librosa.load(filepath, )
        print("y, sr:", y, sr)
    print(y.shape)
    mel = w2m(torch.from_numpy(y))
    # return mel
    # print(mel.transpose(0, 1).shape)
    # plt.figure(0)
    # plt.subplot(2, 1, 1)
    # plt.plot(y)
    # plt.subplot(2, 1, 2)
    # plt.imshow(mel.numpy().astype(np.uint8))
    # plt.show()
    # return y, mel.transpose(0, 1)


class NEUCOUGHDataset(Dataset):
    def __init__(self, meta_file="F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/neucough_metainfo.txt",
                 wav_path="F:/DATAS/NEUCOUGHDATA_FULL/"):
        # w2m = Wave2Mel(sr=22050)
        # filename_list = []
        self.wav_list = []
        self.st_list = []
        self.en_list = []
        duration_list = []
        fout = open("F:/DATAS/NEUCOUGHDATA_COUGHSINGLE/neucough_metainfo_slice.txt", 'w')
        with open(meta_file) as fin:
            lines = fin.readlines()[1:]
            fout.write(
                "fileid,st,en,gender,issmoking,isalcohol,iscough,isfever,isrespill,iscovid19,q10,q15,q21,q26,q31,q38,q60,q61,q62,q63,q64,duration,slice\n")
            for line in tqdm(lines, desc="Load Dataset:"):
                line = line.strip().split(',')
                if len(line) > 2:
                    # print(line)
                    # print("{}_audiodata_元音字母a.wav".format(line[0]))
                    # filename_list.append(wav_path+"{}_audiodata_元音字母a.wav".format(line[0]))
                    # print(min2sec(line[1]), min2sec(line[2]))
                    y, sr = librosa.load(wav_path + "{}_audiodata_元音字母a.wav".format(line[0]))
                    # if (st is not None) and (en is not None):
                    #     st, en = int(st * sr), int(en * sr)
                    #     y = y[st:en]
                    assert sr == 22050, "Sample Rate Error, {}.".format(sr)
                    duration_list.append(len(y) / sr)
                    self.wav_list.append(torch.from_numpy(y))
                    self.st_list.append(min2sec(line[1]))
                    self.en_list.append(min2sec(line[2]))
                    fout.write(
                        ",".join(line) + "," + str(len(y) / sr)[:8] + "," + str(min2sec(line[2]) - min2sec(line[1]))[:5] + "\n")
                # line = fin.readline()
        fout.close()

    def __getitem__(self, ind):
        return self.wav_list[ind], self.st_list[ind], self.en_list[ind]

    def __len__(self):
        return len(self.st_list)


def main():
    wav2mel = Wave2Mel(sr=22050)
    filenames, starts, ends = build_csv()
    read_audio(filenames[0], w2m=wav2mel, st=starts[0], en=ends[0])


if __name__ == '__main__':
    neucough = NEUCOUGHDataset()
    print(len(neucough))
