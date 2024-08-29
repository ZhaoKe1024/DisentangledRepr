#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:04
# @Author: ZhaoKe
# @File : featurizer.py
# @Software: PyCharm
import random
import numpy as np
import librosa
import torch
import torchaudio
from torch import nn
import torchaudio.compliance.kaldi as Kaldi
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC
from audiokits.audio import AudioSegment


def get_wav_label(filename):
    audioseg = AudioSegment.from_file(file=filename)
    audioseg.vad(top_db=40)
    audioseg.resample(target_sample_rate=16000)
    audioseg.crop(duration=3.0, mode="train")
    audioseg.wav_padding()
    return audioseg.samples


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def wav_slice_padding(old_signal, save_len=160000):
    new_signal = np.zeros(save_len)
    if old_signal.shape[0] < save_len:
        resi = save_len - old_signal.shape[0]
        # print("resi:", resi)
        new_signal[:old_signal.shape[0]] = old_signal
        new_signal[old_signal.shape[0]:] = old_signal[-resi:][::-1]
    elif old_signal.shape[0] > save_len:
        posi = random.randint(0, old_signal.shape[0] - save_len)
        new_signal = old_signal[posi:posi + save_len]
    return new_signal


def get_a_wavmel_sample(test_wav_path):
    # test_wav_path = f"G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_{mt}/{mt}/train/normal_id_00_00000016.wav"
    y, sr = librosa.core.load(test_wav_path, sr=16000)
    y = wav_slice_padding(y, 147000)
    w2m = Wave2Mel(16000)
    x_mel = w2m(torch.from_numpy(y.T))
    x_input = x_mel.transpose(0, 1).unsqueeze(0).unsqueeze(0).to(torch.device("cuda"))
    return y, x_input


class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='MelSpectrogram', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**method_args)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(**method_args)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**method_args)
        elif feature_method == 'Fbank':
            self.feat_fun = KaldiFbank(**method_args)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def forward(self, waveforms, input_lens_ratio):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        # 归一化
        feature = feature - feature.mean(1, keepdim=True)
        # 对掩码比例进行扩展
        input_lens = (input_lens_ratio * feature.shape[1])
        mask_lens = torch.round(input_lens).long()
        mask_lens = mask_lens.unsqueeze(1)  # [16, 1]
        # print(mask_lens)
        input_lens = input_lens.int()
        # 生成掩码张量
        idxs_start = (feature.shape[1] - input_lens) // 2
        idxs_start = torch.tensor(idxs_start).unsqueeze(1)
        # print(idxs_start)  # [16, 1]
        idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
        # print(idxs.shape)  # [16, 321]
        # print(idxs_start < idxs < (idxs_start + mask_lens))  # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
        mask = (idxs > idxs_start) * (idxs < (idxs_start + mask_lens))
        mask = mask.unsqueeze(-1)
        # 对特征进行掩码操作
        feature_masked = torch.where(mask, feature, torch.zeros_like(feature))
        return feature_masked, input_lens
        # feature_masked [64, 40, 127]

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'MelSpectrogram':
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 400) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Length, Feature]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank


class SpecAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def freq_mask(self, x):
        batch, _, fea = x.shape
        mask_len = torch.randint(self.freq_mask_width[0], self.freq_mask_width[1], (batch, 1),
                                 device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, fea - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(fea, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(1)
        x = x.masked_fill_(mask, 0.0)
        return x

    def time_mask(self, x):
        batch, time, _ = x.shape
        mask_len = torch.randint(self.time_mask_width[0], self.time_mask_width[1], (batch, 1),
                                 device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, time - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(time, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2)
        x = x.masked_fill_(mask, 0.0)
        return x

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x
