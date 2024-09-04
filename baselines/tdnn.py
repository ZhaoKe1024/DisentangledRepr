#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/17 12:14
# @Author: ZhaoKe
# @File : tdnn.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

from ackit.modules.pooling import AttentiveStatsPool, SelfAttentivePooling, TemporalAveragePooling, \
    TemporalStatisticsPooling


class TDNN(nn.Module):
    def __init__(self, num_class, input_size=80, channels=512, embd_dim=192, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.emb_size = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.td_layer2 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.fc = nn.Linear(embd_dim, num_class)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        # x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        out = self.fc(out)
        # out = self.softmax(out)  # (batch_size, class_pred)
        return out, x


class TDNN_Extractor(nn.Module):
    def __init__(self, mel_dim=80, hidden_size=512, channels=512):
        super(TDNN_Extractor, self).__init__()
        # self.emb_size = embd_dim
        self.wav2mel = nn.Conv1d(in_channels=1, out_channels=mel_dim, kernel_size=1024, stride=488,
                                 padding=1024 // 2, bias=False)
        # self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.td_layer1 = torch.nn.Conv1d(in_channels=mel_dim, out_channels=hidden_size, dilation=1, kernel_size=5,
                                         stride=1)  # IW-5+1
        self.bn1 = nn.LayerNorm(302)
        self.td_layer2 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=2, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-4+1
        self.bn2 = nn.LayerNorm(298)
        self.td_layer3 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=3, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-6+1
        self.bn3 = nn.LayerNorm(294)
        self.td_layer4 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        self.bn4 = nn.LayerNorm(288)
        self.td_layer5 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=channels, dilation=1, kernel_size=1,
                                         stride=1, groups=channels)  # IW+1
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, waveform):
        # x = x.transpose(2, 1)
        x = self.leakyrelu(self.bn1(self.wav2mel(waveform)))
        # print("shape of x as a wave:", x.shape)
        x = self.leakyrelu(self.bn2(self.td_layer1(x)))
        # print("shape of x in layer 1:", x.shape)
        x = self.leakyrelu(self.bn3(self.td_layer2(x)))
        # print("shape of x in layer 2:", x.shape)
        x = self.leakyrelu(self.bn4(self.td_layer3(x)))
        # print("shape of x in layer 3:", x.shape)
        x = self.leakyrelu(self.bn4(self.td_layer4(x)))
        # print("shape of x in layer 4:", x.shape)
        x = self.td_layer5(x)
        # print("shape of x in layer 5:", x.shape)
        return x


if __name__ == '__main__':
    # input_wav = torch.rand(16, 1, 48000)
    from ackit.modules.loss import FocalLoss
    input_wav = torch.rand(16, 80, 94)
    tdnn_model = TDNN(num_class=3, input_size=80, )
    pred, feat = tdnn_model(input_wav)
    loss_fn = FocalLoss(class_num=3)
    print("feat shape:", feat.shape)
    gt = torch.randint(0, 3, size=(16, ))
    loss_val = loss_fn(inputs=pred, targets=gt)
    print(loss_val)
    loss_val.backward()

