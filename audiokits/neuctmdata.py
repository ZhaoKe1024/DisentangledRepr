#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/9/30 21:15
# @Author: ZhaoKe
# @File : neuctmdata.py
# @Software: PyCharm
import os
import numpy as np
import pandas
import librosa
import matplotlib.pyplot as plt
# import seaborn as sns
# import pyecharts

netctm_path = "F:/NEUCTMDATASET"

def build_csv():
    label_wav_dict = dict()
    
