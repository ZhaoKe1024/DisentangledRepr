#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/8/17 14:33
# @Author: ZhaoKe
# @File : code_test.py
# @Software: PyCharm
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_sum_assignment

const_matrix = np.array([[15, 40, 45], [20, 60, 35], [20, 40, 25]])
# matches = linear_assignment(const_matrix)
matches1 = linear_sum_assignment(const_matrix)
# print("matches_type=", type(matches))
print("matches1_type=", type(matches1))
# print("matches=", matches)
print("matches1=", matches1)
print("matches1_T=", np.array(matches1).T)
