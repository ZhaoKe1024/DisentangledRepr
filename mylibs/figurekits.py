#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/9/3 17:05
# @Author: ZhaoKe
# @File : figurekits.py
# @Software: PyCharm
import matplotlib.pyplot as plt

color_8 = [
    "#000075", "#9A6324", "#000000", "#5d1451",
    "#800000", "#f58231", "#ffe119", "#f032e6", ]


def plot_embedding_2D(data, label, title, savepath, names, params=None):
    """
    """
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    plt.figure()
    # print(label)
    label_cnt = [1] * len(names)
    p_list = [None] * len(names)
    for i in range(data.shape[0]):
        # p = plt.scatter(data[i, 0], data[i, 1], s=params["marker_size"], c=rgb_planning_23[label[i]], alpha=params["alpha"])

        if label[i] < 4:
            p = plt.plot(data[i, 0], data[i, 1], marker='o', markersize=6, color=color_8[label[i] + 4],
                         label=names[label[i]])
        else:
            p = plt.plot(data[i, 0], data[i, 1], marker='x', markersize=14, color=color_8[label[i]],
                         label=names[label[i]])
        if label_cnt[label[i]] == 1:
            p_list[label[i]] = p
            label_cnt[label[i]] = 0
    # print(p_list)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.legend(names, bbox_to_anchor=(1.05, 1), loc="upper right")
    # plt.legend()
    plt.legend(p_list, names, loc="lower right")
    # plt.savefig(savepath)
    plt.show()
    plt.savefig(savepath, dpi=600, format=params["format"], bbox_inches="tight")
    plt.close()
    # return fig1  # , fig2
