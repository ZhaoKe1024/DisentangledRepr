#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/9/3 17:05
# @Author: ZhaoKe
# @File : figurekits.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

color_4 = [
    "#3d6499", "#5ab886", "#f5e871", "#c5e7a4"]


def get_heat_map(pred_matrix, label_vec, savepath):
    # max_arg = list(pred_matrix.argmax(axis=-1))
    max_arg = pred_matrix
    conf_mat = metrics.confusion_matrix(max_arg, label_vec)
    print(conf_mat)
    df_cm = pd.DataFrame(conf_mat, index=range(conf_mat.shape[0]), columns=range(conf_mat.shape[0]))
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.xlabel("predict label")
    plt.ylabel("true label")
    plt.savefig(savepath)


def plot_embedding_2D(data, label, title, savepath, names, params=None):
    """
    """
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    print("plot_embedding_2D():", data.shape, label.shape)
    plt.figure()
    # print(label)
    # label_cnt = [1] * len(names)
    # p_list = [None] * len(names)
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=2, color=color_4[label[i]], label=names[label[i]])
        # if label_cnt[label[i]] == 1:
        #     p_list[label[i]] = p
        #     label_cnt[label[i]] = 0
    # print(p_list)
    # plt.xticks([])
    # plt.yticks([])
    plt.title(title)
    # plt.legend(names, bbox_to_anchor=(1.05, 1), loc="upper right")
    # plt.legend()
    # plt.legend(p_list, names, loc="lower right")
    # plt.savefig(savepath)
    plt.savefig(savepath, dpi=600, format=params["format"], bbox_inches="tight")
    # plt.show()
    plt.close()
    # return fig1  # , fig2


def save_legend():
    # params = {"format": "svg", "marker_size": 8, "alpha": 0.8}
    label = list(range(4))
    names = ["healthy", "covid19"]
    plt.figure(0)
    for i in range(2):
        plt.scatter(0, 0, marker='o', label=names[i], color=color_4[i])
    plt.xticks([])
    plt.yticks([])
    plt.legend(names, loc="upper right")
    # plt.savefig("./legend_bar.svg", dpi=600, format=params["format"], bbox_inches="tight")
    plt.show()
    names = ["dry", "wet", "unknown"]
    plt.figure(0)
    for i in range(3):
        plt.scatter(0, 0, marker='o', label=names[i], color=color_4[i])
    plt.xticks([])
    plt.yticks([])
    plt.legend(names, loc="upper right")
    # plt.savefig("./legend_bar.svg", dpi=600, format=params["format"], bbox_inches="tight")
    plt.show()
    names = ["mild", "pseudocough", "severe", "unknown"]
    plt.figure(0)
    for i in range(4):
        plt.scatter(0, 0, marker='o', label=names[i], color=color_4[i])
    plt.xticks([])
    plt.yticks([])
    plt.legend(names, loc="upper right")
    # plt.savefig("./legend_bar.svg", dpi=600, format=params["format"], bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    save_legend()
