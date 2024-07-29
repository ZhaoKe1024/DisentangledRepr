#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/25 15:21
# @Author: ZhaoKe
# @File : gmm_kit.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
sns.set()
from sklearn.datasets import make_moons, make_blobs
from sklearn.mixture import GaussianMixture as GMM


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)

    probs = gmm.predict_proba(X)
    print(probs[:5].round(3))

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()


def cluster_bolb():
    Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
    print(Xmoon.shape, ymoon.shape)
    plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
    # 如果用GMM对数据拟合出两个成分，那么作为一个聚类模型的结果，效果将会很差
    gmm8 = GMM(n_components=6, covariance_type='full', random_state=0)
    labels = gmm8.fit(Xmoon).predict(Xmoon)
    probs = gmm8.predict_proba(Xmoon)
    print(probs[:5].round(3))
    print("weight:")
    print(gmm8.weights_)
    print("mean:")
    print(gmm8.means_)
    print("covars:")
    print(gmm8.covariances_)
    print("params:")
    print(gmm8.get_params(deep=True))


def cluster_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data
    labels = iris.target
    print("shape features:{}, labels:{}".format(features.shape, labels.shape))
    gmm8 = GMM(n_components=3, covariance_type='full').fit(features)
    labels = gmm8.predict(features)
    probs = gmm8.predict_proba(features)

    print(probs[:5].round(3))
    print("weight:")
    print(gmm8.weights_)
    print("mean:")
    print(gmm8.means_)
    print("covars:")
    print(gmm8.covariances_)
    print("params:")
    print(gmm8.get_params(deep=True))

    print(labels.shape)
    print(probs.shape)


def cluster_mnist():
    from sklearn.datasets import load_digits
    mnist = load_digits()
    features = mnist.data
    labels = mnist.target
    print("shape features:{}, labels:{}".format(features.shape, labels.shape))
    gmm8 = GMM(n_components=10, covariance_type='full').fit(features)
    labels = gmm8.predict(features)
    probs = gmm8.predict_proba(features)

    print(probs[:5].round(3))
    print("weight:")
    print(gmm8.weights_)
    print("mean:")
    print(gmm8.means_)
    print("covars:")
    print(gmm8.covariances_)
    print("params:")
    print(gmm8.get_params(deep=True))

    print(labels.shape)
    print(probs.shape)


if __name__ == '__main__':
    # import torch
    # from torchvision import models
    # modelvgg16 = models.vgg16(pretrained=False)
    # weights = torch.load("C:/Users/zhaoke/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
    # modelvgg16.load_state_dict(weights)
    # print(modelvgg16)

    cluster_iris()
    # plt.figure(0)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    # plt.show()

    # print("GMM 前五行的后验概率预测")
    # probs = gmm.predict_proba(X)
    # print(probs[:5].round(3))
    # # 将每个点簇分配的概率可视化
    # size = 50 * probs.max(1) ** 2  # 平方放大概率的差异
    # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
    # # ————————————————
    # # 用椭圆形来拟合数据
    # rng = np.random.RandomState(13)
    # X_stretched = np.dot(X, rng.randn(2, 2))
    # gmm = GMM(n_components=4, covariance_type='full', random_state=42)
    # plot_gmm(gmm, X_stretched)
