#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/22 13:54
# @Author: ZhaoKe
# @File : p3agedr_cls.py
# @Software: PyCharm
import os.path
import random
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from mylibs.figurekits import plot_embedding_2D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from mylibs.modules import Classifier, FocalLoss
from mylibs.figurekits import get_heat_map, plot_embedding_2D
from mylibs.conv_vae import ConvVAE
from audiokits.transforms import *
from p3agedr import AME, setup_seed, bin_upsampling_balance, normalize_data, CoughvidDataset


def get_dataloader(batch_size=16):
    coughvid_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_split_specattri.pkl")
    coughvid_df = coughvid_df.iloc[:, [0, 1, 2, 8, 9]]
    neg_list = list(range(2076))
    pos_list = list(range(2076, 2850))
    random.shuffle(neg_list)
    random.shuffle(pos_list)

    valid_list = neg_list[:100] + pos_list[:100]
    train_list = neg_list[100:] + pos_list[100:]
    train_df = bin_upsampling_balance(coughvid_df.iloc[train_list, :])
    valid_df = coughvid_df.iloc[valid_list, :]
    # print(train_df.head())
    print("train valid length:", train_df.shape, valid_df.shape)
    # normalize the data
    train_df, valid_df = normalize_data(train_df, valid_df)

    train_transforms = transforms.Compose([MyRightShift(input_size=(128, 64),
                                                        width_shift_range=7,
                                                        shift_probability=0.9),
                                           MyAddGaussNoise(input_size=(128, 64),
                                                           add_noise_probability=0.55),
                                           MyReshape(output_size=(1, 128, 64))])
    test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 64))])
    train_ds = CoughvidDataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0)
    # init test data loader
    valid_ds = CoughvidDataset(valid_df, transform=test_transforms)
    valid_loader = DataLoader(valid_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=0)
    return train_loader, valid_loader


def pretrain_AGRDR_cls(device):
    latent_dim = 30
    a1len, a2len = 6, 8
    pretrain_path = "./runs/agedr/202409051036_二层Linear_提取特征/epoch370/"
    ame1 = AME(class_num=3, em_dim=a1len).to(device)
    ame2 = AME(class_num=4, em_dim=a2len).to(device)
    vae = ConvVAE(inp_shape=(1, 128, 64), latent_dim=latent_dim, flat=True).to(device)
    # classifier = Classifier(dim_embedding=latent_dim, dim_hidden_classifier=32, num_target_class=class_num).to(device)
    ame1.load_state_dict(torch.load(pretrain_path + "epoch_370_ame1.pth"))
    ame2.load_state_dict(torch.load(pretrain_path + "epoch_370_ame2.pth"))
    vae.load_state_dict(torch.load(pretrain_path + "epoch_370_vae.pth"))
    # ame1.load_state_dict(torch.load(pretrain_path+""))
    ame1.eval()
    ame2.eval()
    vae.eval()
    return ame1, ame2, vae


def get_representation():
    a1len, a2len = 6, 8
    # latent_dim = 30
    # blen = latent_dim - a1len - a2len
    # class_num = 2
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # set random seed
    setup_seed(12)
    # building data loaders
    print("building data loaders.")
    train_loader, valid_loader = get_dataloader(batch_size=batch_size)
    # loading pretrained models
    print("loading pretrained models.")
    ame1, ame2, _ = pretrain_AGRDR_cls(device=device)
    # build train data
    print("build train data")
    a1feats_tr_input = None
    a2feats_tr_input = None
    # y_labs_input = None
    ctype_tr_label = None
    sevty_tr_label = None
    for jdx, batch in enumerate(train_loader):
        # x_mel = batch["spectrogram"].to(device)
        # y_lab = batch["label"].to(device)
        ctype = batch["cough_type"].to(device)
        sevty = batch["severity"].to(device)
        if ctype_tr_label is None:
            # y_labs_input = y_lab
            ctype_tr_label = ctype
            sevty_tr_label = sevty
        else:
            # y_labs_input = torch.concat((y_labs_input, y_lab), dim=0)
            ctype_tr_label = torch.concat((ctype_tr_label, ctype), dim=0)
            sevty_tr_label = torch.concat((sevty_tr_label, sevty), dim=0)

        with torch.no_grad():
            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            if a1feats_tr_input is None:
                a1feats_tr_input = mu_a_1
                a2feats_tr_input = mu_a_2
            else:
                a1feats_tr_input = torch.concat((a1feats_tr_input, mu_a_1), dim=0)
                a2feats_tr_input = torch.concat((a2feats_tr_input, mu_a_2), dim=0)
    # build valid data
    print("build valid data")
    a1feats_va_input = None
    a2feats_va_input = None
    # y_labs_input = None
    ctype_va_label = None
    sevty_va_label = None
    for jdx, batch in enumerate(valid_loader):
        # x_mel = batch["spectrogram"].to(device)
        # y_lab = batch["label"].to(device)
        ctype = batch["cough_type"].to(device)
        sevty = batch["severity"].to(device)
        if ctype_va_label is None:
            # y_labs_input = y_lab
            ctype_va_label = ctype
            sevty_va_label = sevty
        else:
            # y_labs_input = torch.concat((y_labs_input, y_lab), dim=0)
            ctype_va_label = torch.concat((ctype_va_label, ctype), dim=0)
            sevty_va_label = torch.concat((sevty_va_label, sevty), dim=0)

        with torch.no_grad():
            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            if a1feats_va_input is None:
                a1feats_va_input = mu_a_1
                a2feats_va_input = mu_a_2
            else:
                a1feats_va_input = torch.concat((a1feats_va_input, mu_a_1), dim=0)
                a2feats_va_input = torch.concat((a2feats_va_input, mu_a_2), dim=0)
    return (a1feats_tr_input, ctype_tr_label, a1feats_va_input, ctype_va_label,
            a2feats_tr_input, sevty_tr_label, a2feats_va_input, sevty_va_label)


class Perceptron(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.layer1 = nn.Linear(inp, 8, bias=True)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(8, oup, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_x):
        out = self.relu1(self.layer1(input_x))
        return self.softmax(self.layer2(out))


def trainPerceptron(train_loader, valid_loader, ame_model, input_name, dim_x, class_num, device):
    model = Perceptron(inp=dim_x, oup=class_num).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_f = nn.CrossEntropyLoss().to(device)
    ame_model.eval()
    print("training perceptron...")
    for epoch_id in range(10):
        model.train()
        for batch_id, batch in enumerate(train_loader):
            input_lab = batch[input_name].to(device)
            with torch.no_grad():
                input_mu, _, _ = ame_model(input_lab)
            optimizer.zero_grad()
            y_pred = model(input_mu)
            loss = loss_f(input=y_pred, target=input_lab)
            loss.backward()
            optimizer.step()
    print("validing perceptron...")
    model.eval()
    y_pred_all = None
    y_truth_all = None
    for batch_id, batch in enumerate(valid_loader):
        input_lab = batch[input_name].to(device)
        with torch.no_grad():
            input_mu, _, _ = ame_model(input_lab)
            y_pred = model(input_mu)
        if y_pred_all is None:
            y_pred_all = y_pred
            y_truth_all = input_lab
        else:
            y_pred_all = torch.concat((y_pred_all, y_pred))
            y_truth_all = torch.concat((y_truth_all, input_lab))
    y_pred_all = y_pred_all.argmax(axis=-1)
    y_pred_all = y_pred_all.data.cpu().numpy()
    print(y_pred_all.shape)
    y_truth_all = y_truth_all.data.cpu().numpy()
    print(y_pred_all)
    print(y_truth_all)
    print("classification result:")
    print(confusion_matrix(y_true=y_truth_all, y_pred=y_pred_all))


def evaluate_attri_cls():
    """
    classifying using the attri mapped vector
    :return:
    """
    a1len, a2len = 6, 8
    latent_dim = 30
    # blen = latent_dim - a1len - a2len
    # class_num = 2
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # set random seed
    setup_seed(12)
    # building data loaders
    print("building data loaders.")
    train_loader, valid_loader = get_dataloader(batch_size=batch_size)
    # loading pretrained models
    print("loading pretrained models.")
    ame1, ame2, vae = pretrain_AGRDR_cls(device=device)

    # build train data
    print("build train data")
    a1feats_tr_input = None
    a2feats_tr_input = None
    # y_labs_input = None
    ctype_tr_label = None
    sevty_tr_label = None
    for jdx, batch in enumerate(train_loader):
        # x_mel = batch["spectrogram"].to(device)
        # y_lab = batch["label"].to(device)
        ctype = batch["cough_type"].to(device)
        sevty = batch["severity"].to(device)
        if ctype_tr_label is None:
            # y_labs_input = y_lab
            ctype_tr_label = ctype
            sevty_tr_label = sevty
        else:
            # y_labs_input = torch.concat((y_labs_input, y_lab), dim=0)
            ctype_tr_label = torch.concat((ctype_tr_label, ctype), dim=0)
            sevty_tr_label = torch.concat((sevty_tr_label, sevty), dim=0)

        with torch.no_grad():
            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            if a1feats_tr_input is None:
                a1feats_tr_input = mu_a_1
                a2feats_tr_input = mu_a_2
            else:
                a1feats_tr_input = torch.concat((a1feats_tr_input, mu_a_1), dim=0)
                a2feats_tr_input = torch.concat((a2feats_tr_input, mu_a_2), dim=0)
    # build valid data
    print("build valid data")
    a1feats_va_input = None
    a2feats_va_input = None
    # y_labs_input = None
    ctype_va_label = None
    sevty_va_label = None
    for jdx, batch in enumerate(valid_loader):
        # x_mel = batch["spectrogram"].to(device)
        # y_lab = batch["label"].to(device)
        ctype = batch["cough_type"].to(device)
        sevty = batch["severity"].to(device)
        if ctype_va_label is None:
            # y_labs_input = y_lab
            ctype_va_label = ctype
            sevty_va_label = sevty
        else:
            # y_labs_input = torch.concat((y_labs_input, y_lab), dim=0)
            ctype_va_label = torch.concat((ctype_va_label, ctype), dim=0)
            sevty_va_label = torch.concat((sevty_va_label, sevty), dim=0)

        with torch.no_grad():
            mu_a_1, logvar_a_1, _ = ame1(ctype)  # [32, 6] [32, 6]
            mu_a_2, logvar_a_2, _ = ame2(sevty)  # [32, 8] [32, 8]
            if a1feats_va_input is None:
                a1feats_va_input = mu_a_1
                a2feats_va_input = mu_a_2
            else:
                a1feats_va_input = torch.concat((a1feats_va_input, mu_a_1), dim=0)
                a2feats_va_input = torch.concat((a2feats_va_input, mu_a_2), dim=0)

    # # =================================================================================================
    # # ====================================----------SVM--------========================================
    # save_dir = "./runs/agedr/202409051036_二层Linear_提取特征/epoch370/attri_cls/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # # classifying1
    # print("classifying cough type.")
    # kernel = "poly"
    # # svm_model = svm.SVC(kernel=kernel, gamma='auto')
    # svm_data_tr = a1feats_tr_input.data.cpu().numpy()
    # svm_lab_tr = ctype_tr_label.data.cpu().numpy()
    # svm_data_va = a1feats_va_input.data.cpu().numpy()
    # svm_lab_va = ctype_va_label.data.cpu().numpy()
    # print("train and test input:", svm_data_tr.shape, svm_lab_tr.shape, svm_data_va.shape, svm_lab_va.shape)
    # # classifying2
    # print("classifying for severity.")
    # # svm_model = svm.SVC(kernel=kernel, gamma='auto')
    # svm_data_tr = a2feats_tr_input.data.cpu().numpy()
    # svm_lab_tr = sevty_tr_label.data.cpu().numpy()
    # svm_data_va = a2feats_va_input.data.cpu().numpy()
    # svm_lab_va = sevty_va_label.data.cpu().numpy()
    # print("train and test input:", svm_data_tr.shape, svm_lab_tr.shape, svm_data_va.shape, svm_lab_va.shape)
    # # svm_model.fit(svm_data_tr, svm_lab_tr)
    # # y_pref_tr = svm_model.predict(svm_data_tr)
    # # y_pref_te = svm_model.predict(svm_data_va)
    # # print("train and test pref:", y_pref_tr.shape, y_pref_te.shape)
    # # get_heat_map(pred_matrix=y_pref_te, label_vec=svm_lab_va, savepath=save_dir+"valid_result.png")
    # # print("train precision:", precision_score(svm_lab_tr, y_pref_tr))
    # # print("test precision:", precision_score(svm_lab_va, y_pref_te))
    # # print("train recall:", recall_score(svm_lab_tr, y_pref_tr))
    # # print("test recall:", recall_score(svm_lab_va, y_pref_te))
    # # print("train acc:", roc_auc_score(svm_lab_tr, y_pref_tr))
    # # print("test acc:", roc_auc_score(svm_lab_va, y_pref_te))
    # # ====================================----------SVM--------========================================
    # # =================================================================================================

    # # =================================================================================================
    # # ====================================---------t-SNE-------========================================
    # form = "png"
    # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    # result2D = tsne_model.fit_transform(svm_data_tr)
    # plot_embedding_2D(data=result2D, label=svm_lab_tr, title="t-SNT for cough type train",
    #                   savepath=save_dir + "tsne_{}_{}.{}".format("coughtype", "train", form),
    #                   names=["dry", "wet", "unknown"], params={"format": form})
    #
    # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    # result2D = tsne_model.fit_transform(svm_data_va)
    # print(result2D)
    # print(svm_lab_va)
    # plot_embedding_2D(data=result2D, label=svm_lab_va, title="t-SNT for cough type valid",
    #                   savepath=save_dir + "tsne_{}_{}.{}".format("coughtype", "valid", form),
    #                   names=["dry", "wet", "unknown"], params={"format": form})
    #
    # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    # result2D = tsne_model.fit_transform(svm_data_tr)
    # plot_embedding_2D(data=result2D, label=svm_lab_tr, title="t-SNT for severity train",
    #                   savepath=save_dir + "tsnev_{}_{}.{}".format("severity", "train", form),
    #                   names=["mild", "pseudocough", "severe", "unknown"], params={"format": form})
    #
    # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
    # result2D = tsne_model.fit_transform(svm_data_va)
    # print(result2D)
    # print(svm_lab_va)
    # plot_embedding_2D(data=result2D, label=svm_lab_va, title="t-SNT for severity valid",
    #                   savepath=save_dir + "tsne_{}_{}.{}".format("severity", "valid", form),
    #                   names=["mild", "pseudocough", "severe", "unknown"], params={"format": form})
    # # ====================================---------t-SNE-------========================================
    # # =================================================================================================


def evaluate_attri_perceptron():
    a1len, a2len = 6, 8
    # latent_dim = 30
    # blen = latent_dim - a1len - a2len
    # class_num = 2
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # set random seed
    setup_seed(12)
    # building data loaders
    print("building data loaders.")
    train_loader, valid_loader = get_dataloader(batch_size=batch_size)
    # loading pretrained models
    print("loading pretrained models.")
    ame1, ame2, _ = pretrain_AGRDR_cls(device=device)
    trainPerceptron(train_loader=train_loader, valid_loader=valid_loader,
                    ame_model=ame1, input_name="cough_type", dim_x=a1len, class_num=3, device=device)
    trainPerceptron(train_loader=train_loader, valid_loader=valid_loader,
                    ame_model=ame2, input_name="severity", dim_x=a2len, class_num=4, device=device)


def evaluate_attri_KMeans():
    a1feats_tr_input, ctype_tr_label, a1feats_va_input, ctype_va_label, a2feats_tr_input, sevty_tr_label, a2feats_va_input, sevty_va_label = get_representation()
    from sklearn.cluster import KMeans

    a1feats_tr_input = a1feats_tr_input.data.cpu().numpy()
    ctype_tr_label = ctype_tr_label.data.cpu().numpy()
    n_clusters = 3
    cluster = KMeans(n_clusters=n_clusters, random_state=0)
    cluster.fit(a1feats_tr_input)
    print(cluster.cluster_centers_)
    y_predict = cluster.predict(a1feats_tr_input)
    print(confusion_matrix(y_true=ctype_tr_label, y_pred=y_predict))
    # [[2807    0    0]
    #  [0  899    0]
    # [0 0 246]]

    a1feats_va_input = a1feats_va_input.data.cpu().numpy()
    ctype_va_label = ctype_va_label.data.cpu().numpy()
    y_predict = cluster.predict(a1feats_va_input)
    print(confusion_matrix(y_true=ctype_va_label, y_pred=y_predict))
    # [[135   0   0]
    #  [0  47   0]
    # [0 0 18]]

    a2feats_tr_input = a2feats_tr_input.data.cpu().numpy()
    sevty_tr_label = sevty_tr_label.data.cpu().numpy()
    n_clusters = 4
    cluster = KMeans(n_clusters=n_clusters, random_state=0)
    cluster.fit(a2feats_tr_input)
    print(cluster.cluster_centers_)
    y_predict = cluster.predict(a2feats_tr_input)
    print(confusion_matrix(y_true=sevty_tr_label, y_pred=y_predict))
    # [[2587    0    0    0]
    #  [0    0  531    0]
    # [0 652 0 0]
    # [0    0    0  182]]

    a2feats_va_input = a2feats_va_input.data.cpu().numpy()
    sevty_va_label = sevty_va_label.data.cpu().numpy()
    y_predict = cluster.predict(a2feats_va_input)
    print(confusion_matrix(y_true=sevty_va_label, y_pred=y_predict))
    # [[115   0   0   0]
    #  [0   0  48   0]
    # [0 26 0 0]
    # [0   0   0  11]]


if __name__ == '__main__':
    # evaluate_attri_cls()

    # evaluate_attri_perceptron()

    evaluate_attri_KMeans()

    # agedr.evaluate_retrain_cls(latent_dim=30, onlybeta=False,
    #                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
    #                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld30_retrain30.pth")
