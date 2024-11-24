# DisentangledRepr
 a repository about conditional model, disentangled representation model, style-based model, and compressed model.

# AGEDR
*Disentangling Representations using Attributes-based Gaussian Estimation for Medical Sound Diagnosis*

## file structure

```text
root
└─p3agedr.py  # training and validation for the AGEDR model
└─mylibs
│    └─conv_vae  # the VAE modules
└─audiokits
│    └─transforms.py  # tools for data augmentation
```

run the project
```commandline
python ./p3agedr.py
```

run the project with different experiments
```python
agedr = AGEDRTrainer()
agedr.demo()  # test the code
agedr.train()  # train

# test the NN and SVM
agedr.evaluate_cls(seed=12)  # test NN cls
agedr.evaluate_cls_ml(seed=12)  # test SVM cls
agedr.evaluate_tsne()

agedr.train_cls(latent_dim=30, onlybeta=False, seed=89, vaepath="./runs/agedr/202409061417_一层Linear/")
agedr.train_cls(latent_dim=16, onlybeta=True, seed=89, vaepath="./runs/agedr/202409061417_一层Linear/")

# train from checkpoint
# agedr.train(load_ckpt_path="./runs/agedr/202409041841/")

# test from pretrained AGEDR model
# agedr.evaluate_retrain_cls(latent_dim=30, onlybeta=False,
#                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
#                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld30_retrain30.pth")
# agedr.evaluate_retrain_cls(latent_dim=16, onlybeta=True,
#                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
#                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld16_retrain80.pth")
# agedr.evaluate_retrain_cls(latent_dim=30, onlybeta=False,
#                            vaepath="./runs/agedr/202409042044_一层Linear_分类失败/epoch370/epoch_370_vae.pth",
#                            clspath="./runs/agedr/202409042044_一层Linear_分类失败_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld30_retrain30.pth")
# agedr.evaluate_retrain_cls(latent_dim=16, onlybeta=True,
#                            vaepath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/epoch_370_vae.pth",
#                            clspath="./runs/agedr/202409051036_二层Linear_提取特征/epoch370/retrain_cls/cls_vae370_ld16_retrain80.pth")
```

# Additional Experiments
## About clustering each attribute with or without using z^a
### Source Code:
```commandline
python p3agedr_cls.py
```
```python
if __name__ == '__main__':
    # evaluate_predict_latent()
    # evaluate_attri_tsne()
    # evaluate_attri_perceptron()
    evaluate_attri_KMeans()
```
### Result:
```text
KMeans confusion matrix about attribute cough_type on train set
[[2807 0   0  ]
 [0    899 0  ]
 [0    0   246]]
KMeans confusion matrix about attribute cough_type on valid set
[[135 0  0 ]
 [0   47 0 ]
 [0   0  18]]

KMeans confusion matrix about attribute severity on train set
[[2587 0   0   0  ]
 [0    531 0   0  ]
 [0    0   652 0  ]
 [0    0   0   182]]
KMeans confusion matrix about attribute severity on valid set
[[115 0  0  0 ]
 [0   48 0  0 ]
 [0   0  26 0 ]
 [0   0  0  11]]
```

## About predicting each attribute with or without using z and z^beta
### Source code:
```commandline
python p3agedr_cls.py
```
```python
if __name__ == '__main__':
    predict_using_Perceptron_and_latent()
```
### Result:
```text
    inp=latent_dim, hidden_dim=16
    [[106  20   9]
     [ 36   7   4]
     [ 14   4   0]]
    [[73  0  3 39]
     [30  0  6 12]
     [16  0  1  9]
     [ 7  0  0  4]]
     
     inp=blen, hidden_dim=16
     [[133   2   0]
     [ 47   0   0]
     [ 18   0   0]]
    [[ 0 65 29 21]
     [ 0 26 18  4]
     [ 0 14  8  4]
     [ 0  6  3  2]]
```


## About t-SNE of z and z^{\beta}
### Source code:
```commandline
python p3agedr_cls.py
```
```python
if __name__ == '__main__':
    # evaluate_predict_using_SVM_and_latent()
    evaluate_attri_tsne()
    # evaluate_attri_perceptron()
    # evaluate_attri_KMeans()
```
### Result:
![t-SNE of z on healthy or covid19](/images/tsne_health.png), ![](/images/tsne_coughtype.png), ![](/images/tsne_severity.png)


