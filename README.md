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
