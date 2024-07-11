#!/usr/bin/env python
# coding: utf-8

# # 1. import some modules
import os
import sys
import random
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import idbsr_libs.config_mnist_rot as config
from idbsr_libs.VIB_MNIST_ROT_whole_model import VariationalInformationBottleneck
from idbsr_libs.Get_Datasets import get_datasets
from idbsr_libs.Visualize import tsne_embedding_without_images
from idbsr_libs.VIB_model import Weight_EMA
sys.path.append('../')
sys.path.append('C:/Program Files (zk)/PythonFiles/DisentangledRepr/')


# # 2. the environment settings
gpu_id = 0
model_name = 'fair_single_best.pt'
ema_model_name = 'fair_ema_best.pt'
data_root = 'F:/DATAS/'


# # 3. some configs
train_args = config.train_args()
dataset_args = config.dataset_args()
model_args = config.model_args()
use_cuda = True


# # 4. fixed the seeds

# In[5]:


train_args.gpu = 0
init_seed = train_args.seed
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)
random.seed(init_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(train_args.gpu)


# # 5. loading datasets...
train_loader, test_loader, test_55_loader, test_65_loader = get_datasets(
    dataset_args.dataset, dataset_args.train_batch_size, dataset_args.test_batch_size, root=data_root)
print(len(train_loader), len(test_loader))
print(len(test_55_loader), len(test_65_loader))


# # 6. print some key values
parameter = {
    "reconstruction_weight": model_args.reconstruction_weight,
    "pairwise_kl_clean_weight": model_args.pairwise_kl_clean_weight,
    "pairwise_kl_noise_weight": model_args.pairwise_kl_noise_weight,
    "sparse_kl_weight_clean": model_args.sparse_kl_weight_clean,
    "sparse_kl_weight_noise": model_args.sparse_kl_weight_noise,
    "sparsity_clean": model_args.sparsity_clean,
    "sparsity_noise": model_args.sparsity_noise,
    "num_sensitive_class": dataset_args.num_sensitive_class
}
for k, v in parameter.items():
    print(k, ':\t',v)


# # 7. create the saved dir...
path_1 = '%.03f_%.03f_%.03f' %(model_args.reconstruction_weight, model_args.pairwise_kl_clean_weight, model_args.pairwise_kl_noise_weight)
path_2 = '%.02f_%.02f_%.02f_%.02f' %(model_args.sparse_kl_weight_clean, model_args.sparse_kl_weight_noise, model_args.sparsity_clean, model_args.sparsity_noise)
save_model_dir = os.path.join('./runs/idbsr_mnistrot/saved_model/', path_1, path_2, str(train_args.seed))
os.makedirs(save_model_dir, exist_ok=True)
print(save_model_dir)


# # 8. define the model
vib_model = VariationalInformationBottleneck(
    dataset_args.shape_data, dataset_args.num_target_class,
    model_args.dim_embedding_clean, model_args.dim_embedding_noise,
    model_args.channel_hidden_encoder, model_args.channel_hidden_decoder,
    model_args.dim_hidden_classifier, parameter
)  # (28, 28), 10, clean 10, noise 20, channel encoder 64, decoder (256, 128), cls 128

vib_model_copy = VariationalInformationBottleneck(
    dataset_args.shape_data, dataset_args.num_target_class,
    model_args.dim_embedding_clean, model_args.dim_embedding_noise,
    model_args.channel_hidden_encoder, model_args.channel_hidden_decoder,
    model_args.dim_hidden_classifier, parameter
)

if use_cuda:
    vib_model = vib_model.cuda()
    vib_model_copy = vib_model_copy.cuda()


# # 9. optimizer

# In[10]:


vib_model_EMA = Weight_EMA(vib_model_copy, vib_model.state_dict(), decay=0.999)
optimizer = torch.optim.Adam(vib_model.parameters(), lr=train_args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.95)


# # 10. train one epoch

# In[11]:


def train_one_epoch():
    total_loss_total = 0.0
    classification_loss_total = 0.0
    classification_sensitive_loss_total = 0.0
    reconstruction_loss_total = 0.0
    pairwise_kl_loss_clean_total = 0.0
    pairwise_kl_loss_noise_total = 0.0
    sparse_kl_loss_clean_total = 0.0
    sparse_kl_loss_noise_total = 0.0

    for iter_index, (images, labels, sensitive_labels) in enumerate(train_loader):
        images = np.reshape(images, (-1, 28, 28))
        images = Variable(images.unsqueeze(dim=1).float())
        labels = Variable(labels.long())
        sensitive_labels = Variable(sensitive_labels.long())
        if iter_index == 0:
            print(images.shape, labels.shape, sensitive_labels.shape)
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
            sensitive_labels = sensitive_labels.cuda()

        (total_loss, classification_loss, classification_sensitive_loss, reconstruction_loss, pairwise_kl_loss_clean,
         pairwise_kl_loss_noise, sparse_kl_loss_clean, sparse_kl_loss_noise) = vib_model(
            input_data=images, input_label=labels, input_sensitive_labels=sensitive_labels, num_samples=10, training=True
        )

        total_loss_total = total_loss_total + total_loss.sum(-1)
        classification_loss_total = classification_loss_total + classification_loss.sum(-1)
        classification_sensitive_loss_total = classification_sensitive_loss_total + classification_sensitive_loss.sum(-1)
        reconstruction_loss_total = reconstruction_loss_total + reconstruction_loss.sum(-1)
        pairwise_kl_loss_clean_total = pairwise_kl_loss_clean_total + pairwise_kl_loss_clean.sum(-1)
        pairwise_kl_loss_noise_total = pairwise_kl_loss_noise_total + pairwise_kl_loss_noise.sum(-1)
        sparse_kl_loss_clean_total = sparse_kl_loss_clean_total + sparse_kl_loss_clean.sum(-1)
        sparse_kl_loss_noise_total = sparse_kl_loss_noise_total + sparse_kl_loss_noise.sum(-1)

        optimizer.zero_grad()
        total_loss.mean(-1).backward()
        optimizer.step()
        vib_model_EMA.update(vib_model.state_dict())
    lr_scheduler.step()
    total_loss_mean = total_loss_total / len(train_loader.dataset)
    classification_loss_mean = classification_loss_total / len(train_loader.dataset)
    classification_sensitive_loss_mean = classification_sensitive_loss_total / len(train_loader.dataset)
    reconstruction_loss_mean = reconstruction_loss_total / len(train_loader.dataset)
    pairwise_kl_loss_clean_mean = pairwise_kl_loss_clean_total / len(train_loader.dataset)
    pairwise_kl_loss_noise_mean = pairwise_kl_loss_noise_total / len(train_loader.dataset)
    sparse_kl_loss_clean_mean = sparse_kl_loss_clean_total / len(train_loader.dataset)
    sparse_kl_loss_noise_mean = sparse_kl_loss_noise_total / len(train_loader.dataset)
    return (total_loss_mean, classification_loss_mean, classification_sensitive_loss_mean,
            reconstruction_loss_mean, pairwise_kl_loss_clean_mean, pairwise_kl_loss_noise_mean,
            sparse_kl_loss_clean_mean, sparse_kl_loss_noise_mean)


# # 11. evaluation

# In[12]:


def evaluation(epoch_index, test_dataloader, is_drawing=False):
    vib_model.eval()
    vib_model_EMA.model.eval()
    avg_correct = 0.0
    single_correct = 0.0

    valid_embedding_labels = []
    valid_embedding_sensitive_labels = []
    valid_embedding_clean_images = []

    for iter_index, data in enumerate(test_dataloader):
        images = data[0]
        labels = data[1]
        images = np.reshape(images, (-1, 28, 28))
        images = Variable(images.unsqueeze(dim=1).float())
        labels = Variable(labels.long())
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        avg_embedding, _, avg_classification_prob = vib_model_EMA.model(
            images, labels, num_samples=100, training=False
        )
        avg_prediction = avg_classification_prob.max(1)[1]
        avg_correct = avg_correct + torch.eq(avg_prediction, labels).float().sum()

        single_embedding, _, single_classification_prob = vib_model(
            images, labels, num_samples=100, training=False
        )
        single_prediction = single_classification_prob.max(1)[1]
        single_correct = single_correct + torch.eq(single_prediction, labels).float().sum()

        if is_drawing:
            valid_embedding_labels.extend(np.asarray(labels.detach().numpy()))
            valid_embedding_clean_images.extend(np.asarray(single_embedding.detach().numpy()))

    if is_drawing:
        os.makedirs("./runs/idbsr_mnistrot/Log/", exist_ok=True)
        tsne_embedding_without_images(images=valid_embedding_clean_images,
                                      labels=[valid_embedding_sensitive_labels],
                                      save_name="./runs/idbsr_mnistrot/Log/result_" + str(epoch_index) + "_clean.png")

    avg_correct_mean = avg_correct / len(test_dataloader.dataset)
    single_correct_mean = single_correct / len(test_dataloader.dataset)
    return avg_correct_mean * 100, single_correct_mean * 100


# # 12. train

# In[13]:


def train():
    best_avg_correct = 0.0
    best_single_correct = 0.0

    epoches = 500  # train_args.max_epoch
    for epoch_index in tqdm(range(epoches)):
        vib_model.train()
        vib_model_EMA.model.train()

        # avg_correct, single_correct = evaluation(epoch_index + 1, test_loader)

        (total_loss, classification_loss, classification_sensitive_loss, reconstruction_loss,
         pairwise_kl_loss_clean, pairwise_kl_loss_noise, sparse_kl_loss_clean, sparse_kl_loss_noise) = train_one_epoch()
        if (epoch_index + 1) % 2 == 0:
            print('[train]Epoch: {}, total_loss: {:.4}, classification_loss: {:.4}, classification_sensitive_loss: {:.4}, '
                  'reconstruction_loss: {:.4}, pairwise_kl_loss_clean: {:.4}, pairwise_kl_loss_noise: {:.4}, '
                  'sparse_kl_loss_clean: {:.4}, sparse_kl_loss_noise: {:.4}'
                  .format(epoch_index + 1, total_loss, classification_loss, classification_sensitive_loss, reconstruction_loss,
                          pairwise_kl_loss_clean, pairwise_kl_loss_noise, sparse_kl_loss_clean, sparse_kl_loss_noise))

        if (epoch_index + 1) % 2 == 0:
            is_drawing = False
            print('##################### test #####################')
            avg_correct, single_correct = evaluation(epoch_index + 1, test_loader, is_drawing=is_drawing)

            if best_avg_correct <= avg_correct:
                best_avg_correct = avg_correct
                print("##################### save #####################")
                torch.save(vib_model_EMA.model.state_dict(),
                           os.path.join(save_model_dir, ema_model_name))

            if best_single_correct <= single_correct:
                best_single_correct = single_correct
                print("##################### save #####################")
                torch.save(vib_model.state_dict(),
                           os.path.join(save_model_dir, model_name))

            print('[test]Epoch: {}, avg_correct: {:.4}, best_avg_correct: {:.4}, '
                  'single_correct: {:.4}, best_single_correct: {:.4}'
                  .format(epoch_index + 1, avg_correct, best_avg_correct,
                          single_correct, best_single_correct))


# ## 13. start training

# In[ ]:


train()


# In[ ]:




