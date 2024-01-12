

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow.pytorch
import numpy as np
from tqdm import tqdm
from dataset import MoleculeDataset
from model import *
# import mlflow.pytorch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from utils import *
from vae_model import *

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## PARAMS
num_batch = 64
num_epoch = 50
lr = 0.001
log_interval = 5
max_beta = .5
min_beta = 0
annealing_steps = num_epoch


train_dataset = MoleculeDataset(root="./data/", filename="train_bace.csv")
test_dataset = MoleculeDataset(root="./data/", filename="test_bace.csv", test=True)

# Prepare training
train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


model = mymolGen(real=False)
model = model.to(device)

print('NumParams = :{}'.format(count_parameters(model)))

# < 1 increases precision, > 1 recall
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01)

# Start training


## train loop

all_loss_train = []
loss_test = []
train_acc = []
test_acc = []
allbeta = []
for iter in tqdm(range(num_epoch), position=0, leave=True):
    model.train()
    running_loss = 0
    acc = 0
    current_beta = min(max_beta, (max_beta - min_beta) * iter / annealing_steps)
    allbeta.append(current_beta)
    for _, data in enumerate(train_loader):
        data.to(device)
        optimizer.zero_grad()
        data.x = data.x.float()
        # data.molfeature = data.molfeature.float()
        # len = nn.functional.one_hot(data.len, 200)
        pred, mu, logstd = model(data.x, data.edge_index, data.batch, data.len)
        # l2_regularization = sum(p.pow(2).sum() for p in model.parameters())
        loss = current_beta * kl_loss(mu, logstd) + \
               loss_fn(pred, data.smile_encoded.float().view(data.y.shape[0], -1).clamp(0))
               # 0.01 * l2_regularization

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    all_loss_train.append(running_loss/num_batch)
    train_acc.append(acc/num_batch)
    mlflow.log_metric("train_loss", running_loss/num_batch, step=iter)
    # calculate_metrics((torch.where(pred >= 0.5, 1, 0).T, data.y.float(), epoch, "train")

    with torch.no_grad():
        model.eval()
        test_data = next(enumerate(test_loader))[1]
        test_data.x = test_data.x.float()
        test_data.molfeature = test_data.molfeature.float()
        pred, mu, logstd = model(test_data.x, test_data.edge_index, test_data.batch, test_data.len)
        test_loss = current_beta * kl_loss(mu, logstd) + \
               loss_fn(pred, test_data.smile_encoded.float().view(test_data.y.shape[0], -1).clamp(0))

        loss_test.append(test_loss.item())

    if iter % log_interval == 0:
        print('Train Epoch: {} \tTrainLoss: {:.6f} \tTestLoss:{:.6f}'.format(
            iter, all_loss_train[iter], test_loss.item()))


torch.save(model.state_dict(), './saved_model/model_smile2.pth')
torch.save(optimizer.state_dict(), './saved_model/optimizer1.pth')

##
plt.figure()
plt.plot(allbeta)


plt.figure()
plt.plot(all_loss_train, 'b', label='train')
plt.plot(loss_test, 'r', label='test')
plt.legend()

model.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
test_data = next(enumerate(test_loader))[1]
test_data.x = test_data.x.float()
pred = model(test_data.x, test_data.edge_index, test_data.batch, test_data.len)[0]
plt.figure()
plt.imshow(test_data.smile_encoded.float().view(200, -1))
plt.figure()
res_mat = np.squeeze(pred.detach().numpy()).reshape(200, -1)
bin_mat = np.zeros_like(res_mat)
bin_mat[np.arange(len(res_mat)), res_mat.argmax(axis=1)] = 1
plt.imshow(bin_mat)


print(test_data.smiles)
print(smiles_decoder(bin_mat.astype(int), mode='B'))


plt.show()