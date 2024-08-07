

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

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, type):

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)


## PARAMS
num_batch = 32
num_epoch = 200
lr = 0.001
log_interval = 5


train_dataset = MoleculeDataset(root="data/", filename="train_bace.csv")
test_dataset = MoleculeDataset(root="data/", filename="test_bace.csv", test=True)

# Prepare training
train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


model = myGNN()
model = model.to(device)

print('NumParams = :{}'.format(count_parameters(model)))

# < 1 increases precision, > 1 recall
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


## train loop
mlflow.start_run()
# mlflow.log_param("hidden_size", hidden_size)
mlflow.log_param("learning_rate", lr)
mlflow.log_param("num_epochs", num_epoch)

all_loss_train = []
loss_test = []
train_acc = []
test_acc = []
for iter in tqdm(range(num_epoch), position=0, leave=True):
    model.train()
    running_loss = 0
    acc = 0
    for _, data in enumerate(train_loader):
        data.to(device)
        optimizer.zero_grad()
        data.x = data.x.float()
        data.molfeature = data.molfeature.float()
        pred = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(torch.squeeze(pred), data.y.float())
        acc += torch.where((torch.where(pred >= 0.5, 1, 0).T - data.y.float()) == 0)[0].shape[0]/data.y.shape[0]
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
        pred = model(test_data.x, test_data.edge_index, test_data.batch)
        test_loss = loss_fn(torch.squeeze(pred), test_data.y.float())
        mlflow.log_metric("test_loss", test_loss, step=iter)
        calculate_metrics(torch.squeeze(torch.where(pred >= 0.5, 1, 0)), test_data.y.float(), iter, "test")

        loss_test.append(test_loss.item())
        test_acc.append(torch.where((torch.where(pred >= 0.5, 1, 0).T - test_data.y.float()) == 0)[0].shape[0]/test_data.y.shape[0]
)

    if iter % log_interval == 0:
        print('Train Epoch: {} \tTrainLoss: {:.6f} \tTestLoss: {:.6f} \tTrainACC: {:.2f} \tTestACC: {:.2f}'.format(
            iter, running_loss, test_loss.item(), train_acc[iter], test_acc[iter]))


torch.save(model.state_dict(), './saved_model/model_nomolfeat.pth')
torch.save(optimizer.state_dict(), './saved_model/optimizer1.pth')
mlflow.pytorch.log_model(model, "model_nofeat")

mlflow.end_run()

##
plt.figure()
plt.plot(all_loss_train, 'b', label='train')
plt.plot(loss_test, 'r', label='test')
plt.legend()

plt.show()