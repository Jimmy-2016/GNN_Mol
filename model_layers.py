
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from model import *
from dataset import *
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt

PATH = './saved_model/model_nomolfeat.pth'


model = myGNN()
model.load_state_dict(torch.load(PATH))
print('Loaded!')

train_dataset = MoleculeDataset(root="data/", filename="train_bace.csv")
test_dataset = MoleculeDataset(root="data/", filename="test_bace.csv", test=True)

# Prepare training
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model.eval()
test_data = next(enumerate(test_loader))[1]
test_data.x = test_data.x.float()
test_data.molfeature = test_data.molfeature.float()
pred, out_pool1, edge_index1, perm1, score1, out_pool2, edge_index2, perm2, score2 = \
    model(test_data.x, test_data.edge_index, test_data.batch)

print(pred)

# model_children = list(model.children())
data = test_data
# topkdata = []
# for i in range(len(model_children)):
#     data = model_children[i](data.x, data.edge_index, data.batch)
#     if i == 1:
#         topkdata.append(data)

gcn_layer_params = None
for name, param in model.named_parameters():
    if 'gcn_layer' in name:
        gcn_layer_params = param


x = model.relu(model.conv1(data.x, data.edge_index))
out_pool1, edge_index1, _, batch, perm1, score1 = model.pool1(x, data.edge_index, None, data.batch)

x = model.relu(model.conv2(out_pool1, edge_index1))
out_pool2, edge_index2, _, batch, perm2, score2 = model.pool2(x, edge_index1, None, batch)

# mol_emb = self.fc_molfeature(mol_feat)
x = torch.cat([gmp(out_pool2, batch), gap(out_pool2, batch)], dim=1)
x = model.relu(model.fc1(x))
# x = model.dropout(x)
x = model.fc2(x)

print(torch.nn.Sigmoid(x))
