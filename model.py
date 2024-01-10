import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class myGNN(nn.Module):
    def __init__(self, hidden_dim1=50, hidden_dim2=100, fc_hiddim=100, molfeature_dim=20, training=True):
        super(myGNN, self).__init__()

        self.conv1 = GCNConv(in_channels=30, out_channels=hidden_dim1)
        self.pool1 = TopKPooling(in_channels=hidden_dim1, ratio=0.8)

        self.conv2 = GCNConv(in_channels=hidden_dim1, out_channels=hidden_dim2)
        self.pool2 = TopKPooling(in_channels=hidden_dim2, ratio=0.8)

        self.fc_molfeature = nn.Linear(14, molfeature_dim)
        # self.fc1 = nn.Linear(2 * hidden_dim2 + molfeature_dim, fc_hiddim)
        self.fc1 = nn.Linear(2 * hidden_dim2, fc_hiddim)

        self.fc2 = nn.Linear(fc_hiddim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.training = training

    def forward(self, data):
        x, edge_index, batch, edge_attr, mol_feat = data.x, data.edge_index, data.batch, data.edge_attr, data.molfeature

        mol_feat = mol_feat.view(batch.unique().shape[0], 14)
        x = self.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index, None, batch)

        x = self.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm2, score2 = self.pool2(x, edge_index, None, batch)

        # mol_emb = self.fc_molfeature(mol_feat)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.sigmoid(x), perm1, score1, perm2, score2

