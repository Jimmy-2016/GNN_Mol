import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



class mymolGen(nn.Module):
    def __init__(self, en_hidden_dim1=50, en_hidden_dim2=100,
                 latent_dim=50, decoder_hid1=100, decoder_hid2=200, max_smile=200, molfeature_dim=20, real=True):
        super(mymolGen, self).__init__()
        if real:
            self.max_smile = 200
            self.outactive = nn.ReLU()
        else:
            self.max_smile = 200 * 56
            self.outactive = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

        ## encoder
        self.en_conv1 = GCNConv(in_channels=30, out_channels=en_hidden_dim1)
        self.en_pool1 = TopKPooling(in_channels=en_hidden_dim1, ratio=0.8)

        self.en_conv2 = GCNConv(in_channels=en_hidden_dim1, out_channels=en_hidden_dim2)
        self.en_pool2 = TopKPooling(in_channels=en_hidden_dim2, ratio=0.8)

        # self.fc_molfeature = nn.Linear(14, molfeature_dim)
        # self.fc1 = nn.Linear(2 * hidden_dim2 + molfeature_dim, fc_hiddim)
        self.fc_mean = nn.Linear(2 * en_hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(2 * en_hidden_dim2, latent_dim)

        ## embed lenght
        self.fc_emd_len = nn.Embedding(200, 200)

        ## decoder

        self.fc_decoder1 = nn.Linear(latent_dim + 200, decoder_hid1)
        self.fc_decoder2 = nn.Linear(decoder_hid1, decoder_hid2)
        self.fc_out = nn.Linear(decoder_hid2, self.max_smile)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index, batch, len):

        # encoder
        x = self.relu(self.en_conv1(x, edge_index))
        out_pool1, edge_index1, _, batch, perm1, score1 = self.en_pool1(x, edge_index, None, batch)

        x = self.relu(self.en_conv2(out_pool1, edge_index1))
        out_pool2, edge_index2, _, batch, perm2, score2 = self.en_pool2(x, edge_index1, None, batch)

        # mol_emb = self.fc_molfeature(mol_feat)
        x = torch.cat([gmp(out_pool2, batch), gap(out_pool2, batch)], dim=1)
        mu = self.relu(self.fc_mean(x))
        logvar = self.relu(self.fc_logvar(x))

        # reparm
        z = self.reparameterize(mu, logvar)
        len_emb = self.relu(self.fc_emd_len(len))
        z = torch.cat([z, len_emb], dim=1)
        # decoder
        x = self.relu(self.fc_decoder1(z))
        x = self.relu(self.fc_decoder2(x)) + len_emb
        x = self.fc_out(x)

        return self.outactive(x), mu, logvar



