import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import Data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(in_channels=1, out_channels=64)
        self.pool1 = TopKPooling(in_channels=64, ratio=0.8)

        self.conv2 = GCNConv(in_channels=64, out_channels=128)
        self.pool2 = TopKPooling(in_channels=128, ratio=0.8)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, perm, score = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _, _ = self.pool2(x, edge_index, None, batch)

        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), perm

# Sample graph data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Instantiate and forward pass through the network
model = Net()
output, perm = model(data)

# Print the permutation indices
print("Permutation Indices:", perm)
