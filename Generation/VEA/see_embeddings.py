

from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torchvision
import umap
from dataset import *
from utils import *
from vae_model import *
from torch_geometric.data import DataLoader


batch_size_test = 1000

torch.manual_seed(1)
np.random.seed(1)


PATH = './saved_model/model_smile2.pth'


model = mymolGen(real=False)
model.load_state_dict(torch.load(PATH))
print('Loaded!')

train_dataset = MoleculeDataset(root="data/", filename="train_bace.csv")
test_dataset = MoleculeDataset(root="data/", filename="test_bace.csv", test=True)

# Prepare training
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

model.eval()
test_data = next(enumerate(test_loader))[1]
test_data.x = test_data.x.float()
target = test_data.y

pred, mu, sigma = model(test_data.x, test_data.edge_index, test_data.batch, test_data.len)
##

umap_hparams = {'n_neighbors': 5,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean'}


fig, ax = plt.subplots(constrained_layout=False)

umap_embedding = umap.UMAP(n_neighbors=umap_hparams['n_neighbors'], min_dist=umap_hparams['min_dist'],
                           n_components=umap_hparams['n_components'],
                           metric=umap_hparams['metric']).fit_transform(mu.detach().numpy())
scatter = ax.scatter(x=umap_embedding[:, 0], y=umap_embedding[:, 1], s=20, c=target, cmap='tab10')

cbar = plt.colorbar(scatter, boundaries=np.arange(2)-0.5)
cbar.set_ticks(np.arange(2))
cbar.set_ticklabels(np.arange(2))

plt.title('UMAP Dimensionality reduction', fontsize=25, fontweight='bold')


plt.show()