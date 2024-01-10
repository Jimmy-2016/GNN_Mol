
from torch_geometric.explain import Explainer, GNNExplainer
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
# pred, out_pool1, edge_index1, perm1, score1, out_pool2, edge_index2, perm2, score2 = \
#     model(test_data.x, test_data.edge_index, test_data.batch)

# print(pred)

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type=None,
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',  # Model returns log probabilities.
    ),
)

model.eval()

tmpdata = Data(x=test_data.x.squeeze(), edge_index=test_data.edge_index, batch=test_data.batch)

explanation = explainer(x=tmpdata.x, edge_index=tmpdata.edge_index, batch=tmpdata.batch)
# print(explanation.edge_mask)
# print(explanation.node_mask)

explanation.visualize_feature_importance(top_k=10)

explanation.visualize_graph()