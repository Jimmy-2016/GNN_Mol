import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from model import *
from dataset import *
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt


##
def smiles_to_graph(smiles):
    # Parse SMILES to obtain a molecule object
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is not None:
        # Convert RDKit molecule to NetworkX graph
        graph = nx.Graph()

        for atom in molecule.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            graph.add_node(atom_idx, atom_symbol=atom_symbol)

        for bond in molecule.GetBonds():
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())
            graph.add_edge(start_atom_idx, end_atom_idx, bond_type=bond_type)

        return graph
    else:
        print("Failed to parse SMILES string.")
        return None


def graph_to_rdkit(graph):
    mol = Chem.RWMol()

    for atom_features in graph.x:
        for f in atom_features:
            atom = Chem.Atom(f.int())
            mol.AddAtom(atom)

    for edge_start, edge_end in graph.edge_index.t().tolist():
        mol.AddBond(edge_start, edge_end, Chem.BondType.SINGLE)  # Adjust bond type if needed

    return mol

def re_arange_edge(edges, nodes):
    for i in range(edges.shape[1]):
        edges[0, i] = nodes[edges[0, i]]
        edges[1, i] = nodes[edges[1, i]]
    return edges



PATH = 'saved_model/model_nomolfeat.pth'


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


x = model.relu(model.conv1(test_data.x, test_data.edge_index))
out_pool1, edge_index1, _, batch, perm1, score1 = model.pool1(x, test_data.edge_index, None, test_data.batch)

x = model.relu(model.conv2(out_pool1, edge_index1))
out_pool2, edge_index2, _, batch, perm2, score2 = model.pool2(x, edge_index1, None, batch)

x = torch.cat([gmp(out_pool2, batch), gap(out_pool2, batch)], dim=1)
x = model.relu(model.fc1(x))
pred = torch.nn.functional.sigmoid(model.fc2(x))


print(pred)

intermediate_graph = Data(x=out_pool1.squeeze(), edge_index=edge_index1, batch=1)


## plotting

org_mol = test_data.node_stores[0]['smiles'][0]
mol = Chem.MolFromSmiles(org_mol)
img = Draw.MolToImage(mol)
img.show()


molecular_graph = smiles_to_graph(org_mol)


##
G = nx.Graph()
nodes = np.arange(test_data.x.shape[0])
G.add_nodes_from(nodes)
G.add_edges_from(test_data.edge_index.t().tolist())
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.title("Original Graph")


G = nx.Graph()
G.add_nodes_from(perm1)
G.add_edges_from(re_arange_edge(edge_index1, perm1).t().tolist())
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.title("TopK 1")


G = nx.Graph()
G.add_nodes_from(perm1[perm2])
G.add_edges_from(re_arange_edge(edge_index2, perm1[perm2]).t().tolist())
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.title("TopK 2")

print(nodes)
print(perm1)
print(perm1[perm2])

print(nodes.shape)
print(perm1.shape)
print(perm1[perm2].shape)

##
plt.figure()
plt.plot(perm1.detach().numpy(), score1.detach().numpy(), '.', ms=20, c='b')
plt.plot(perm1[perm2].detach().numpy() , score2.detach().numpy(), '.', ms=20, c='r')



plt.show()




