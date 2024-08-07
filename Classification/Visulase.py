
from rdkit import Chem
from rdkit.Chem import Draw
from dataset import *
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

train_dataset = MoleculeDataset(root="data/", filename='train_bace.csv')
# test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
# params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

# Prepare training
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

# define the smiles string and covert it into a molecule sturcture ------------
sample_num = 100

sample_smile = train_dataset[sample_num].smiles
mol = Chem.MolFromSmiles(sample_smile)

# draw the modecule -----------------------------------------------------------
Draw.MolToFile(mol, 'mol.png')

# draw the molecule with property ---------------------------------------------
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

Draw.MolToFile(mol, 'mol_num.png')


##
alldata = next(enumerate(train_loader))[1]
plt.hist(alldata.y)
for _, batch in enumerate(tqdm(train_loader)):
    a = 1
    plt.hist(batch.y)

plt.show()