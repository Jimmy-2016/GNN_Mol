import torch

from utils import *
from dataset import *
from vae_model import *
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize



# torch.manual_seed(1)
# np.random.seed(1)

def correct_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Use MolStandardize to standardize and correct the SMILES
            corrected_smiles = MolStandardize.standardize_smiles(smiles)
            return corrected_smiles
    except Exception as e:
        print(f"Error: {e}")
    return None


PATH = './saved_model/model_smile.pth'


model = mymolGen(real=False)
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


# reparm
mol_len = 20
z = torch.randn((1, 50))
z = torch.cat([z, model.relu(model.fc_emd_len(torch.tensor(mol_len)).unsqueeze(0))], dim=1)
# decoder
x = model.relu(model.fc_decoder1(z))
x = model.relu(model.fc_decoder2(x))
x = model.fc_out(x)

pred = model.outactive(x)

plt.figure()
res_mat = np.squeeze(pred.detach().numpy()).reshape(200, -1)
bin_mat = np.zeros_like(res_mat)
bin_mat[np.arange(len(res_mat)), res_mat.argmax(axis=1)] = 1
plt.imshow(bin_mat)

pred_smile = smiles_decoder(bin_mat.astype(int), mode='B')

# pred_smile = test_data.smiles
print(pred_smile[:mol_len])



input_smiles = pred_smile[:mol_len]
corrected_smiles = correct_smiles(input_smiles)

if corrected_smiles:
    print(f"Original SMILES: {input_smiles}")
    print(f"Corrected SMILES: {corrected_smiles}")
else:
    print("Unable to correct the SMILES.")

# test_smile = 'S1=Oc'
mol = Chem.MolFromSmiles('Brc1ccsc1CC(NC1=NC(Cc2c1ccc(Cl)c2)(C)C)C(=O)[O-]')
img = Draw.MolToImage(mol)
img.show()

plt.show()