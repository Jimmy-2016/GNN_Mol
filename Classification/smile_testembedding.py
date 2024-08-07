
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np

# Function to convert SMILES string to Morgan fingerprint
def smiles_to_fingerprint(smiles, radius=2, size=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
    return fingerprint

# Example SMILES string
smiles_string = "CN\C(=N\S(=O)(=O)c1cc(CCNC(=O)c2cc(Cl)ccc2OC)ccc1OCCOC)\S"

# Convert SMILES to Morgan fingerprint
fingerprint = smiles_to_fingerprint(smiles_string)


def smiles_to_one_hot(smiles, charset):
    # Create a dictionary to map characters to indices
    char_to_index = {char: i for i, char in enumerate(charset)}

    # Initialize the one-hot encoding vector
    one_hot_encoding = [0] * len(charset)

    # Set the corresponding indices to 1 for characters present in the SMILES
    for char in smiles:
        if char in char_to_index:
            one_hot_encoding[char_to_index[char]] = 1

    return one_hot_encoding


# Example usage
smiles = "CCO"
charset = set("".join(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))))

one_hot_encoding = smiles_to_one_hot(smiles, charset)
print(one_hot_encoding)

# Print the fingerprint
print("Original SMILES string:", smiles_string)
print("Morgan Fingerprint:", fingerprint)

plt.plot(fingerprint.ToList())

##
def smiles_to_one_hot(smiles, charset):
    # Create a dictionary to map characters to indices
    char_to_index = {char: i for i, char in enumerate(charset)}

    # Initialize the one-hot encoding vector
    one_hot_encoding = [0] * len(charset)

    # Set the corresponding indices to 1 for characters present in the SMILES
    for char in smiles:
        if char in char_to_index:
            one_hot_encoding[char_to_index[char]] = 1

    return one_hot_encoding


# Example usage
smiles = "CCO"
smiles = "CN\C(=N\S(=O)(=O)c1cc(CCNC(=O)c2cc(Cl)ccc2OC)ccc1OCCOC)\S"

charset = set("".join(Chem.MolToSmiles(Chem.MolFromSmiles(smiles))))

one_hot_encoding = smiles_to_one_hot(smiles, charset)
plt.figure()
plt.plot(one_hot_encoding)

##
SMILES_CHARS = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']

# define encoder and decoder --------------------------------------------------
smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))


def smiles_encoder(smiles, maxlen=120):
    X = np.zeros((maxlen, len(SMILES_CHARS)))
    pos = np.zeros(maxlen)
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
        pos[i] = smi2index[c]
    return X, pos


def smiles_decoder(X, mode='real'):
    smi = ''
    if mode == 'B':
        X = X.argmax(axis=-1)
    else:
        for i in X:
            smi += index2smi[i]
    return smi


# get a taste of caffeine -----------------------------------------------------
caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

smile = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

caffeine_encoding, pos = smiles_encoder(smile)

smile_decoded = smiles_decoder(pos, mode='real')

print(smile_decoded)
plt.figure()
plt.plot(pos)
# print(pos)

plt.figure()
plt.imshow(caffeine_encoding)

plt.show()