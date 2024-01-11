import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem




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
    X = 0 * np.ones((maxlen, len(SMILES_CHARS)))
    pos = 0 * np.ones(maxlen)
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
        pos[i] = smi2index[c]
    return X, pos


def smiles_decoder(X, mode='real'):
    smi = ''
    if mode == 'B':
        X = X.argmax(axis=-1)
    for i in X:
        smi += index2smi[i]
    return smi