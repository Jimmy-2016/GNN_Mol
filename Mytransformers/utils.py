import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.impute import KNNImputer
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Tokenize and encode the SMILES strings into input tensors
def encode_data(df, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for smiles in df['Smiles']:
        if len(smiles) > 100:
            # Skip SMILES strings that exceed the maximum length threshold
            input_ids.append(torch.zeros(max_length, dtype=torch.long))
            attention_masks.append(torch.zeros(max_length, dtype=torch.long))
        else:
            encoded_dict = tokenizer.encode_plus(
                smiles,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'].squeeze(0))
            attention_masks.append(encoded_dict['attention_mask'].squeeze(0))

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    return input_ids, attention_masks



def generate_smiles(input_smiles, tokenizer, model, max_length=512):
    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(input_smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length - 10)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
        generated_smiles = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_smiles


# Define a function to process a chunk of data
def process_chunk(chunk_df):
    chunk_df['rdkit_mol'] = chunk_df['Smiles'].apply(MolFromSmiles)
    return chunk_df

# Function to return the size of an object in MB
def get_size(obj):
    return sys.getsizeof(obj) / (1024 * 1024)
def generate_fingerprints(mol):
    return GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

