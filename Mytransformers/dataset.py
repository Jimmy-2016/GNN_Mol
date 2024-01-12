
from utils import *

# 1. Preprocess the data
class SmilesDataset(Dataset):
    def __init__(self, smiles, tokenizer, max_length=512):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_string = str(self.smiles[idx])
        tokenized = self.tokenizer(smiles_string, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

