import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
from torch_geometric.data import Data


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.filename = filename
        self.test = test

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row['mol'])
            f = featurizer._featurize(mol)
            data = Data(x=torch.from_numpy(f.node_features).float())
            # data = f.to_pyg_graph()
            data.y = self._get_label(row["Class"])
            data.smiles = row["mol"]
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data


if __name__ == '__main__':
    train_dataset = MoleculeDataset(root="data/", filename="train_bace.csv", test=False)
    test_dataset = MoleculeDataset(root="data/", filename="test_bace.csv", test=True)