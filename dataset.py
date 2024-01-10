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
            mol = Chem.MolFromSmiles(row['mol'])

            # Get node features
            # torch.from_numpy(f.node_features).float()
            node_feats = torch.from_numpy(featurizer._featurize(mol).node_features).float()
            # Get edge features
            edge_feats = self._get_edge_features(mol)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol)
            # Get labels info
            label = self._get_labels(row["Class"])

            molfeatures = self._get_mol_features(row.iloc[6:20])

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=row["mol"],
                        molfeature=molfeatures
                        )
            # Featurize molecule
            # mol = Chem.MolFromSmiles(row['mol'])
            # f = featurizer._featurize(mol)
            # data = Data(x=torch.from_numpy(f.node_features).float())
            # # data = f.to_pyg_graph()
            # data.y = row["Class"]
            # data.smiles = row["mol"]
            # data.molfeature = row.iloc[6:20]
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))
    #
    # def _get_label(self, label):
    #     label = np.asarray([label])
    #     return torch.tensor(label, dtype=torch.int64)

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def _get_mol_features(self, molfeatures):
        # molfeatures = np.asarray([molfeatures])
        return torch.tensor(molfeatures, dtype=torch.int64)


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