import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def read_idx(file_name):
    with open(file_name, 'r') as f: 
        ids = f.read().splitlines()
    return ids

def build_frame_graph(coords1, coords2, atom_feats, k, distance_threshold, graph_type):
    if graph_type == 'knn':
        raise ValueError("KNN graph type is not supported in this function.")

    elif graph_type == 'threshold':
        distances = cdist(coords1, coords2, metric='euclidean')
        edge_indices = np.argwhere(distances < distance_threshold)
        edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
        edge_attr = 1.0 / (distances[edge_indices[:, 0], edge_indices[:, 1]] + 1e-6)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)

    if not torch.is_tensor(atom_feats):
        x = torch.tensor(atom_feats, dtype=torch.float32)
    else:
        x = atom_feats

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(coords1, dtype=torch.float))

def get_atom_class(atoms, residues, ATOM_MAPPING):
    """
    Retrieve atom classes based on atom and residue types.

    Parameters:
    - atoms (list): List of atom types.
    - residues (list): List of residue types.

    Returns:
    - list: List of corresponding atom classes.
    """
    return [ATOM_MAPPING[(atom, residue)] for atom, residue in zip(atoms, residues)]

def calc_atom_1hot(atom_idx, ATOM_MAPPING):
    """
    Convert atom index to one-hot encoding.

    Parameters:
    - atom_idx (int): Index of the atom.

    Returns:
    - torch.Tensor: One-hot encoded tensor for the atom.
    """
    num_classes = len(set(ATOM_MAPPING.values()))
    return torch.nn.functional.one_hot(atom_idx, num_classes=num_classes + 1)