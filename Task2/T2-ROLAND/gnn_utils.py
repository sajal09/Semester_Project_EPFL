import torch
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def read_idx(file_name):
    with open(file_name, 'r') as f: 
        ids = f.read().splitlines()
    return ids

def build_frame_graph(coords, atom_feats, k, distance_threshold, graph_type):
    if graph_type == 'knn':
        N = coords.shape[0]
        nbrs = NearestNeighbors(n_neighbors=min(k+1, N)).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        send, recv, edge_attr = [], [], []
        for i in range(N):
            for j in indices[i, 1:]:
                send.append(i)
                recv.append(j)
                edge_attr.append([np.linalg.norm(coords[i]-coords[j])])

        edge_index = torch.tensor([send, recv], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    elif graph_type == 'threshold':
        distances = cdist(coords, coords, metric='euclidean')
        edge_indices = np.argwhere(distances < distance_threshold)
        edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
        edge_attr = 1.0 / (distances[edge_indices[:, 0], edge_indices[:, 1]] + 1e-6)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        edge_index = torch.tensor(edge_indices.T, dtype=torch.long)

    x = torch.tensor(atom_feats, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(coords, dtype=torch.float))

def get_ca_atoms(atoms_type, atoms_residue, atoms_coordinates, protein_indices, ca_type_code):
    protein_atoms_type = atoms_type[protein_indices]
    protein_atoms_residue = atoms_residue[protein_indices]
    protein_atoms_coordinates = atoms_coordinates[protein_indices]

    # Get Cα atom indices and coordinates
    ca_mask = protein_atoms_type == ca_type_code
    ca_indices = protein_indices[ca_mask]
    ca_coords = protein_atoms_coordinates[ca_mask]
    ca_residue_types = protein_atoms_residue[ca_mask]

    return ca_indices, ca_coords, ca_residue_types

def load_mappings():
    # Load atoms_type_map to get the code corresponding to 'CX' (Cα atoms)
    with open('misato-dataset/src/data/processing/Maps/atoms_type_map.pickle', 'rb') as f:
        typeMap = pickle.load(f)
    ca_type_codes = [code for code, type_str in typeMap.items() if type_str == 'CX']
    if not ca_type_codes:
        raise ValueError("Could not find the atoms_type code corresponding to 'CX'")
    ca_type_code = ca_type_codes[0]
    print(f"The atoms_type code corresponding to 'CX' is {ca_type_code}")

    # Load residue_Map to map residue codes to residue names
    with open('misato-dataset/src/data/processing/Maps/atoms_residue_map.pickle', 'rb') as f:
        residue_Map = pickle.load(f)
    return ca_type_code, residue_Map

def get_node_features(ca_residue_types, residue_Map):
    # Map residue codes to residue names
    residue_names = [residue_Map.get(res_type, 'UNK') for res_type in ca_residue_types]
    # Define allowable residues (20 standard amino acids + 'UNK')
    allowable_residues = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL', 'UNK'
    ]
    residue_to_idx = {res: idx for idx, res in enumerate(allowable_residues)}
    # Build node features (one-hot encoding)
    node_features = []
    for res_name in residue_names:
        one_hot = [0] * len(allowable_residues)
        idx = residue_to_idx.get(res_name, residue_to_idx['UNK'])
        one_hot[idx] = 1
        node_features.append(one_hot)
    node_features = np.array(node_features, dtype=np.float32)
    return node_features