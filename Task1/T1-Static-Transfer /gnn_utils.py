import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

atom_mapping = {0:'C', 1:'N', 2:'O', 3:'F', 4:'P', 5:'S', 6:'CL', 7:'BR', 8:'I', 9: 'UNK'}

def one_of_k_encoding_unk_indices(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    one_hot_encoding = [0] * len(allowable_set)
    if x in allowable_set:
        one_hot_encoding[x] = 1
    else:
        one_hot_encoding[-1] = 1
    return one_hot_encoding

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
