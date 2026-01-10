import os
import gc
import h5py
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

import sys
import os
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/'))
from data.processing.h5_to_pdb import create_pdb_lines_MD

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

def get_distance_from_backbone(locant_name):
    """
    Extract the distance from backbone based on Greek letter in locant name
    
    Parameters:
    -----------
    locant_name : str
        PDB atom name (e.g., 'CA', 'CB', 'CG1', 'NE2')
    
    Returns:
    --------
    int : Distance from backbone (0 for backbone, 1 for beta, 2 for gamma, etc.)
          Returns -1 if cannot determine
    """
    # Clean the name
    locant = locant_name.strip().upper()
    
    # Backbone atoms (distance = 0)
    if locant in ['N', 'CA', 'C', 'O', 'OXT']:
        return 0
    
    # Extract the Greek letter position (second character usually)
    # The pattern is: Element + GreekLetter + OptionalNumber
    
    # Greek letter mapping to distance
    greek_to_distance = {
        'A': 0,  # Alpha (backbone)
        'B': 1,  # Beta
        'G': 2,  # Gamma
        'D': 3,  # Delta
        'E': 4,  # Epsilon
        'Z': 5,  # Zeta
        'H': 6,  # Eta
    }
    
    # Check second character for Greek letter
    if len(locant) >= 2:
        greek_letter = locant[1]
        if greek_letter in greek_to_distance:
            return greek_to_distance[greek_letter]
    
    # Special cases for single letter + number (e.g., 'C1', 'N1', 'O1')
    if len(locant) == 2 and locant[1].isdigit():
        return -1  # Unknown/special atom
    
    return -1  # Cannot determine

def get_imputationmean_actualmean_actualstd_of_locantdistance_feature():
    '''
    Unique Distances: {0, 1, 2, 3, 4, 5, 6}
    Unknown Distances: 3
    Mean of Known Data: 1.30716335308741, Mean after Imputation: 1.30716335308741, Std after Imputation: 1.6262458705982417
    '''

    with open('misato-dataset/src/data/processing/Maps/atoms_residue_map.pickle', mode='rb') as f:
        residue_mapping = pickle.load(f)

    with open('misato-dataset/src/data/processing/Maps/atoms_name_map_for_pdb.pickle', mode='rb') as f:
        name_mapping = pickle.load(f)

    with open('misato-dataset/src/data/processing/Maps/atoms_type_map.pickle', mode='rb') as f:
        type_mapping = pickle.load(f)

    train_datatype_ids_file = os.path.join('', f"misato-dataset/data/MD/splits/train_MD.txt")
    train_ids = read_idx(train_datatype_ids_file)

    mddata_withoutH = h5py.File('/scratch/izar/chaurasi/md_out.hdf5', 'r')
    mddata_withH = h5py.File('/scratch/izar/chaurasi/MD.hdf5', 'r')

    all_atom_distances = []
    all_count_atoms = 0
    unknown_count = 0

    for protein_name in train_ids:

        print(protein_name)

        total_atoms_withoutH = mddata_withoutH[protein_name]['atoms_element'][()].shape[0]

        molecules_begin_atom_index = mddata_withoutH[protein_name]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]
        all_count_atoms = all_count_atoms + ligand_start

        total_atoms_withH = mddata_withH[protein_name]['atoms_element'][()].shape[0]

        pdb_lines = create_pdb_lines_MD(mddata_withH[protein_name]['trajectory_coordinates'][()][0, :, :], 
                    mddata_withH[protein_name]['atoms_type'][()],
                    mddata_withH[protein_name]['atoms_number'][()],
                    mddata_withH[protein_name]['atoms_residue'][()],
                    mddata_withH[protein_name]['molecules_begin_atom_index'][()],
                    type_mapping,
                    residue_mapping,
                    name_mapping)
        
        refined_pdb_lines = []
        for line in pdb_lines:
            if line != 'TER' and line.strip()[-5:].strip() != 'H':
                refined_pdb_lines.append(line.strip())

        for line in refined_pdb_lines[:ligand_start]:
            atom_locant = line[12:17].strip()
            distance = get_distance_from_backbone(atom_locant)
            if distance != -1:
                all_atom_distances.append(distance)
            else:
                unknown_count += 1

        assert len(pdb_lines), total_atoms_withH + len(molecules_begin_atom_index) - 1
        assert len(refined_pdb_lines), total_atoms_withoutH

    print("Unique Distances:", set(all_atom_distances))
    print("Unknown Distances:", unknown_count)

    all_atom_distances = np.array(all_atom_distances)
    assert all_count_atoms == all_atom_distances.shape[0] + unknown_count

    mean_imputation = np.mean(all_atom_distances)

    actual_mean = (np.sum(all_atom_distances) + unknown_count*mean_imputation)/(all_count_atoms)
    actual_std = np.std(np.concatenate([all_atom_distances, unknown_count*np.array([mean_imputation])]))

    print(f"Mean of Known Data: {mean_imputation}, Mean after Imputation: {actual_mean}, Std after Imputation: {actual_std}")
    return mean_imputation, actual_mean, actual_std


def create_relative_distance_feature(mean_imputation, actual_mean, actual_std):
    '''
    # Get the Mean and Std of locant distance feature from training set by running:
    get_imputationmean_actualmean_actualstd_of_locantdistance_feature()

    # Then run this function to create the relative_distance_feature in the md_out hdf5 file.
    create_relative_distance_feature(1.30716335308741, 1.30716335308741, 1.6262458705982417)
    '''

    with open('misato-dataset/src/data/processing/Maps/atoms_residue_map.pickle', mode='rb') as f:
        residue_mapping = pickle.load(f)

    with open('misato-dataset/src/data/processing/Maps/atoms_name_map_for_pdb.pickle', mode='rb') as f:
        name_mapping = pickle.load(f)

    with open('misato-dataset/src/data/processing/Maps/atoms_type_map.pickle', mode='rb') as f:
        type_mapping = pickle.load(f)

    mddata_withoutH = h5py.File('/scratch/izar/chaurasi/md_out.hdf5', 'r+')
    mddata_withH = h5py.File('/scratch/izar/chaurasi/MD.hdf5', 'r')

    all_count_atoms = 0

    for protein_name in mddata_withoutH:

        print(protein_name)

        total_atoms_withoutH = mddata_withoutH[protein_name]['atoms_element'][()].shape[0]

        molecules_begin_atom_index = mddata_withoutH[protein_name]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]
        all_count_atoms = all_count_atoms + ligand_start

        total_atoms_withH = mddata_withH[protein_name]['atoms_element'][()].shape[0]

        pdb_lines = create_pdb_lines_MD(mddata_withH[protein_name]['trajectory_coordinates'][()][0, :, :], 
                    mddata_withH[protein_name]['atoms_type'][()],
                    mddata_withH[protein_name]['atoms_number'][()],
                    mddata_withH[protein_name]['atoms_residue'][()],
                    mddata_withH[protein_name]['molecules_begin_atom_index'][()],
                    type_mapping,
                    residue_mapping,
                    name_mapping)
        
        refined_pdb_lines = []
        for line in pdb_lines:
            if line != 'TER' and line.strip()[-5:].strip() != 'H':
                refined_pdb_lines.append(line.strip())

        protein_atom_distances = []
        for line in refined_pdb_lines[:ligand_start]:
            atom_locant = line[12:17].strip()
            distance = get_distance_from_backbone(atom_locant)
            if distance != -1:
                protein_atom_distances.append(distance)
            else:
                print('Unknown atom locant:', atom_locant)
                protein_atom_distances.append(mean_imputation)

        assert len(pdb_lines), total_atoms_withH + len(molecules_begin_atom_index) - 1
        assert len(refined_pdb_lines), total_atoms_withoutH

        mddata_withoutH[protein_name]['relative_distance_feature'] = (np.array(protein_atom_distances)-actual_mean)/actual_std

    mddata_withoutH.close()
    mddata_withH.close()

def get_actualmean_actualstd_of_nodedegree_feature():
    '''
    Mean of node degree feature: 15.850955230130838, Std of node degree feature: 4.811981472952439
    '''
    T = 100

    train_datatype_ids_file = os.path.join('', f"misato-dataset/data/MD/splits/train_MD.txt")
    train_ids = read_idx(train_datatype_ids_file)

    processed_dir = '/work/lts2/users/sajal/data/full_data/train_data'

    mddata_withoutH = h5py.File('/scratch/izar/chaurasi/md_out.hdf5', 'r')

    atom_count = 0
    sum_node_degrees = 0
    sum_node_degrees_squared = 0

    for protein_name in train_ids:

        print(protein_name)

        data_dict = torch.load(os.path.join(processed_dir, f'{protein_name}.pt'))

        molecules_begin_atom_index = mddata_withoutH[protein_name]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]
        n_atom = mddata_withoutH[protein_name]['atoms_element'][()][:ligand_start].shape[0]

        atom_count += n_atom

        node_degree = degree(data_dict['edge_index'][0], num_nodes=n_atom*T)
        sum_node_degrees += torch.sum(node_degree).item()
        sum_node_degrees_squared += torch.sum(node_degree**2).item()

        assert node_degree.shape[0] == n_atom*T

        # Free memory for tensors and clear cache
        del data_dict, node_degree
        torch.cuda.empty_cache()
        gc.collect()

    atom_count = atom_count*T
    mean = sum_node_degrees / atom_count
    std = ((sum_node_degrees_squared / atom_count) - mean**2)**0.5

    print(f"Mean of node degree: {mean}, Std of node degree: {std}")
    return mean, std

def create_nodedegree_feature(node_mean, node_std, splitname):
    '''
    Get the Mean and Std of all node degrees from training set by running:
    get_actualmean_actualstd_of_nodedegree_feature()

    Then run this function for train and val splits to create the nodedegree_feature in the hdf5 files.
    create_nodedegree_feature(15.850955230130838, 4.811981472952439, 'train')
    create_nodedegree_feature(15.850955230130838, 4.811981472952439, 'val')
    '''

    T = 100

    datatype_ids_file = os.path.join('', f"misato-dataset/data/MD/splits/{splitname}_MD.txt")
    protein_ids = read_idx(datatype_ids_file)

    processed_dir = f'/work/lts2/users/sajal/data/full_data/{splitname}_data'

    mddata_withoutH = h5py.File('/scratch/izar/chaurasi/md_out.hdf5', 'r+')

    for protein_name in protein_ids:

        print(protein_name)

        data_dict = torch.load(os.path.join(processed_dir, f'{protein_name}.pt'))

        molecules_begin_atom_index = mddata_withoutH[protein_name]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]
        n_atom = mddata_withoutH[protein_name]['atoms_element'][()][:ligand_start].shape[0]

        node_degree = degree(data_dict['edge_index'][0], num_nodes=n_atom*T)

        mddata_withoutH[protein_name]['nodedegree_feature'] = (node_degree.numpy()-node_mean)/node_std

        assert node_degree.shape[0] == n_atom*T

        # Free memory for tensors and clear cache
        del data_dict, node_degree
        torch.cuda.empty_cache()
        gc.collect()

    mddata_withoutH.close()

if __name__ == "__main__":
    pass