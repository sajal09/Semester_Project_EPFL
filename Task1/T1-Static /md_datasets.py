import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from joblib import Parallel, delayed

from gnn_utils import one_of_k_encoding_unk_indices, atom_mapping, read_idx, build_frame_graph

###################### For Dynamic Model ######################

# Prepare the MD Dataset
class MDTrajDataset(Dataset):
    def __init__(self, k, distance_threshold, graph_type, processed_dir, datatype_ids, mdh5_file, T):
        """
        coords_list: [(T, N, 3), ...]
        atom_feats_list: [(N, F), ...]
        atom_targets_list: [(N, 1), ...] one target per atom
        """
        self.k = k
        self.distance_threshold = distance_threshold
        self.T = T
        self.graph_type = graph_type
        self.processed_dir = processed_dir
        self.datatype_ids = datatype_ids
        self.mdh5_file = mdh5_file
    
    def process(self):
        for idx in range(self.__len__()):
            with h5py.File(self.mdh5_file, 'r') as md_H5File:
                protein_key = self.datatype_ids[idx]

                molecules_begin_atom_index = md_H5File[protein_key]['molecules_begin_atom_index'][()]
                molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
                ligand_start = molecules_begin_atom_index[-1]

                atoms_element = md_H5File[protein_key]['atoms_element'][()]
                protein_atoms_element = atoms_element[:ligand_start]
                protein_atom_feats = np.array([one_of_k_encoding_unk_indices(elem, atom_mapping) for elem in protein_atoms_element]) # (N, F)

                atoms_coords = md_H5File[protein_key]['trajectory_coordinates'][()] # (T, N, 3)
                protein_coords = atoms_coords[:, :ligand_start, :] # (T, N, 3)
            
            graphs = Parallel(n_jobs=-1)(
                        delayed(build_frame_graph)(protein_coords[t], protein_atom_feats, self.k, self.distance_threshold, self.graph_type)
                        for t in range(self.T)
                        )
            batched_graph = Batch.from_data_list(graphs)
            data_dict = {
                "edge_index": batched_graph.edge_index.to(torch.int32),
                "edge_attr": batched_graph.edge_attr,
                "batch": batched_graph.batch.to(torch.int32),
                "ptr": batched_graph.ptr.to(torch.int32)
                }
            torch.save(data_dict, os.path.join(self.processed_dir, f'{protein_key}.pt'))
            print(f"Saved {protein_key}.pt protein in {self.processed_dir} folder")

    def __getitem__(self, idx):

        with h5py.File(self.mdh5_file, 'r') as md_H5File:
            protein_key = self.datatype_ids[idx]

            molecules_begin_atom_index = md_H5File[protein_key]['molecules_begin_atom_index'][()]
            molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
            ligand_start = molecules_begin_atom_index[-1]

            atoms_element = md_H5File[protein_key]['atoms_element'][()]
            protein_atoms_element = atoms_element[:ligand_start]
            protein_atom_feats = np.array([one_of_k_encoding_unk_indices(elem, atom_mapping) for elem in protein_atoms_element]) # (N, F)

            atoms_coords = md_H5File[protein_key]['trajectory_coordinates'][()]
            protein_coords = atoms_coords[:, :ligand_start, :] # (T, N, 3)

            feature_atoms_adaptability = md_H5File[protein_key]['feature_atoms_adaptability'][()]
            protein_atoms_adaptability = feature_atoms_adaptability[:ligand_start] # (N, 1)

            n_atom = protein_atoms_element.shape[0]

        data_dict = torch.load(os.path.join(self.processed_dir, f'{protein_key}.pt'))
        batched_graph = Batch()

        batched_graph.edge_index = data_dict['edge_index'].to(torch.int64)
        batched_graph.ptr = torch.linspace(0, n_atom*self.T, steps=self.T+1, dtype=torch.int64)
        batched_graph.batch = torch.repeat_interleave(torch.arange(self.T, dtype=torch.int64), n_atom)
        batched_graph.edge_attr = data_dict['edge_attr']
        batched_graph.pos = torch.tensor(protein_coords.reshape(-1, 3), dtype=torch.float32)
        batched_graph.x = torch.tensor(np.tile(protein_atom_feats, (self.T, 1)), dtype=torch.float32)

        target = torch.tensor(protein_atoms_adaptability, dtype=torch.float32) # (N, 1)
        return batched_graph, target

    def __len__(self):
        return len(self.datatype_ids)

def create_dataset(data_type, k, distance_threshold, graph_type, folder_path, T):

    # Read the md file containing all the data.
    files_root =  ""

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    datatype_ids_file = os.path.join(files_root, f"misato-dataset/data/MD/splits/{data_type}_MD.txt")
    datatype_ids = read_idx(datatype_ids_file)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = MDTrajDataset(k, distance_threshold, graph_type, folder_path, datatype_ids, mdh5_file, T)

    return datatype_dataset

###################### For Static Model ######################

class StaticDataset(MDTrajDataset):
    def __init__(self, k, distance_threshold, graph_type, processed_dir, datatype_ids, mdh5_file, T):
        super().__init__(k, distance_threshold, graph_type, processed_dir, datatype_ids, mdh5_file, T)
        self.T = 1  # Override T to 1 for static dataset
        self.mdh5_file = mdh5_file
        #self.md_H5File = h5py.File(mdh5_file, 'r')  # Keep the file open for the lifetime of the dataset

    def __getitem__(self, idx):
        with h5py.File(self.mdh5_file, 'r') as md_H5File:
            protein_key = self.datatype_ids[idx]

            molecules_begin_atom_index = md_H5File[protein_key]['molecules_begin_atom_index'][()]
            molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
            ligand_start = molecules_begin_atom_index[-1]

            atoms_element = md_H5File[protein_key]['atoms_element'][()]
            protein_atoms_element = atoms_element[:ligand_start]
            protein_atom_feats = np.array([one_of_k_encoding_unk_indices(elem, atom_mapping) for elem in protein_atoms_element]) # (N, F)

            atoms_coords = md_H5File[protein_key]['atoms_coordinates_ref'][()]
            protein_coords = atoms_coords[:ligand_start] # (N, 3)
            protein_graph = build_frame_graph(protein_coords, protein_atom_feats, self.k, self.distance_threshold, self.graph_type)

            feature_atoms_adaptability = md_H5File[protein_key]['feature_atoms_adaptability'][()]
            
        protein_atoms_adaptability = feature_atoms_adaptability[:ligand_start] # (N, 1)
        target = torch.tensor(protein_atoms_adaptability, dtype=torch.float32) # (N, 1)

        return protein_graph, target
    

def create_static_model_dataset(data_type, k, distance_threshold, graph_type):

    # Read the md file containing all the data.
    files_root =  ""

    mdh5_file = "/scratch/izar/chaurasi/data/md_out.hdf5"#"/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    datatype_ids_file = os.path.join(files_root, f"misato-dataset/data/MD/splits/{data_type}_MD.txt")
    datatype_ids = read_idx(datatype_ids_file)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = StaticDataset(k, distance_threshold, graph_type, "", datatype_ids, mdh5_file, 1)

    return datatype_dataset