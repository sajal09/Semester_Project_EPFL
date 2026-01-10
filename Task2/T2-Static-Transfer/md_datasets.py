import os
import h5py
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from joblib import Parallel, delayed

from gnn_utils import read_idx, build_frame_graph, get_ca_atoms, load_mappings, get_node_features, one_of_k_encoding_unk_indices, atom_mapping

###################### For Task 2 Dynamic Model ######################

# Prepare the MD Dataset
class MDTrajDatasetTask2(Dataset):
    def __init__(self, k, distance_threshold, graph_type, processed_dir, datatype_ids, mdh5_file, T):
        self.k = k
        self.distance_threshold = distance_threshold
        self.T = T
        self.graph_type = graph_type

        self.processed_dir = processed_dir
        self.datatype_ids = datatype_ids
        self.mdh5_file = mdh5_file
        self.md_H5File = h5py.File(self.mdh5_file, 'r')

        self.ca_type_code, self.residue_mapping = load_mappings()

    def process(self):
        for idx in range(self.__len__()):
            
            protein_key = self.datatype_ids[idx]
            group_in = self.md_H5File[protein_key]

            # Read necessary data
            atoms_coordinates = group_in['atoms_coordinates_ref'][()]
            atoms_type = group_in['atoms_type'][()]
            atoms_residue_number = group_in['atoms_residue_number'][()]
            atoms_residue = group_in['atoms_residue'][()]
            residue_binding_labels = group_in['residue_binding_labels'][()]
            residue_ids = group_in['residue_ids'][()]
            molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]
            atom_trajectories = group_in['trajectory_coordinates'][()]

            # Get total number of atoms
            total_atoms = len(atoms_type)

            # Identify ligand and protein atom indices
            molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
            num_molecules = len(molecules_begin_atom_index)

            # Assume ligand is the last molecule
            ligand_start = molecules_begin_atom_index[-1]
            ligand_end = total_atoms
            ligand_indices = np.arange(ligand_start, ligand_end)

            # Protein atom indices (assuming protein is before the ligand)
            protein_end = ligand_start  # All atoms before the ligand
            protein_indices = np.arange(0, protein_end)

            # Get Cα atom indices, coordinates, and residue types
            ca_indices, ca_coords, ca_residue_types = get_ca_atoms(
                atoms_type, atoms_residue, atoms_coordinates, protein_indices, self.ca_type_code)

            # Build distance graph
            # distance_threshold       # Adjust as needed
            graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        get_ca_atoms(atoms_type, atoms_residue, atom_trajectories[t, :, :], protein_indices, self.ca_type_code)[1], 
                        np.ones(ca_coords.shape[0]), 
                        self.k, 
                        self.distance_threshold, 
                        self.graph_type)
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

    def __getitem__(self, idx, task1_features_dict):
        
        protein_key = self.datatype_ids[idx]
        group_in = self.md_H5File[protein_key]

        # Read necessary data
        atoms_coordinates = group_in['atoms_coordinates_ref'][()]
        atoms_type = group_in['atoms_type'][()]
        atoms_residue_number = group_in['atoms_residue_number'][()]
        atoms_residue = group_in['atoms_residue'][()]
        residue_binding_labels = group_in['residue_binding_labels'][()]
        residue_ids = group_in['residue_ids'][()]
        molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]

        # Get total number of atoms
        total_atoms = len(atoms_type)

        # Identify ligand and protein atom indices
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        num_molecules = len(molecules_begin_atom_index)

        # Assume ligand is the last molecule
        ligand_start = molecules_begin_atom_index[-1]
        ligand_end = total_atoms
        ligand_indices = np.arange(ligand_start, ligand_end)

        # Protein atom indices (assuming protein is before the ligand)
        protein_end = ligand_start  # All atoms before the ligand
        protein_indices = np.arange(0, protein_end)

        # Get Cα atom indices, coordinates, and residue types
        ca_indices, ca_coords, ca_residue_types = get_ca_atoms(
        atoms_type, atoms_residue, atoms_coordinates, protein_indices, self.ca_type_code)

        # Build node features
        task2_node_features = get_node_features(ca_residue_types, self.residue_mapping)
        task1_features = task1_features_dict[protein_key]  # (Protein Atoms, Feature Dim)
        ca_task1_features = task1_features[ca_indices, :]  # (Cα Atoms, Feature Dim)
        node_features = np.hstack((task2_node_features, ca_task1_features))  # (Cα Atoms, Combined Feature Dim)

        # Build node labels (binding site labels)
        residue_num_to_label = dict(zip(residue_ids, residue_binding_labels))
        # Map ca_residue_numbers to residue_ids
        ca_residue_numbers = atoms_residue_number[ca_indices]
        node_labels = [residue_num_to_label[res_num] for res_num in ca_residue_numbers]
        node_labels = np.array(node_labels, dtype=np.int64)

        # Create the graph data
        resgraph = build_frame_graph(ca_coords, 
                                    node_features, 
                                    self.k, 
                                    self.distance_threshold, 
                                    self.graph_type)
        batched_resgraph = Batch.from_data_list([resgraph])

        batched_graph = Batch()
        batched_graph.edge_index = batched_resgraph.edge_index
        batched_graph.ptr = batched_resgraph.ptr
        batched_graph.batch = batched_resgraph.batch
        batched_graph.edge_attr = batched_resgraph.edge_attr
        batched_graph.pos = batched_resgraph.pos
        batched_graph.x = batched_resgraph.x

        target = torch.tensor(node_labels, dtype=torch.int64) # (N,)
        return batched_graph, target

    def __len__(self):
        return len(self.datatype_ids)


def create_dataset_task2(data_type, k, distance_threshold, graph_type, folder_path, T):

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    datatype_ids_file = os.path.join(f"misato-dataset/data/MD/splits/{data_type}_MD.txt")
    datatype_ids = read_idx(datatype_ids_file)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = MDTrajDatasetTask2(k, distance_threshold, graph_type, folder_path, datatype_ids, mdh5_file, T)

    return datatype_dataset


###################### For Task 1 ######################

# Prepare the MD Dataset
class MDTrajDatasetTask1(Dataset):
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
        self.md_H5File = h5py.File(self.mdh5_file, 'r')

        with open('misato-dataset/src/data/processing/Maps/atoms_residue_map.pickle', mode='rb') as f:
            self.residue_mapping = pickle.load(f)
    
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

        protein_key = self.datatype_ids[idx]

        molecules_begin_atom_index = self.md_H5File[protein_key]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]

        atoms_element = self.md_H5File[protein_key]['atoms_element'][()]
        protein_atoms_element = atoms_element[:ligand_start]
        protein_atom_feats = np.array([one_of_k_encoding_unk_indices(elem, atom_mapping) for elem in protein_atoms_element]) # (N, F)

        atoms_residues = self.md_H5File[protein_key]['atoms_residue'][()]
        protein_residues = atoms_residues[:ligand_start]
        protein_atom_residue_feats = np.array([one_of_k_encoding_unk_indices(elem, self.residue_mapping) for elem in protein_residues]) # (N, R)

        protein_atom_relative_distance_feature = self.md_H5File[protein_key]['relative_distance_feature'][()]

        protein_node_degree_feature = self.md_H5File[protein_key]['nodedegree_feature'][()]

        combined_protein_feats = np.hstack((protein_atom_feats, protein_atom_residue_feats, protein_atom_relative_distance_feature[:, np.newaxis]))
        combined_protein_feats = np.tile(combined_protein_feats, (self.T, 1))
        combined_protein_feats = np.hstack((combined_protein_feats, protein_node_degree_feature[:, np.newaxis]))
        combined_protein_feats = torch.tensor(combined_protein_feats, dtype=torch.float32)

        atoms_coords = self.md_H5File[protein_key]['trajectory_coordinates'][()]
        protein_coords = atoms_coords[:, :ligand_start, :] # (T, N, 3)

        feature_atoms_adaptability = self.md_H5File[protein_key]['feature_atoms_adaptability'][()]
        protein_atoms_adaptability = feature_atoms_adaptability[:ligand_start] # (N, 1)

        n_atom = protein_atoms_element.shape[0]

        data_dict = torch.load(os.path.join(self.processed_dir, f'{protein_key}.pt'))
        batched_graph = Batch()

        batched_graph.edge_index = data_dict['edge_index'].to(torch.int64)
        batched_graph.ptr = torch.linspace(0, n_atom*self.T, steps=self.T+1, dtype=torch.int64)
        batched_graph.batch = torch.repeat_interleave(torch.arange(self.T, dtype=torch.int64), n_atom)
        batched_graph.edge_attr = data_dict['edge_attr']
        batched_graph.pos = torch.tensor(protein_coords.reshape(-1, 3), dtype=torch.float32)
        batched_graph.x = combined_protein_feats

        target = torch.tensor(protein_atoms_adaptability, dtype=torch.float32) # (N, 1)
        return protein_key, batched_graph, target

    def __len__(self):
        return len(self.datatype_ids)

def create_dataset_task1(data_type, k, distance_threshold, graph_type, folder_path, T):

    # Read the md file containing all the data.
    files_root =  ""

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    datatype_ids_file = os.path.join(files_root, f"misato-dataset/data/MD/splits/{data_type}_MD.txt")
    datatype_ids = read_idx(datatype_ids_file)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = MDTrajDatasetTask1(k, distance_threshold, graph_type, folder_path, datatype_ids, mdh5_file, T)

    return datatype_dataset