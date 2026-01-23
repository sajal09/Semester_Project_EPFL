import os
import h5py
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from joblib import Parallel, delayed

from gnn_utils import build_frame_graph

###################### For Task 3 Dynamic Model ######################

def normalized(feature):
    feature = feature - feature.mean()
    feature = feature / feature.std()
    return feature

# Prepare the MD Dataset
class MDTrajDataset(Dataset):
    def __init__(self, k, distance_threshold, graph_type, processed_dir, datatype_ids, mdh5_file, T):
        self.k = k
        self.distance_threshold = distance_threshold
        self.T = T
        self.graph_type = graph_type

        self.processed_dir = processed_dir
        self.datatype_ids = datatype_ids
        self.mdh5_file = mdh5_file
        self.md_H5File = h5py.File(self.mdh5_file, 'r')
        self.preprocessed_graphs_H5File = h5py.File('/work/lts2/users/sajal/data/task3/atom_graph_BindingSiteAndLigand_distance_4.5_corr_aligned_0.6.h5', 'r')

    def process(self):
        for idx in range(self.__len__()):
            
            protein_key = self.datatype_ids[idx]
            group_in = self.md_H5File[protein_key]

            # Read necessary data from the input group
            atoms_coordinates = group_in['atoms_coordinates_ref'][()]
            atoms_trajectories = group_in['trajectory_coordinates'][()]
            atoms_residue_number = group_in['atoms_residue_number'][()]
            residue_binding_labels = group_in['residue_binding_labels'][()]
            residue_ids = group_in['residue_ids'][()]
            molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]
            atoms_element = group_in['atoms_element'][()]

            # Determine indices for ligand atoms
            molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
            ligand_start = molecules_begin_atom_index[-1]
            ligand_atoms_indices = np.arange(ligand_start, len(atoms_coordinates))

            # Identify binding site residue indices
            binding_residue_indices = residue_ids[residue_binding_labels == 1]

            # Map atoms to residues
            atom_to_residue = atoms_residue_number

            # Identify binding site atom indices in protein
            binding_site_atom_indices = np.where(np.isin(atom_to_residue[:ligand_start], binding_residue_indices))[0]

            # Combine binding site atom indices and ligand atom indices
            selected_atom_indices = np.concatenate([binding_site_atom_indices, ligand_atoms_indices])

            # Build distance graph
            # distance_threshold       # Adjust as needed
            graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        atoms_trajectories[t, :, :][selected_atom_indices], 
                        np.ones(len(selected_atom_indices)), 
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


    def __getitem__(self, idx):
        protein_key = self.datatype_ids[idx]
        group_in = self.md_H5File[protein_key]

        # Read necessary data from the input group
        atoms_coordinates = group_in['atoms_coordinates_ref'][()]
        atoms_trajectories = group_in['trajectory_coordinates'][()]
        atoms_residue_number = group_in['atoms_residue_number'][()]
        residue_binding_labels = group_in['residue_binding_labels'][()]
        residue_ids = group_in['residue_ids'][()]
        molecules_begin_atom_index = group_in['molecules_begin_atom_index'][()]
        atoms_element = group_in['atoms_element'][()]

        # Determine indices for ligand atoms
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]
        ligand_atoms_indices = np.arange(ligand_start, len(atoms_coordinates))

        # Identify binding site residue indices
        binding_residue_indices = residue_ids[residue_binding_labels == 1]

        # Map atoms to residues
        atom_to_residue = atoms_residue_number

        # Identify binding site atom indices in protein
        binding_site_atom_indices = np.where(np.isin(atom_to_residue[:ligand_start], binding_residue_indices))[0]

        # Combine binding site atom indices and ligand atom indices
        selected_atom_indices = np.concatenate([binding_site_atom_indices, ligand_atoms_indices])

        # node_features = atom type + charge + ligand or not
        node_features = torch.as_tensor(self.preprocessed_graphs_H5File[protein_key]['atom_1hot'][()])
        charges = normalized(torch.as_tensor(self.preprocessed_graphs_H5File[protein_key]['charges'][()]))
        lig_index = self.preprocessed_graphs_H5File[protein_key]["ligand_begin_index"][()]
        lig_len = int(charges.shape[0]) - lig_index
        nodes_numlig = torch.cat((torch.zeros(lig_index), torch.ones(lig_len))).unsqueeze(1)
        node_features = torch.cat((node_features, charges, nodes_numlig), 1)

        # Load the graph data
        data_dict = torch.load(os.path.join(self.processed_dir, f'{protein_key}.pt'))
        batched_graph = Batch()
        batched_graph.edge_index = data_dict['edge_index'].to(torch.int64)
        batched_graph.ptr = data_dict['ptr'].to(torch.int64)
        batched_graph.batch = data_dict['batch'].to(torch.int64)
        batched_graph.edge_attr = data_dict['edge_attr']
        batched_graph.pos = torch.tensor(atoms_trajectories[:, selected_atom_indices, :].reshape(-1, 3), dtype=torch.float32)
        batched_graph.x = torch.tile(node_features, (self.T, 1))

        target = torch.tensor(self.preprocessed_graphs_H5File[protein_key]['-logKd_Ki'][()], dtype=torch.float32) # (1,)
        return batched_graph, target

    def __len__(self):
        return len(self.datatype_ids)


def create_dataset(data_type, k, distance_threshold, graph_type, folder_path, T):

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    data_type_parent_path = '/work/lts2/users/sajal/data/task3'
    if data_type == 'train':
        datatype_ids_file = os.path.join(data_type_parent_path, 'refined_remove_core_filtered_train_with_binding_site.pickle')
    elif data_type == 'val':
        datatype_ids_file = os.path.join(data_type_parent_path, 'refined_remove_core_filtered_val_with_binding_site.pickle')
    elif data_type == 'test':
        datatype_ids_file = os.path.join(data_type_parent_path, 'core_filtered_with_binding_site.pickle')
    else:
        raise ValueError("data_type should be one of 'train', 'val', or 'test'")
    
    with open(datatype_ids_file, 'rb') as f:
        datatype_ids = pickle.load(f)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = MDTrajDataset(k, distance_threshold, graph_type, folder_path, datatype_ids, mdh5_file, T)

    return datatype_dataset