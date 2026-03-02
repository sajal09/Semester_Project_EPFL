import os
import h5py
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData

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
            all_graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        atoms_trajectories[t, :, :][selected_atom_indices],
                        atoms_trajectories[t, :, :][selected_atom_indices], 
                        np.ones((len(selected_atom_indices), 1)), 
                        self.k, 
                        self.distance_threshold, 
                        self.graph_type)
                    for t in range(self.T)
                    )
            all_batched_graph = Batch.from_data_list(all_graphs)

            # Hetero Graph construction
            prot_prot_graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        atoms_trajectories[t, :, :][binding_site_atom_indices],
                        atoms_trajectories[t, :, :][binding_site_atom_indices],
                        np.ones((len(binding_site_atom_indices), 1)), 
                        self.k, 
                        self.distance_threshold, 
                        self.graph_type)
                    for t in range(self.T)
                    )
            
            lig_lig_graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        atoms_trajectories[t, :, :][ligand_atoms_indices],
                        atoms_trajectories[t, :, :][ligand_atoms_indices],
                        np.ones((len(ligand_atoms_indices), 1)), 
                        self.k, 
                        self.distance_threshold, 
                        self.graph_type)
                    for t in range(self.T)
                    )

            prot_lig_graphs = Parallel(n_jobs=-1)(
                    delayed(build_frame_graph)(
                        atoms_trajectories[t, :, :][binding_site_atom_indices],
                        atoms_trajectories[t, :, :][ligand_atoms_indices],
                        1, 
                        self.k, 
                        self.distance_threshold, 
                        self.graph_type)
                    for t in range(self.T)
                    )
            
            hetero_graphs_list = []

            for t in range(self.T):
                data = HeteroData()
                data['protein'].x = prot_prot_graphs[t].x
                data['ligand'].x = lig_lig_graphs[t].x

                data['protein', 'bond', 'protein'].edge_index = prot_prot_graphs[t].edge_index
                data['protein', 'bond', 'protein'].edge_weight = prot_prot_graphs[t].edge_attr

                data['ligand', 'bond', 'ligand'].edge_index = lig_lig_graphs[t].edge_index
                data['ligand', 'bond', 'ligand'].edge_weight = lig_lig_graphs[t].edge_attr

                data['protein', 'interacts', 'ligand'].edge_index = prot_lig_graphs[t].edge_index
                data['protein', 'interacts', 'ligand'].edge_weight = prot_lig_graphs[t].edge_attr

                data['ligand', 'interacts', 'protein'].edge_index = torch.flip(data['protein', 'interacts', 'ligand'].edge_index, [0])
                data['ligand', 'interacts', 'protein'].edge_weight = data['protein', 'interacts', 'ligand'].edge_weight

                hetero_graphs_list.append(data)

            hetero_batched_graph = Batch.from_data_list(hetero_graphs_list)

            data_dict = {
                        "protein_protein_edge_index": hetero_batched_graph[('protein', 'bond', 'protein')]['edge_index'].to(torch.int32),
                        "protein_protein_edge_attr": hetero_batched_graph[('protein', 'bond', 'protein')]['edge_weight'],
                        "protein_protein_batch": hetero_batched_graph['protein']['batch'].to(torch.int32),
                        "protein_protein_ptr": hetero_batched_graph['protein']['ptr'].to(torch.int32),
                        "ligand_ligand_edge_index":hetero_batched_graph[('ligand', 'bond', 'ligand')]['edge_index'].to(torch.int32),
                        "ligand_ligand_edge_attr":hetero_batched_graph[('ligand', 'bond', 'ligand')]['edge_weight'],
                        "ligand_ligand_batch":hetero_batched_graph['ligand']['batch'].to(torch.int32),
                        "ligand_ligand_ptr":hetero_batched_graph['ligand']['ptr'].to(torch.int32),
                        "protein_ligand_edge_index":hetero_batched_graph[('protein', 'interacts', 'ligand')]['edge_index'].to(torch.int32),
                        "protein_ligand_edge_attr":hetero_batched_graph[('protein', 'interacts', 'ligand')]['edge_weight'],
                        "all_edge_index": all_batched_graph.edge_index.to(torch.int32),
                        "all_edge_attr": all_batched_graph.edge_attr,
                        "all_batch": all_batched_graph.batch.to(torch.int32),
                        "all_ptr": all_batched_graph.ptr.to(torch.int32)
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
        one_hot_atom_features = torch.as_tensor(self.preprocessed_graphs_H5File[protein_key]['atom_1hot'][()])
        one_hot_atom_protein = one_hot_atom_features[:len(binding_site_atom_indices), :]
        one_hot_atom_ligand = one_hot_atom_features[len(binding_site_atom_indices):, :]

        charges = normalized(torch.as_tensor(self.preprocessed_graphs_H5File[protein_key]['charges'][()]))
        charges_protein = charges[:len(binding_site_atom_indices), :]
        charges_ligand = charges[len(binding_site_atom_indices):, :]

        node_features_protein = torch.cat((one_hot_atom_protein, charges_protein, torch.zeros(one_hot_atom_protein.shape[0]).unsqueeze(1)), 1)
        node_features_ligand = torch.cat((one_hot_atom_ligand, charges_ligand, torch.ones(one_hot_atom_ligand.shape[0]).unsqueeze(1)), 1)

        #------ Load the Hetero Batched Graph data ------#
    
        # Create the dummy Hetero Batched Graph
        data = HeteroData()
        data['protein'].x = torch.ones((100,1))
        data['ligand'].x = torch.ones((20,1))

        data['protein', 'bond', 'protein'].edge_index = torch.randint(0, 100, (2, 2))
        data['protein', 'bond', 'protein'].edge_weight = torch.rand(2) 

        data['ligand', 'bond', 'ligand'].edge_index = torch.randint(0, 20, (2, 2))
        data['ligand', 'bond', 'ligand'].edge_weight = torch.rand(2)

        src_p = torch.randint(0, 100, (1, 2))
        dst_l = torch.randint(0, 20, (1, 2))
        data['protein', 'interacts', 'ligand'].edge_index = torch.cat([src_p, dst_l], dim=0)
        data['protein', 'interacts', 'ligand'].edge_weight = torch.rand(2)

        data['ligand', 'interacts', 'protein'].edge_index = torch.flip(data['protein', 'interacts', 'ligand'].edge_index, [0])
        data['ligand', 'interacts', 'protein'].edge_weight = data['protein', 'interacts', 'ligand'].edge_weight

        hetero_batched_graph = Batch.from_data_list([data, data])  # Dummy batch of size 2

        # Load the graph data from disk
        data_dict = torch.load(os.path.join(self.processed_dir, f'{protein_key}.pt'))

        # Assign the loaded data to the Hetero Batched Graph
        hetero_batched_graph['protein'].x = torch.tile(node_features_protein, (self.T, 1))
        hetero_batched_graph['protein'].batch = data_dict['protein_protein_batch'].to(torch.int64)
        hetero_batched_graph['protein'].ptr = data_dict['protein_protein_ptr'].to(torch.int64)

        hetero_batched_graph['ligand'].x = torch.tile(node_features_ligand, (self.T, 1))
        hetero_batched_graph['ligand'].batch = data_dict['ligand_ligand_batch'].to(torch.int64)
        hetero_batched_graph['ligand'].ptr = data_dict['ligand_ligand_ptr'].to(torch.int64)

        hetero_batched_graph[('protein', 'bond', 'protein')].edge_index = data_dict['protein_protein_edge_index'].to(torch.int64)
        hetero_batched_graph[('protein', 'bond', 'protein')].edge_weight = data_dict['protein_protein_edge_attr']

        hetero_batched_graph[('ligand', 'bond', 'ligand')].edge_index = data_dict['ligand_ligand_edge_index'].to(torch.int64)
        hetero_batched_graph[('ligand', 'bond', 'ligand')].edge_weight = data_dict['ligand_ligand_edge_attr']

        hetero_batched_graph[('protein', 'interacts', 'ligand')].edge_index = data_dict['protein_ligand_edge_index'].to(torch.int64)
        hetero_batched_graph[('protein', 'interacts', 'ligand')].edge_weight = data_dict['protein_ligand_edge_attr']

        hetero_batched_graph[('ligand', 'interacts', 'protein')].edge_index = torch.flip(hetero_batched_graph['protein', 'interacts', 'ligand'].edge_index, [0])
        hetero_batched_graph[('ligand', 'interacts', 'protein')].edge_weight = hetero_batched_graph['protein', 'interacts', 'ligand'].edge_weight

        #------ Load the All Batched Graph data ------#
        all_batched_graph = Batch()
        all_batched_graph.edge_index = data_dict['all_edge_index'].to(torch.int64)
        all_batched_graph.ptr = data_dict['all_ptr'].to(torch.int64)
        all_batched_graph.batch = data_dict['all_batch'].to(torch.int64)
        all_batched_graph.edge_attr = data_dict['all_edge_attr']
        all_batched_graph.pos = torch.tensor(atoms_trajectories[:, selected_atom_indices, :].reshape(-1, 3), dtype=torch.float32)

        #------ Prepare the target ------#
        target = torch.tensor(self.preprocessed_graphs_H5File[protein_key]['-logKd_Ki'][()], dtype=torch.float32) # (1,)

        return hetero_batched_graph, all_batched_graph, target

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