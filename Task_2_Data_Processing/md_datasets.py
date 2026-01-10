import os
import pickle
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from joblib import Parallel, delayed

from gnn_utils import one_of_k_encoding_unk_indices, atom_mapping, read_idx, build_frame_graph

###################### For Task 2 Dynamic Model ######################

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

        self.ca_type_code, self.residue_mapping = load_mappings()

    def process(self):
        for idx in range(self.__len__()):
            with h5py.File(self.mdh5_file, 'r') as md_H5File:
                protein_key = self.datatype_ids[idx]
                group_in = md_H5File[protein_key]

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

    def __len__(self):
        return len(self.datatype_ids)


def create_dataset(data_type, k, distance_threshold, graph_type, folder_path, T):

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    #md_H5File = h5py.File(mdh5_file)

    # Read the protein ids corresponding to data_type
    datatype_ids_file = os.path.join(f"misato-dataset/data/MD/splits/{data_type}_MD.txt")
    datatype_ids = read_idx(datatype_ids_file)

    print(f"Loading data from {datatype_ids_file}")

    # Initialize the Dataset
    datatype_dataset = MDTrajDataset(k, distance_threshold, graph_type, folder_path, datatype_ids, mdh5_file, T)

    return datatype_dataset