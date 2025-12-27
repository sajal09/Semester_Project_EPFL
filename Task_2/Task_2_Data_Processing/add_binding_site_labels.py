import h5py
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import os

# HDF5 file path
h5_file_path = '/work/lts2/users/sajal/data/md_out.hdf5'

# Open the HDF5 file
with h5py.File(h5_file_path, 'r+') as h5_file:
    pdb_codes = list(h5_file.keys())

    # Load atoms_type_map.pickle to get the code corresponding to 'CX' (C-alpha atom)
    with open('misato-dataset/src/data/processing/Maps/atoms_type_map.pickle', 'rb') as f:
        typeMap = pickle.load(f)
    ca_type_codes = [code for code, type_str in typeMap.items() if type_str == 'CX']
    if not ca_type_codes:
        raise ValueError("Could not find the atoms_type code corresponding to 'CX'")
    ca_type_code = ca_type_codes[0]
    print(f"The atoms_type code corresponding to 'CX' is {ca_type_code}")

    # Process the first 5 data points for testing
    for pdb_code in tqdm(pdb_codes, desc="Processing PDB codes"):
        # print(f"\nProcessing PDB code: {pdb_code}")
        group = h5_file[pdb_code]

        # Get necessary data
        atoms_coordinates = group['atoms_coordinates_ref'][()]
        atoms_type = group['atoms_type'][()]
        atoms_residue_number = group['atoms_residue_number'][()]
        molecules_begin_atom_index = group['molecules_begin_atom_index'][()]

        total_atoms = len(atoms_type)

        # Identify ligand and protein atom indices
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        num_molecules = len(molecules_begin_atom_index)

        # Assume the ligand is the last molecule
        ligand_start = molecules_begin_atom_index[-1]
        ligand_end = total_atoms
        ligand_indices = np.arange(ligand_start, ligand_end)
        ligand_coords = atoms_coordinates[ligand_indices]

        # Protein atom indices (assuming protein is the first molecule)
        protein_end = ligand_start  # All atoms before the ligand
        protein_indices = np.arange(0, protein_end)
        protein_coords = atoms_coordinates[protein_indices]
        protein_residue_numbers = atoms_residue_number[protein_indices]
        protein_atoms_type = atoms_type[protein_indices]

        # Get C-alpha atoms in the protein
        ca_mask = protein_atoms_type == ca_type_code
        ca_indices = protein_indices[ca_mask]
        ca_coords = atoms_coordinates[ca_indices]
        ca_residue_numbers = atoms_residue_number[ca_indices]

        # Compute distances between C-alpha atoms and ligand atoms
        distances = cdist(ca_coords, ligand_coords)

        # Find the minimum distance for each C-alpha atom
        min_distances = distances.min(axis=1)

        distance_threshold = 10.0  # Adjust as needed
        residue_labels = {}
        for res_num, min_dist in zip(ca_residue_numbers, min_distances):
            residue_labels[res_num] = int(min_dist <= distance_threshold)

        # Save residue numbers and labels to the HDF5 file
        unique_residue_numbers = np.sort(np.unique(atoms_residue_number))

        # Create label array
        labels = np.array([residue_labels.get(res_num, 0) for res_num in unique_residue_numbers], dtype=np.int8)

        # Delete existing datasets if they exist
        if 'residue_binding_labels' in group:
            del group['residue_binding_labels']
        if 'residue_ids' in group:
            del group['residue_ids']

        # Save datasets
        group.create_dataset('residue_ids', data=unique_residue_numbers)
        group.create_dataset('residue_binding_labels', data=labels)

        # Print the result
        num_binding_residues = np.sum(labels)
        total_residues = unique_residue_numbers[-1] + 1  # Residue numbers start from 0
        print(f"{pdb_code} has {total_residues} residues, with {num_binding_residues} binding site residues")