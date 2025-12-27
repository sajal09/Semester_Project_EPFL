import h5py
import numpy as np
import pytraj as pt
import os
from tqdm import tqdm
import shutil

# HDF5 file path
h5_file_path = '/work/lts2/users/sajal/data/md_out.hdf5'

# .nc and .top files directory (assuming filenames are {pdb_code}.nc and {pdb_code}.top)
traj_dir = '/scratch/izar/chaurasi/traj_files'

# Open the HDF5 file for modification
with h5py.File(h5_file_path, 'r+') as h5_file:
    # Get the list of all pdb_codes
    pdb_codes = list(h5_file.keys())
    
    for pdb_code in tqdm(pdb_codes, desc="Structure Progress"):
        # Structure Information
        group = h5_file[pdb_code]
        
        # Get the number of atoms from HDF5 data
        total_atoms = len(group['atoms_type'])
        
        # Construct paths for .nc and .top files
        nc_file = os.path.join(traj_dir, f"{pdb_code}.nc")
        top_file = os.path.join(traj_dir, f"{pdb_code}.top")
        
        # Load trajectory using PyTraj
        traj = pt.load(nc_file, top_file)
        
        # Remove hydrogen atoms to match HDF5 data
        traj = traj['!@H*']
        
        # Get the residue numbers from PyTraj
        pytraj_residue_numbers = [atom.resid for atom in traj.topology.atoms]
        
        # Check if the number of atoms matches
        if len(pytraj_residue_numbers) != total_atoms:
            print(f"Atom count mismatch for {pdb_code}, skipping")
            print(f"HDF5 atom count: {total_atoms}, PyTraj atom count: {len(pytraj_residue_numbers)}")
            continue
        
        # Replace the residue numbers in the HDF5 file
        if 'atoms_residue_number' in group:
            del group['atoms_residue_number']
        group.create_dataset('atoms_residue_number', data=pytraj_residue_numbers)
        
        # Print results for verification
        # print(f"First 10 residue numbers for {pdb_code}: {pytraj_residue_numbers[:10]}")
        # print(f"Last 10 residue numbers for {pdb_code}: {pytraj_residue_numbers[-10:]}")