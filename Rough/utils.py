import h5py
import os
import numpy as np
from gnn_utils import read_idx
import matplotlib.pyplot as plt
import time

def rename_files_to_match_hdf5():
    mdh5_file = "/scratch/izar/chaurasi/data/md_out.hdf5"
    md_H5File = h5py.File(mdh5_file)

    datatype_ids = os.path.join('', "misato-dataset/data/MD/splits/train_MD.txt")
    datatype_ids = read_idx(datatype_ids)[8000:]

    new_list = []
    for key in md_H5File:
        if key in datatype_ids:
            new_list.append(key)

    print(len(new_list))

    for idx, name in enumerate(new_list):
        os.rename(os.path.join('/scratch/izar/chaurasi/data/full_data/train_data/', f'data_{idx+8000}.pt'), os.path.join('/scratch/izar/chaurasi/data/full_data/train_data/', f'{name}.pt'))


def get_statistics_on_large_graphs(filter_ids_by=None):
    '''
    Overall Stats 
    Above 10000:  2.851% 

    Above 14000:  0.318% 

    Above 16000:  0.170% 

    Above 17000:  0.100% 

    Above 17500:  0.100% 

    Only Validation Set Stats: Total 1594 
    Above 10000:  3.450% 

    Above 14000:  1.066% 

    Above 16000:  0.627% 

    Above 17000:  0.313% 

    Above 17500:  0.313% ~ 5 examples 
    '''

    start_time = time.time()

    if filter_ids_by:
        datatype_ids_file = os.path.join('', f"misato-dataset/data/MD/splits/{filter_ids_by}_MD.txt")
        filter_ids = read_idx(datatype_ids_file)
    else:
        filter_ids = None

    mdh5_file = '/scratch/izar/chaurasi/md_out.hdf5' #/work/lts2/users/sajal/data/md_out.hdf5'
    md_H5File = h5py.File(mdh5_file)

    count_list = []
    for key in md_H5File.keys():
        if filter_ids is not None and key not in filter_ids:
            continue
        molecules_begin_atom_index = md_H5File[key]['molecules_begin_atom_index'][()]
        molecules_begin_atom_index = molecules_begin_atom_index.astype(int)
        ligand_start = molecules_begin_atom_index[-1]

        atoms_element = md_H5File[key]['atoms_element'][()]
        protein_atoms_element = atoms_element[:ligand_start]
        atom_numbers = protein_atoms_element.shape[0]

        count_list.append(atom_numbers)

    count_list = np.array(count_list)
    print("Max: ", np.max(count_list))
    print("Min: ", np.min(count_list))
    print("Above 10000: ", 100*np.sum(count_list > 10000)/len(count_list))
    print("Above 14000: ", 100*np.sum(count_list > 14000)/len(count_list))
    print("Above 16000: ", 100*np.sum(count_list > 16000)/len(count_list))
    print("Above 17000: ", 100*np.sum(count_list > 17000)/len(count_list))
    print("Above 17500: ", 100*np.sum(count_list > 17500)/len(count_list))

    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    return count_list

def protein_atom_count_histogram():
    count_list = get_statistics_on_large_graphs()
    plt.hist(count_list, bins=30, color='skyblue', edgecolor='black')
    plt.title('Protein Atom Counts')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Number of Proteins')
    plt.savefig('protein_atom_count_histogram.png')

def residue_stats():
    '''
    Total Atoms:  55889160
    Total Residues:  55889160
    Max Residue ID:  22
    Min Residue ID:  0
    Unique Residues:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
    '''

    mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
    md_H5File = h5py.File(mdh5_file)
    total_atoms = 0

    all_residues = []
    for protein_name in md_H5File:
        protein_data = md_H5File[protein_name]
        
        all_residues.extend(protein_data['atoms_residue'][()])
        total_atoms += protein_data['trajectory_coordinates'][()].shape[1]

    print("Total Atoms: ", total_atoms)
    print("Total Residues: ", len(all_residues))
    print("Max Residue ID: ", np.max(all_residues))
    print("Min Residue ID: ", np.min(all_residues))
    print("Unique Residues: ", np.unique(all_residues))
