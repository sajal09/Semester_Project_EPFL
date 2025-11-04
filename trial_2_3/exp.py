import h5py
import os
import numpy as np
from gnn_utils import read_idx
import matplotlib.pyplot as plt

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


def get_statistics_on_large_graphs():

    mdh5_file = "/scratch/izar/chaurasi/data/md_out.hdf5"
    md_H5File = h5py.File(mdh5_file)

    count_list = []
    for key in md_H5File.keys():
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
    print("Above 17500: ", np.sum(count_list > 17500))
    print("Above 17000: ", np.sum(count_list > 17000))
    print("Above 16000: ", np.sum(count_list > 16000)) 

    return count_list

def protein_atom_count_histogram():
    count_list = get_statistics_on_large_graphs()
    plt.hist(count_list, bins=30, color='skyblue', edgecolor='black')
    plt.title('Protein Atom Counts')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Number of Proteins')
    plt.savefig('protein_atom_count_histogram.png')