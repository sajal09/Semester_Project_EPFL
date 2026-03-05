import pandas as pd
import h5py
import numpy as np
import os
import pickle
import random
from md_datasets import create_dataset
from torch.utils.data import DataLoader

affinity_df = pd.read_csv("/work/lts2/users/sajal/data/task3/all_dataset_merged.csv")
general_refined_set = list(affinity_df[affinity_df.Set == 'general_refined']['PDB_code']) # 5050 ids

mdh5_file = "/work/lts2/users/sajal/data/md_out.hdf5"
md_H5File = h5py.File(mdh5_file)

# Filter the general_refined_set to include only those PDB codes that are present in the md_H5File
general_refined_set = [i for i in general_refined_set if i in md_H5File] # 4641 ids

data_type_parent_path = '/work/lts2/users/sajal/data/task3'
p_train = pickle.load(open(os.path.join(data_type_parent_path, 'refined_remove_core_filtered_train_with_binding_site.pickle'), 'rb'))
p_val = pickle.load(open(os.path.join(data_type_parent_path, 'refined_remove_core_filtered_val_with_binding_site.pickle'), 'rb'))

# Remove PDB codes in p_train and p_val from general_refined_set.
# We are checking the unaccounted PDB codes in the general_refined_set.
general_refined_set = [i for i in general_refined_set if i not in p_train]
general_refined_set = [i for i in general_refined_set if i not in p_val]
print(len(general_refined_set)) # 858 ids

def read_idx(file_name):
    with open(file_name, 'r') as f: 
        ids = f.read().splitlines()
    return ids

# We check the overlap of the unaccounted general_refined_set with MD train and val splits.
# Since the overlap is not of the ratio 90:10, we will not use this technique to split into train and val sets.
m_train = read_idx("/work/lts2/users/sajal/data/misato-dataset/data/MD/splits/train_MD.txt")
m_val = read_idx("/work/lts2/users/sajal/data/misato-dataset/data/MD/splits/val_MD.txt")
print(len([i for i in general_refined_set if i in m_train])) # 373
print(len([i for i in general_refined_set if i in m_val])) # 298

# Random split into 90:10
random.shuffle(general_refined_set)
split_idx = int(len(general_refined_set) * 0.9)
split_90 = general_refined_set[:split_idx]
split_10 = general_refined_set[split_idx:]

# Combine them with already split data by Pengkang.
f_train = split_90 + p_train
f_val = split_10 + p_val

# affinity_df.Set == 'core' gives 285 ids (core set (casp 2016)) and then we check its overlap with misato.
general_refined_set = list(affinity_df[affinity_df.Set == 'core']['PDB_code'])
general_refined_set = [i for i in general_refined_set if i in md_H5File]
f_test = general_refined_set

# There are some bad ids for which ligand data is not consistent between MD and QM datasets. We check for those bad ids and remove them from the train, test, val sets.
qm_H5File = h5py.File("/work/lts2/users/sajal/data/QM.hdf5", 'r')
bad_ids = []
for protein_key in md_H5File:
    if protein_key not in qm_H5File or qm_H5File[protein_key]["atom_properties"]["atom_properties_values"][:, 7][np.where(qm_H5File[protein_key]["atom_properties"]["atom_names"][()] != b"1")[0]].shape[0] != (md_H5File[protein_key]['atoms_coordinates_ref'][()].shape[0] - md_H5File[protein_key]['molecules_begin_atom_index'][()][-1]):
            bad_ids.append(protein_key)

f_train = [i for i in f_train if i not in bad_ids]
f_val = [i for i in f_val if i not in bad_ids]
f_test = [i for i in f_test if i not in bad_ids]

# There are some bad ids in which there is no protein atom which is less than 4.5A distance from the ligand.
# Such examples have zero protein atoms in the protein-protein graph which causes NaN loss during training.
# We remove them from the train, test, val sets.
k = 2
distance_threshold = 4.5
graph_type = 'threshold'
folder_path = '/work/lts2/users/sajal/data/task3/task3_full_data'
T = 100
train_dataset = create_dataset('train', k, distance_threshold, graph_type, os.path.join(folder_path, 'train_data'), T)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
val_dataset = create_dataset('val', k, distance_threshold, graph_type, os.path.join(folder_path, 'val_data'), T)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
test_dataset = create_dataset('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

problematic_ids = []
for i in range(len(train_dataloader)):
    hetero_graph, batched_graph, target = train_dataset.__getitem__(i)
    if hetero_graph['protein'].x.shape[0] == 0:
        problematic_ids.append(train_dataset.datatype_ids[i])

for i in range(len(val_dataloader)):
    hetero_graph, batched_graph, target = val_dataset.__getitem__(i)
    if hetero_graph['protein'].x.shape[0] == 0:
        problematic_ids.append(val_dataset.datatype_ids[i])

for i in range(len(test_dataloader)):
    hetero_graph, batched_graph, target = test_dataset.__getitem__(i)
    if hetero_graph['protein'].x.shape[0] == 0:
        problematic_ids.append(test_dataset.datatype_ids[i])

f_train = [i for i in f_train if i not in problematic_ids] # 4DHL, 4MMM, 4K7N removed
f_val = [i for i in f_val if i not in problematic_ids] # None removed
f_test = [i for i in f_test if i not in problematic_ids] # None removed

with open('train_ids.txt', 'w') as f:
    f.write('\n'.join(f_train))
with open('val_ids.txt', 'w') as f:
    f.write('\n'.join(f_val))
with open('test_ids.txt', 'w') as f:
    f.write('\n'.join(f_test))