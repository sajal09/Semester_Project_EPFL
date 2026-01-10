import os
from md_datasets import create_dataset

# Specify whether you want to save train/val/test data here.
data_type = 'train'

# Initialize the Dataset
k = 2
distance_threshold = 10
graph_type = 'threshold'
folder_path = '/work/lts2/users/sajal/data/task2_full_data/'
T = 100

datatype_dataset = create_dataset(data_type, k, distance_threshold, graph_type, os.path.join(folder_path, f'{data_type}_data'), T)

datatype_dataset.process()