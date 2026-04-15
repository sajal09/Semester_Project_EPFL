import os
from md_datasets import create_dataset

# Specify whether you want to save train/val/test data here.
data_type = 'test'

# Initialize the Dataset
k = 2
distance_threshold = 4.5
graph_type = 'threshold'
folder_path = '/work/lts2/users/sajal/data/task3/task3_full_data'
T = 100

datatype_dataset = create_dataset(data_type, k, distance_threshold, graph_type, os.path.join(folder_path, f'{data_type}_data'), T)

datatype_dataset.process()