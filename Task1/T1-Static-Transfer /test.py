
import gc
import sys
import os
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/'))
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/data/components/'))

import random
import numpy as np
import wandb
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import StaticModel, RolandDynamicModelGRU_EGNN
from gnn_utils import atom_mapping
from md_datasets import create_static_model_dataset, create_dataset_task1

# Version check
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

# GPU or CPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used: ", device)

def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 1
seed_everything(seed)
g = torch.Generator()
g.manual_seed(seed)

# ---------- Collect the Task 1 Features first ----------
# Initialize the dataset and dataloaders
k = 2
distance_threshold = 4.5
graph_type = 'threshold'
T = 100
folder_path = '/work/lts2/users/sajal/data/full_data/'

test_dataset = create_dataset_task1('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Test Examples in Task 1: ", len(test_dataloader))

# Load the Pretrained Model
node_dim = 35
egnn_hidden = 8
gnn_out = 16
gru_hidden = 32
num_gru_layers = 2
gru_dropout = 0.0

task1_model = RolandDynamicModelGRU_EGNN( node_feat_dim=node_dim,
                                    egnn_hidden=egnn_hidden,
                                    gnn_out=gnn_out,
                                    gru_hidden=gru_hidden,
                                    num_gru_layers=num_gru_layers,
                                    gru_dropout=gru_dropout,
                                    out_dim=1
                                ).to(device)
print('Number of trainable parameters in Task 1 model:', sum(p.numel() for p in task1_model.parameters() if p.requires_grad))

# Load the weights of the pretrained task 1 model
task1_model_checkpoint = torch.load(os.path.join("best_dynamic_model.pth"))
task1_model.load_state_dict(task1_model_checkpoint['model_state_dict'], strict = True)

for param in task1_model.parameters():
    param.requires_grad = False

# Collect the node features from the pretrained task 1 model's in a protein dict
# For test data
task1_model.eval()
test_protein_dict = {}
with torch.no_grad():
    for protein_name, batched_graph, target in tqdm(test_dataloader, desc="Collecting Task 1 Features for Test Data"):
        batched_graph = batched_graph.to(device)
        _, task1_node_features = task1_model(batched_graph)
        test_protein_dict[protein_name] = task1_node_features.cpu()

# ---------- Training Loop ----------
# Initialize the dataset and dataloaders
k = 2
distance_threshold = 4.5
graph_type = 'threshold'

class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

test_dataset = create_static_model_dataset('test', k, distance_threshold, graph_type)
test_data_list = [test_dataset.__getitem__(i, test_protein_dict) for i in tqdm(range(len(test_dataset)), desc="Preloading test data of Task 2")]
test_dataset = PreloadedDataset(test_data_list)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Test Examples: ", len(test_dataloader))

gnn_hidden=32
gnn_out=32
node_feat_dim = 10+32

model = StaticModel(node_feat_dim=node_feat_dim,
                        gnn_hidden=gnn_hidden, 
                        gnn_out=gnn_out,
                        out_dim=1
                        ).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

checkpoint = torch.load(os.path.join("/home/chaurasi/semesterproject/finalefrozen/wandb/CheckpointStaticModel_lr5e-4/best_model.pth"))
model.load_state_dict(checkpoint['model_state_dict'], strict = True)

loss_fn = nn.MSELoss()
model.eval()
total_test_loss = 0.0
test_count = 0

with torch.no_grad():
    for batched_graph, target in test_dataloader:

        batched_graph = batched_graph.to(device)
        target = target.unsqueeze(-1).to(device) # (N, 1)

        pred = model(batched_graph)     # (N, 1)
        loss = loss_fn(pred, target)
        total_test_loss += loss.item()
        test_count += 1

        del loss, batched_graph, target, pred
        
        if test_count % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

print(test_count)
avg_test_loss = total_test_loss/test_count
print("Test Loss: ", avg_test_loss)