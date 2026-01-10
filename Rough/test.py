
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

from models import DynamicModelGRU
from gnn_utils import atom_mapping
from md_datasets import create_dataset

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

# ---------- Training Loop ----------

# Initialize the dataset and dataloaders
k = 2
distance_threshold = 4.5
graph_type = 'threshold'
T = 100
folder_path = '/scratch/izar/chaurasi/data/full_data/'#'/work/lts2/users/sajal/data/full_data/'

test_dataset = create_dataset('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Test Examples: ", len(test_dataloader))

gnn_hidden=32
gnn_out=32
gru_hidden=64
num_gru_layers = 2
gru_dropout = 0.0

model = DynamicModelGRU(node_feat_dim=len(atom_mapping),
                             gnn_hidden=gnn_hidden, 
                             gnn_out=gnn_out,
                             gru_hidden=gru_hidden,
                             num_gru_layers=num_gru_layers,
                             gru_dropout=gru_dropout,
                             out_dim=1
                             ).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

checkpoint = torch.load(os.path.join("/work/lts2/users/sajal/finaltesting/Rec_DynamicModelGRU/best_model.pth"))
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