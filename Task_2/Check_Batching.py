######### Check 1 ##########
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
from torchmetrics.classification import BinaryF1Score

from models import DynamicModelGRU, DynamicModelGRUBatched
from md_datasets import create_dataset
from torch_geometric.data import Batch

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
folder_path = '/work/lts2/users/sajal/data/task2_full_data/'#'/work/lts2/users/sajal/data/full_data/'

train_dataset = create_dataset('train', k, distance_threshold, graph_type, os.path.join(folder_path, 'train_data'), T)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], generator=g)
print("Number of Training Examples: ", len(train_dataloader))

val_dataset = create_dataset('val', k, distance_threshold, graph_type, os.path.join(folder_path, 'val_data'), T)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Validation Examples: ", len(val_dataloader))

test_dataset = create_dataset('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Test Examples: ", len(test_dataloader))

gnn_hidden=32
gnn_out=32
gru_hidden=64
num_gru_layers = 2
gru_dropout = 0.0
node_feat_dim = 21

model = DynamicModelGRU(node_feat_dim=node_feat_dim,
                             gnn_hidden=gnn_hidden, 
                             gnn_out=gnn_out,
                             gru_hidden=gru_hidden,
                             num_gru_layers=num_gru_layers,
                             gru_dropout=gru_dropout,
                             out_dim=2
                             )

modelbatched = DynamicModelGRUBatched(node_feat_dim=node_feat_dim,
                             gnn_hidden=gnn_hidden, 
                             gnn_out=gnn_out,
                             gru_hidden=gru_hidden,
                             num_gru_layers=num_gru_layers,
                             gru_dropout=gru_dropout,
                             out_dim=2
                             )
modelbatched.load_state_dict(model.state_dict())

b1 = train_dataset.__getitem__(5)[0]
b2 = train_dataset.__getitem__(55)[0]
b3 = train_dataset.__getitem__(555)[0]

H1 = model.spatial_gnn(b1)
H2 = model.spatial_gnn(b2)
H3 = model.spatial_gnn(b3)

print(H1.shape)
print(H2.shape)
print(H3.shape)

b = Batch.from_data_list([b1, b2, b3])
H = model.spatial_gnn(b)
print(H.shape)

torch.equal(H[:H1.shape[0]], H1)
torch.equal(H[H1.shape[0]:H1.shape[0]+H2.shape[0]], H2)
torch.equal(H[H1.shape[0]+H2.shape[0]:], H3)

T = 100
n1, n2, n3 = 490, 302, 231
block_lengths = [n1 * T, n2 * T, n3 * T]

Hm1, Hm2, Hm3 = torch.split(H, block_lengths, dim=0) 
torch.equal(Hm1, H1)
torch.equal(Hm2, H2)
torch.equal(Hm3, H3)

Hm1 = Hm1.reshape(T, n1, -1)   # (100, 490, 32)
Hm2 = Hm2.reshape(T, n2, -1)   # (100, 302, 32)
Hm3 = Hm3.reshape(T, n3, -1)   # (100, 231, 32)

Hm1 = torch.swapaxes(Hm1, 0, 1)
Hm2 = torch.swapaxes(Hm2, 0, 1)
Hm3 = torch.swapaxes(Hm3, 0, 1)

H1 = H1.reshape((100, -1, 32))
H2 = H2.reshape((100, -1, 32))
H3 = H3.reshape((100, -1, 32))
H1 = torch.swapaxes(H1, 0, 1)
H2 = torch.swapaxes(H2, 0, 1)
H3 = torch.swapaxes(H3, 0, 1)

torch.equal(Hm1, H1)
torch.equal(Hm2, H2)
torch.equal(Hm3, H3)

finalH = torch.cat([Hm1, Hm2, Hm3], dim=0) 
Zm, _ = model.gru(finalH)

Z1, _ = model.gru(H1)
Z2, _ = model.gru(H2)
Z3, _ = model.gru(H3)

torch.equal(Zm[:n1, :, :], Z1)
torch.equal(Zm[n1:n1+n2, :, :], Z2)
torch.equal(Zm[n1+n2:, :, :], Z3)

Z_finalm = Zm[:, -1, :] 
finalm = model.readout(Z_finalm)

torch.equal(finalm[:n1, :], model.readout(Z1[:, -1, :]))
torch.equal(finalm[n1:n1+n2, :], model.readout(Z2[:, -1, :]))
torch.equal(finalm[n1+n2:, :], model.readout(Z3[:, -1, :]))


############ Check 2 ##############
b1 = train_dataset.__getitem__(4445)[0]
b2 = train_dataset.__getitem__(56)[0]
b3 = train_dataset.__getitem__(5456)[0]
b4 = train_dataset.__getitem__(5545)[0]

block_lengths = [b1.x.shape[0], b2.x.shape[0], b3.x.shape[0], b4.x.shape[0]]
b = Batch.from_data_list([b1, b2, b3, b4])
o = modelbatched(b, block_lengths)
n1 = int(b1.x.shape[0]/100)
n2 = int(b2.x.shape[0]/100)
n3 = int(b3.x.shape[0]/100)
n4 = int(b4.x.shape[0]/100)
o1 = model(b1)
o2 = model(b2)
o3 = model(b3)
o4 = model(b4)
torch.equal(o[:n1, :], o1)
torch.equal(o[n1:n1+n2, :], o2)
torch.equal(o[n1+n2:n1+n2+n3, :], o3)
torch.equal(o[n1+n2+n3:, :], o4)
