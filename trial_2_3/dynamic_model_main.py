'''
Things to check before running:
1. resume and wandbid
2. md_out path
3. datatype_ids_file path
4. train,val,test splits path
5. time, script to run
'''

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

from models import SpatialTemporalModel3
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
folder_path = '/work/lts2/users/sajal/data/full_data/'#'/scratch/izar/chaurasi/data/full_data/'

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
gru_hidden=128
num_gru_layers = 1
gru_dropout = 0.0
static_gnn_hidden=64
static_gnn_out=64

model = SpatialTemporalModel3(node_feat_dim=len(atom_mapping),#atom_feats_list['train'][0].shape[1],
                             gnn_hidden=gnn_hidden, 
                             gnn_out=gnn_out,
                             gru_hidden=gru_hidden,
                             num_gru_layers=num_gru_layers,
                             gru_dropout=gru_dropout,
                             static_gnn_hidden=static_gnn_hidden,
                             static_gnn_out=static_gnn_out,
                             out_dim=1
                             ).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

lr = 1e-4
start_epoch = 0
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn = nn.MSELoss()
global_step = 0
best_val_loss = 99999999.0
resume = True

if resume:
    wandb.init(project="semester_project_epfl", id="32wpeock", resume='must')
    checkpoint = torch.load(os.path.join("/home/chaurasi/semesterproject/trial/wandb/CheckpointDynamicModel/model_epoch.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
else:
    # Initialize Weights and Biases
    wandb.init(project="semester_project_epfl", config={
        "gnn_hidden": gnn_hidden,
        "gnn_out": gnn_out,
        "gru_hidden": gru_hidden,
        "num_gru_layers": num_gru_layers,
        "gru_dropout": gru_dropout,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "batch_size": 1,
        "static_gnn_hidden": static_gnn_hidden,
        "static_gnn_out": static_gnn_out
    })

start = time.time()
graph_node_limit = 17500

for epoch in tqdm(range(start_epoch, epochs), desc="Training"):

    model.train()
    total_loss = 0.0
    
    for first_graph, batched_graph, target in train_dataloader:

        if (batched_graph.x.shape[0]/100) > graph_node_limit:
            print(f"Skipping graph with {batched_graph.x.shape[0]} nodes")
            continue
        elif (batched_graph.x.shape[0]/100) > 10000:
            print(f"Large graph with {batched_graph.x.shape[0]} nodes, clearing cache nonperiodically")
            torch.cuda.empty_cache()
            gc.collect()

        batched_graph = batched_graph.to(device)
        target = target.unsqueeze(-1).to(device) # (N, 1)
        first_graph = first_graph.to(device)

        pred = model(first_graph, batched_graph)     # (N, 1)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1

        del loss, batched_graph, target, pred
        
        if global_step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    avg_loss = total_loss/len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluation phase for val loss
    model.eval()
    total_val_loss = 0.0
    val_count = 0
    
    with torch.no_grad():
        for first_graph, batched_graph, target in val_dataloader:

            if (batched_graph.x.shape[0]/100) > graph_node_limit:
                continue

            batched_graph = batched_graph.to(device)
            target = target.unsqueeze(-1).to(device) # (N, 1)
            first_graph = first_graph.to(device)

            pred = model(first_graph, batched_graph)    # (N, 1)
            loss = loss_fn(pred, target)
            total_val_loss += loss.item()
            val_count += 1

            del loss, batched_graph, target, pred
            
            if val_count % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    avg_val_loss = total_val_loss/len(val_dataloader)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
    
    wandb.log({"epoch": epoch + 1, "Train_Loss": avg_loss})
    wandb.log({"epoch": epoch + 1, "Val_Loss": avg_val_loss})

    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({    'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'global_step': global_step
                }, os.path.join("/home/chaurasi/semesterproject/trial/wandb/CheckpointDynamicModel/best_model.pth"))
        print(f"✅ New best model saved with val loss: {avg_val_loss:.4f}")

    torch.save({    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'global_step': global_step
            }, os.path.join("/home/chaurasi/semesterproject/trial/wandb/CheckpointDynamicModel/model_epoch.pth"))
    print(f"✅ Model checkpoint saved at epoch {epoch+1}")


end = time.time()

print(f"Elapsed time for training: {end - start:.4f} seconds")

wandb.finish()

print('Done')