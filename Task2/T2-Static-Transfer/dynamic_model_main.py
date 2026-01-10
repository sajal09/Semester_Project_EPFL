'''
Things to check before running:
1. resume and wandbid
2. md_out path
3. datatype_ids_file path
4. train,val,test splits path
5. time, script to run
6. checkpoint path
7. 10000
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
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryF1Score

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import StaticModelBatched, RolandDynamicModelGRU_EGNN
from md_datasets import create_dataset_task1, create_dataset_task2

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
folder_path = '/scratch/chaurasi/data/full_data/'

train_dataset = create_dataset_task1('train', k, distance_threshold, graph_type, os.path.join(folder_path, 'train_data'), T)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], generator=g)
print("Number of Training Examples in Task 1: ", len(train_dataloader))

val_dataset = create_dataset_task1('val', k, distance_threshold, graph_type, os.path.join(folder_path, 'val_data'), T)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
print("Number of Validation Examples in Task 1: ", len(val_dataloader))

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
# For train data
task1_model.eval()
train_protein_dict = {}
with torch.no_grad():
    for protein_name, batched_graph, target in tqdm(train_dataloader, desc="Collecting Task 1 Features for Train Data"):
        batched_graph = batched_graph.to(device)
        _, task1_node_features = task1_model(batched_graph)
        train_protein_dict[protein_name] = task1_node_features.cpu()

# For val data
val_protein_dict = {}
with torch.no_grad():
    for protein_name, batched_graph, target in tqdm(val_dataloader, desc="Collecting Task 1 Features for Val Data"):
        batched_graph = batched_graph.to(device)
        _, task1_node_features = task1_model(batched_graph)
        val_protein_dict[protein_name] = task1_node_features.cpu()

# ---------- Training Loop ----------

# Initialize the dataset and dataloaders
k = 2
distance_threshold = 10
graph_type = 'threshold'
T = 100
folder_path = '/scratch/chaurasi/data/task2_full_data/'#'/work/lts2/users/sajal/data/full_data/'

def collate(x):
    return Batch.from_data_list([i[0] for i in x]), [i[0].x.shape[0] for i in x], torch.cat([i[1] for i in x], axis = 0)

class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
batch_size = 4
train_dataset = create_dataset_task2('train', k, distance_threshold, graph_type, os.path.join(folder_path, 'train_data'), T)
train_data_list = [train_dataset.__getitem__(i, train_protein_dict) for i in tqdm(range(len(train_dataset)), desc="Preloading train data of Task 2")]
train_dataset = PreloadedDataset(train_data_list)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
print("Number of Training Examples in Task 2: ", len(train_dataloader))

val_dataset = create_dataset_task2('val', k, distance_threshold, graph_type, os.path.join(folder_path, 'val_data'), T)
val_data_list = [val_dataset.__getitem__(i, val_protein_dict) for i in tqdm(range(len(val_dataset)), desc="Preloading val data of Task 2")]
val_dataset = PreloadedDataset(val_data_list)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
print("Number of Validation Examples in Task 2: ", len(val_dataloader))

test_dataset = create_dataset_task2('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
print("Number of Test Examples in Task 2: ", len(test_dataloader))

gnn_hidden=32
gnn_out=32
node_feat_dim = 53

model = StaticModelBatched(node_feat_dim=node_feat_dim,
                             gnn_hidden=gnn_hidden, 
                             gnn_out=gnn_out,
                             out_dim=2
                             ).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

lr = 5e-4
start_epoch = 0
epochs = 200
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
class_weights = torch.tensor([0.1279, 0.8721]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
global_step = 0
best_val_f1 = 0.0
resume = False
f1_metric = BinaryF1Score()
checkpoint_folder = '/work/lts2/users/sajal/temp30/wandb/CheckpointDynamicModel'

if resume:
    wandb.init(project="semester_project_epfl", id="", resume='must')
    checkpoint = torch.load(os.path.join(checkpoint_folder, "model_epoch.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step'] + 1
    best_val_f1 = checkpoint['best_val_f1']
    print(f"Resumed from epoch {start_epoch} with best val f1 {best_val_f1:.4f}")
else:
    # Initialize Weights and Biases
    wandb.init(project="semester_project_epfl", config={
        "node_feat_dim": node_feat_dim,
        "gnn_hidden": gnn_hidden,
        "gnn_out": gnn_out,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size
    })

start = time.time()
graph_node_limit = 17500

# Clear GPU cache before training starts
torch.cuda.empty_cache()
gc.collect()


for epoch in tqdm(range(start_epoch, epochs), desc="Training"):

    model.train()
    total_loss = 0.0
    train_count = 0
    
    for batched_graph, block_lengths, target in train_dataloader:

        if (batched_graph.x.shape[0]/100) > graph_node_limit:
            print(f"Skipping graph with {batched_graph.x.shape[0]} nodes")
            continue
        elif (batched_graph.x.shape[0]/100) > 10000:
            print(f"Large graph with {batched_graph.x.shape[0]} nodes, clearing cache nonperiodically")
            torch.cuda.empty_cache()
            gc.collect()

        batched_graph = batched_graph.to(device)
        target = target.to(device) # (N,)

        pred = model(batched_graph, block_lengths)     # (N, 2)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1
        train_count += 1

        # Free GPU memory
        del loss, batched_graph, target, pred
        
        if global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    avg_loss = total_loss/train_count
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "Train_Loss": avg_loss})

    # Evaluation phase for val f1
    all_preds = []
    all_targets = []
    val_count = 0
    
    if epoch % 4 == 0:
        model.eval()
        with torch.no_grad():
            for batched_graph, block_lengths, target in val_dataloader:

                if (batched_graph.x.shape[0]/100) > graph_node_limit:
                    continue

                batched_graph = batched_graph.to(device)
                target = target.to(device) # (N,)

                pred = model(batched_graph, block_lengths)     # (N, 2)

                pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
                target_labels = target.cpu().numpy()
                all_preds.extend(pred_labels)
                all_targets.extend(target_labels)
                val_count += 1

                del batched_graph, target, pred
                
                if val_count % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        print(train_count, val_count)
        avg_val_f1 = f1_metric(torch.tensor(all_preds), torch.tensor(all_targets)).item()
        print(f"Epoch {epoch+1}, Val F1: {avg_val_f1:.4f}")
        wandb.log({"epoch": epoch + 1, "Val_F1": avg_val_f1})

    scheduler.step()

    if epoch % 4 == 0:
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save({    'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_f1': best_val_f1,
                            'global_step': global_step
                    }, os.path.join(checkpoint_folder, "best_model.pth"))
            print(f"✅ New best model saved with val f1: {avg_val_f1:.4f}")

    torch.save({    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_f1': best_val_f1,
                    'global_step': global_step
            }, os.path.join(checkpoint_folder, "model_epoch.pth"))
    print(f"✅ Model checkpoint saved at epoch {epoch+1}")


end = time.time()

print(f"Elapsed time for training: {end - start:.4f} seconds")

wandb.finish()

print('Done')
