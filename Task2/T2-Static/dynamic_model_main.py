'''
Things to check before running:
1. resume and wandbid
2. md_out path
3. datatype_ids_file path
4. train,val,test splits path
5. time, script to run
6. checkpoint path
7. val count
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
from torchmetrics.classification import BinaryF1Score
from torch_geometric.data import Batch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import StaticModelBatched
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
distance_threshold = 10
graph_type = 'threshold'
T = 100
folder_path = '/work/lts2/users/sajal/data/task2_full_data/'#'/work/lts2/users/sajal/data/full_data/'

def collate(x):
    return Batch.from_data_list([i[0] for i in x]), [i[0].x.shape[0] for i in x], torch.cat([i[1] for i in x], axis = 0)

train_dataset = create_dataset('train', k, distance_threshold, graph_type, os.path.join(folder_path, 'train_data'), T)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate, generator=g)
print("Number of Training Examples: ", len(train_dataloader))

val_dataset = create_dataset('val', k, distance_threshold, graph_type, os.path.join(folder_path, 'val_data'), T)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate)
print("Number of Validation Examples: ", len(val_dataloader))

test_dataset = create_dataset('test', k, distance_threshold, graph_type, os.path.join(folder_path, 'test_data'), T)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate)
print("Number of Test Examples: ", len(test_dataloader))

gnn_hidden=32
gnn_out=32
node_feat_dim = 21

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
resume = True
f1_metric = BinaryF1Score().to(device)
checkpoint_folder = '/work/lts2/users/sajal/temp11/wandb/CheckpointDynamicModel'

if resume:
    wandb.init(project="semester_project_epfl", id="h8m8j3qu", resume='must')
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
        "batch_size": 1
    })

start = time.time()
graph_node_limit = 17500

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

        del loss, batched_graph, target, pred
        
        if global_step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    avg_loss = total_loss/train_count
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "Train_Loss": avg_loss})

    # Evaluation phase for val f1
    all_preds = []
    all_targets = []
    val_count = 0
    
    if epoch % 3 == 0:
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
                
                if val_count % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        print(train_count, val_count)
        avg_val_f1 = f1_metric(torch.tensor(all_preds), torch.tensor(all_targets)).item()
        print(f"Epoch {epoch+1}, Val F1: {avg_val_f1:.4f}")
        wandb.log({"epoch": epoch + 1, "Val_F1": avg_val_f1})

    scheduler.step()

    if epoch % 3 == 0:
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