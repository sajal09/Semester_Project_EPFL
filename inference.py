import gc
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from md_datasets import create_static_model_dataset
from models import StaticModel
from gnn_utils import atom_mapping

def check_RMSE_on_Test_Set():
    '''
    Test RMSE: 0.4381
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    k = 2
    distance_threshold = 4.5
    graph_type = 'threshold'

    test_dataset = create_static_model_dataset('test', k, distance_threshold, graph_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    print("Number of Test Examples: ", len(test_dataloader))

    gnn_hidden=32
    gnn_out=32
    model = StaticModel(node_feat_dim=len(atom_mapping),
                        gnn_hidden=gnn_hidden, 
                        gnn_out=gnn_out,
                        out_dim=1
                        ).to(device)

    checkpoint_folder = '/home/chaurasi/semesterproject/trial_2_3/wandb/CheckpointStaticModel'
    checkpoint = torch.load(os.path.join(checkpoint_folder, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluation phase for test loss
    loss_fn = nn.MSELoss()
    model.eval()
    total_test_loss = 0.0
    test_count = 0

    with torch.no_grad():
        for batched_graph, target in tqdm(test_dataloader):

            batched_graph = batched_graph.to(device)
            target = target.unsqueeze(-1).to(device) # (N, 1)

            pred = model(batched_graph)     # (N, 1)
            loss = loss_fn(pred, target)
            total_test_loss += loss.item()
            test_count += 1

            del loss, batched_graph, target, pred

            if test_count % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    assert test_count == len(test_dataloader)
    avg_test_loss = total_test_loss/test_count

    print(f"Test RMSE: {avg_test_loss**0.5:.4f}")
    return avg_test_loss**0.5


if __name__ == "__main__":
    check_RMSE_on_Test_Set()