import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ---------- Dynamic Model Trial 2----------
class SpatialGNNDynamicModel2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv2(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv3(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv4(x, edge_index, edge_weight), inplace=True)
        x = self.conv5(x, edge_index, edge_weight)
        return x  # (N, out_dim)


class SpatialTemporalModel2(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                gru_hidden,
                num_gru_layers,
                gru_dropout,
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNNDynamicModel2(node_feat_dim, gnn_hidden, gnn_out)
        self.gru = nn.GRU(input_size=gnn_out, 
                          hidden_size=gru_hidden, 
                          num_layers=num_gru_layers, 
                          dropout=gru_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.readout = nn.Linear(gru_hidden, out_dim)

    def forward(self, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, len(graphs), self.gnn_out))

        # 2) Temporal GRU per atom
        Z, _ = self.gru(H_stack)          # (N, T, gru_hidden)
        # For stacking, can use batch.Batch and use id to separate

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gru_hidden)

        # 4) Per-atom prediction
        return self.readout(Z_final)     # (N, 1)

# ---------- Dynamic Model Trial 3----------
class SpatialGNNDynamicModel3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv2(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv3(x, edge_index, edge_weight), inplace=True)
        x = F.relu(self.conv4(x, edge_index, edge_weight), inplace=True)
        x = self.conv5(x, edge_index, edge_weight)
        return x  # (N, out_dim)
    
class SpatialGNNStaticModel3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight), inplace=True)
        x = self.conv2(x, edge_index, edge_weight)
        return x  # (N, out_dim)


class SpatialTemporalModel3(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                gru_hidden,
                num_gru_layers,
                gru_dropout,
                static_gnn_hidden,
                static_gnn_out,
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNNDynamicModel3(node_feat_dim, gnn_hidden, gnn_out)
        self.gru = nn.GRU(input_size=gnn_out, 
                          hidden_size=gru_hidden, 
                          num_layers=num_gru_layers, 
                          dropout=gru_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.readout = nn.Linear(gru_hidden, out_dim)

        self.spatial_gnn_static_model = SpatialGNNStaticModel3(node_feat_dim, static_gnn_hidden, static_gnn_out)
        self.readout_static = nn.Linear(static_gnn_out, out_dim)
        self.static_gnn_out = static_gnn_out

    def forward(self, first_graph, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, len(graphs), self.gnn_out))

        # 2) Temporal GRU per atom
        Z, _ = self.gru(H_stack)          # (N, T, gru_hidden)
        # For stacking, can use batch.Batch and use id to separate

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gru_hidden)

        # 4) Per-atom prediction
        per_atom_prediction = self.readout(Z_final)     # (N, 1)

        # Static model prediction
        H_static = self.spatial_gnn_static_model(first_graph).reshape((-1, self.static_gnn_out)) # (N, gnn_out)
        static_prediction = self.readout_static(H_static) # (N, 1)

        return static_prediction+per_atom_prediction     # (N, 1)


#---------- Static Model for Ablation Study ------------
class SpatialGNNStaticModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight), inplace=True)
        x = self.conv2(x, edge_index, edge_weight)
        return x  # (N, out_dim)

class SpatialModel(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNNStaticModel(node_feat_dim, gnn_hidden, gnn_out)

        self.gnn_out = gnn_out
        self.readout = nn.Linear(gnn_out, out_dim)

    def forward(self, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, self.gnn_out)) # (N, gnn_out)

        # 4) Per-atom prediction
        return self.readout(H_stack)     # (N, 1)

