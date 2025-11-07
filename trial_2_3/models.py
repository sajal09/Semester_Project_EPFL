import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm

# ---------- Graph Layers -------------
class SpatialGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.norm3 = GraphNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.norm4 = GraphNorm(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, out_dim)
        self.norm5 = GraphNorm(out_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x, data.batch)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x, data.batch)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.norm3(x, data.batch)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.norm4(x, data.batch)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.norm5(x, data.batch)
        return x  # (N, out_dim)


# ---------- Static Model ----------
class StaticModel(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(node_feat_dim, 
                                      gnn_hidden, 
                                      gnn_out
                                      )

        self.gnn_out = gnn_out
        self.readout = nn.Linear(gnn_out, out_dim)

    def forward(self, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, self.gnn_out)) # (N, gnn_out)

        # 4) Per-atom prediction
        return self.readout(H_stack)     # (N, 1)


################ Trial 2 ##############################

# ---------- Dynamic Model - Transformers ----------
class DynamicModelTransformers(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                transformer_hidden,
                num_transformer_layers,
                transformer_dropout,
                num_heads,
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(node_feat_dim, gnn_hidden, gnn_out)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_out,
            nhead=num_heads,
            dim_feedforward=transformer_hidden,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.gnn_out = gnn_out
        self.readout = nn.Linear(gnn_out, out_dim)

    def forward(self, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, len(graphs), self.gnn_out))

        # 2) Temporal Transformer per atom
        Z = self.transformer(H_stack)          # (N, T, gnn_out)

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gnn_out)

        # 4) Per-atom prediction
        return self.readout(Z_final)     # (N, 1)


# ---------- Dynamic Model - LSTM ----------
class DynamicModelLSTM(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                lstm_hidden,
                num_lstm_layers,
                lstm_dropout,
                out_dim=1
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(node_feat_dim, gnn_hidden, gnn_out)
        self.lstm = nn.LSTM(input_size=gnn_out, 
                          hidden_size=lstm_hidden, 
                          num_layers=num_lstm_layers, 
                          dropout=lstm_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.readout = nn.Linear(lstm_hidden, out_dim)

    def forward(self, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, len(graphs), self.gnn_out))

        # 2) Temporal LSTM per atom
        Z, _ = self.lstm(H_stack)          # (N, T, lstm_hidden)
        # For stacking, can use batch.Batch and use id to separate

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, lstm_hidden)

        # 4) Per-atom prediction
        return self.readout(Z_final)     # (N, 1)
    

# ---------- Dynamic Model - GRU ----------
class DynamicModelGRU(nn.Module):
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
        self.spatial_gnn = SpatialGNN(node_feat_dim, gnn_hidden, gnn_out)
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

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gru_hidden)

        # 4) Per-atom prediction
        return self.readout(Z_final)     # (N, 1)
    

################ Trial 3 ##############################

#--------- Static_Dynamic Model - GRU ----------
class StaticDynamicModelGRU(nn.Module):
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
        self.spatial_gnn = SpatialGNN(node_feat_dim, 
                                      gnn_hidden, 
                                      gnn_out
                                      )
        self.gru = nn.GRU(input_size=gnn_out, 
                          hidden_size=gru_hidden, 
                          num_layers=num_gru_layers, 
                          dropout=gru_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.readout = nn.Linear(gru_hidden, out_dim)

        self.spatial_gnn_static_model = SpatialGNN(node_feat_dim, 
                                                   static_gnn_hidden, 
                                                   static_gnn_out
                                                   )
        self.readout_static = nn.Linear(static_gnn_out, out_dim)
        self.static_gnn_out = static_gnn_out

    def forward(self, first_graph, graphs):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs).reshape((-1, len(graphs), self.gnn_out))

        # 2) Temporal GRU per atom
        Z, _ = self.gru(H_stack)          # (N, T, gru_hidden)

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gru_hidden)

        # 4) Per-atom prediction
        per_atom_prediction = self.readout(Z_final)     # (N, 1)

        # Static model prediction
        H_static = self.spatial_gnn_static_model(first_graph).reshape((-1, self.static_gnn_out)) # (N, gnn_out)
        static_prediction = self.readout_static(H_static) # (N, 1)

        return static_prediction+per_atom_prediction     # (N, 1)