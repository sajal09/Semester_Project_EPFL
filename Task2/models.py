import torch
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


# ---------- T2-Static and T2-Static-Transfer ----------
class StaticModelBatched(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                out_dim
                ):
        super().__init__()
        self.spatial_gnn = SpatialGNN(node_feat_dim, 
                                      gnn_hidden, 
                                      gnn_out
                                      )

        self.gnn_out = gnn_out
        self.readout = nn.Linear(gnn_out, out_dim)

    def forward(self, graphs, block_lengths):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs)
        
        H_blocks = torch.split(H_stack, block_lengths, dim=0)

        T = 1
        processed_blocks = []

        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        H_stack = torch.cat(processed_blocks, dim=0) # (N, T, gnn_out)

        # 3) Final state for each atom
        H_stack = H_stack[:, -1, :]            # (N, gnn_out)

        # 4) Per-atom prediction
        return self.readout(H_stack)     # (N, 2)


# ---------- T2-GCRN ----------
class DynamicModelGRUBatched(nn.Module):
    def __init__(self, 
                node_feat_dim, 
                gnn_hidden, 
                gnn_out, 
                gru_hidden,
                num_gru_layers,
                gru_dropout,
                out_dim
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

    def forward(self, graphs, block_lengths):
        # 1) Run GNN on each frame
        H_stack = self.spatial_gnn(graphs)

        H_blocks = torch.split(H_stack, block_lengths, dim=0)

        T = 100
        processed_blocks = []

        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        H_stack = torch.cat(processed_blocks, dim=0)

        # 2) Temporal GRU per atom
        Z, _ = self.gru(H_stack)          # (N, T, gru_hidden)

        # 3) Final hidden state for each atom
        Z_final = Z[:, -1, :]            # (N, gru_hidden)

        # 4) Per-atom prediction
        return self.readout(Z_final)     # (N, 2)


# ---------- T2-ROLAND ----------
class RolandDynamicModelGRUBatched(nn.Module):
    def __init__(self, 
                node_feat_dim,  
                gnn_out, 
                gru_hidden,
                num_gru_layers,
                gru_dropout,
                out_dim=1
                ):
        super().__init__()

        self.spatialgnn1 = GCNConv(node_feat_dim, gnn_out)
        self.norm1 = GraphNorm(gnn_out)

        self.spatialgnn2 = GCNConv(gru_hidden, gnn_out)
        self.norm2 = GraphNorm(gnn_out)

        self.spatialgnn3 = GCNConv(gru_hidden, gnn_out)
        self.norm3 = GraphNorm(gnn_out)

        self.spatialgnn4 = GCNConv(gru_hidden, gnn_out)
        self.norm4 = GraphNorm(gnn_out)

        self.spatialgnn5 = GCNConv(gru_hidden, gnn_out)
        self.norm5 = GraphNorm(gnn_out)

        self.gru = nn.GRU(input_size=gnn_out, 
                          hidden_size=gru_hidden, 
                          num_layers=num_gru_layers, 
                          dropout=gru_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.gru_hidden = gru_hidden
        self.readout = nn.Linear(gru_hidden, out_dim)

    def forward(self, graphs, block_lengths, block_lengths_dyn):

        x, edge_index, edge_weight, graph_batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch
        
        # 1) Run GNN1 on each frame (N_atom * T, node_feat_dim) -> (N_atom, T, gnn_out)
        x = self.spatialgnn1(x, edge_index, edge_weight)
        x = self.norm1(x, graph_batch)
        x = F.relu(x)

        H_blocks = torch.split(x, block_lengths, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = x.reshape((len(graphs), -1, self.gnn_out))
        #x = torch.swapaxes(x, 0, 1)

        # 2) Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # 3) Run GNN2 on each frame (N_atom, T, gru_hidden) -> (N_atom, T, gnn_out)
        H_blocks = torch.split(x, block_lengths_dyn, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths_dyn):
            n = int(L) 
            Hm = torch.swapaxes(Hm, 0, 1)
            Hm = Hm.reshape((-1, self.gru_hidden))
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = torch.swapaxes(x, 0, 1)
        #x = x.reshape((-1, self.gru_hidden))
        
        x = self.spatialgnn2(x, edge_index, edge_weight)
        x = self.norm2(x, graph_batch)
        x = F.relu(x)
        
        H_blocks = torch.split(x, block_lengths, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = x.reshape((len(graphs), -1, self.gnn_out))
        #x = torch.swapaxes(x, 0, 1)

        # 4) Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # 5) Run GNN3 on each frame (N_atom, T, gru_hidden) -> (N_atom, T, gnn_out)
        H_blocks = torch.split(x, block_lengths_dyn, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths_dyn):
            n = int(L) 
            Hm = torch.swapaxes(Hm, 0, 1)
            Hm = Hm.reshape((-1, self.gru_hidden))
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = torch.swapaxes(x, 0, 1)
        #x = x.reshape((-1, self.gru_hidden)) 

        x = self.spatialgnn3(x, edge_index, edge_weight)
        x = self.norm3(x, graph_batch)
        x = F.relu(x)

        H_blocks = torch.split(x, block_lengths, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = x.reshape((len(graphs), -1, self.gnn_out))
        #x = torch.swapaxes(x, 0, 1)

        # 6) Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # 7) Run GNN4 on each frame (N_atom, T, gru_hidden) -> (N_atom, T, gnn_out)
        H_blocks = torch.split(x, block_lengths_dyn, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths_dyn):
            n = int(L) 
            Hm = torch.swapaxes(Hm, 0, 1)
            Hm = Hm.reshape((-1, self.gru_hidden))
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = torch.swapaxes(x, 0, 1)
        #x = x.reshape((-1, self.gru_hidden))

        x = self.spatialgnn4(x, edge_index, edge_weight)
        x = self.norm4(x, graph_batch)
        x = F.relu(x)
        
        H_blocks = torch.split(x, block_lengths, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = x.reshape((len(graphs), -1, self.gnn_out))
        #x = torch.swapaxes(x, 0, 1)

        # 8) Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # 9) Run GNN5 on each frame (N_atom, T, gru_hidden) -> (N_atom, T, gnn_out)
        H_blocks = torch.split(x, block_lengths_dyn, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths_dyn):
            n = int(L) 
            Hm = torch.swapaxes(Hm, 0, 1)
            Hm = Hm.reshape((-1, self.gru_hidden))
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = torch.swapaxes(x, 0, 1)
        #x = x.reshape((-1, self.gru_hidden))
        
        x = self.spatialgnn5(x, edge_index, edge_weight)
        x = self.norm5(x, graph_batch)
        x = F.relu(x)
        
        H_blocks = torch.split(x, block_lengths, dim=0)
        T = 100
        processed_blocks = []
        for Hm, L in zip(H_blocks, block_lengths):
            n = int(L / T) 
            Hm = Hm.reshape(T, n, self.gnn_out)
            Hm = torch.swapaxes(Hm, 0, 1)
            processed_blocks.append(Hm)

        x = torch.cat(processed_blocks, dim=0)
        #x = x.reshape((len(graphs), -1, self.gnn_out))
        #x = torch.swapaxes(x, 0, 1)

        # 10) Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # 11) Final hidden state for each atom (N_atom, T, gru_hidden) -> (N_atom, gru_hidden)
        x = x[:, -1, :]

        # 4) Per-atom prediction (N_atom, gru_hidden) -> (N_atom, 1)
        return self.readout(x)
    

# ---------- Task 1 Model T1-ROLAND-EGNN needed for T2-Static-Transfer ----------
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    count = torch.clamp(count, min=1)
    return result / count

class E_GCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer for graph level tasks.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, coords_agg='mean', 
                 tanh=False, update_coords=False):  # Default to False
        super(E_GCL, self).__init__()
        
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.update_coords = update_coords  # Store the flag
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf*2+edges_in_d+1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(input_nf+hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )
        
        # Only create coordinate MLP if we're updating coordinates
        if self.update_coords:
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            self.coord_mlp = nn.Sequential(*coord_mlp)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        
        out = self.edge_mlp(out)
        
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        
        if self.residual:
            out = x + out
        
        return out, agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
            
        return radial, coord_diff

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        
        # Add clipping for stability
        max_trans = 10.0
        trans = torch.clamp(trans, -max_trans, max_trans)
        
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter: %s' % self.coords_agg)
            
        # Additional clipping for stability
        agg = torch.clamp(agg, -max_trans, max_trans)
        coord = coord + agg
        
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        
        # Only update coordinates if flag is True
        if self.update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_feat
    
    
class RolandDynamicModelGRU_EGNN(nn.Module):
    def __init__(self, 
                node_feat_dim,
                egnn_hidden,
                gnn_out, 
                gru_hidden,
                num_gru_layers,
                gru_dropout,
                out_dim=1
                ):
        super().__init__()

        self.conv1 = E_GCL(node_feat_dim, gnn_out, egnn_hidden, edges_in_d=1, act_fn=nn.ReLU(), 
                 residual=False, attention=False, normalize=True, coords_agg='mean', 
                 tanh=False, update_coords=False)
        self.norm1 = GraphNorm(gnn_out)

        self.conv2 = E_GCL(gru_hidden, gnn_out, egnn_hidden, edges_in_d=1, act_fn=nn.ReLU(), 
                 residual=False, attention=False, normalize=True, coords_agg='mean', 
                 tanh=False, update_coords=False)
        self.norm2 = GraphNorm(gnn_out)

        self.conv3 = E_GCL(gru_hidden, gnn_out, egnn_hidden, edges_in_d=1, act_fn=nn.ReLU(), 
                 residual=False, attention=False, normalize=True, coords_agg='mean', 
                 tanh=False, update_coords=False)
        self.norm3 = GraphNorm(gnn_out)

        self.conv4 = E_GCL(gru_hidden, gnn_out, egnn_hidden, edges_in_d=1, act_fn=nn.ReLU(), 
                 residual=False, attention=False, normalize=True, coords_agg='mean', 
                 tanh=False, update_coords=False)
        self.norm4 = GraphNorm(gnn_out)

        self.conv5 = E_GCL(gru_hidden, gnn_out, egnn_hidden, edges_in_d=1, act_fn=nn.ReLU(), 
                 residual=False, attention=False, normalize=True, coords_agg='mean', 
                 tanh=False, update_coords=False)
        self.norm5 = GraphNorm(gnn_out)

        self.gru = nn.GRU(input_size=gnn_out, 
                          hidden_size=gru_hidden, 
                          num_layers=num_gru_layers, 
                          dropout=gru_dropout, 
                          batch_first=True
                          )
        self.gnn_out = gnn_out
        self.gru_hidden = gru_hidden
        self.readout = nn.Linear(gru_hidden, out_dim)

    def forward(self, graphs):
        # Run EGNN1 on each frame (N_atom*T, node_feat_dim) -> (N_atom*T, gnn_out)
        x, edge_index, edge_weight, pos = graphs.x, graphs.edge_index, graphs.edge_attr.unsqueeze(-1), graphs.pos
        x, pos, _ = self.conv1(x, edge_index, pos, edge_weight)
        x = self.norm1(x, graphs.batch)
        x = F.relu(x)

        # (N_atom*T, gnn_out) -> (T, N_atom, gnn_out) -> (N_atom, T, gnn_out)
        x = x.reshape((len(graphs), -1, self.gnn_out))
        x = torch.swapaxes(x, 0, 1)

        # Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # (N_atom, T, gru_hidden) -> (T, N_atom, gru_hidden) -> (N_atom*T, gru_hidden)
        x = torch.swapaxes(x, 0, 1)
        x = x.reshape((-1, self.gru_hidden))

        # Run EGNN2 on each frame (N_atom*T, gru_hidden) -> (N_atom*T, gnn_out)
        x, pos, _ = self.conv2(x, edge_index, pos, edge_weight)
        x = self.norm2(x, graphs.batch)
        x = F.relu(x)

        # (N_atom*T, gnn_out) -> (T, N_atom, gnn_out) -> (N_atom, T, gnn_out)
        x = x.reshape((len(graphs), -1, self.gnn_out))
        x = torch.swapaxes(x, 0, 1)

        # Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # (N_atom, T, gru_hidden) -> (T, N_atom, gru_hidden) -> (N_atom*T, gru_hidden)
        x = torch.swapaxes(x, 0, 1)
        x = x.reshape((-1, self.gru_hidden))

        # Run EGNN3 on each frame (N_atom*T, gru_hidden) -> (N_atom*T, gnn_out)
        x, pos, _ = self.conv3(x, edge_index, pos, edge_weight)
        x = self.norm3(x, graphs.batch)
        x = F.relu(x)

        # (N_atom*T, gnn_out) -> (T, N_atom, gnn_out) -> (N_atom, T, gnn_out)
        x = x.reshape((len(graphs), -1, self.gnn_out))
        x = torch.swapaxes(x, 0, 1)

        # Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # (N_atom, T, gru_hidden) -> (T, N_atom, gru_hidden) -> (N_atom*T, gru_hidden)
        x = torch.swapaxes(x, 0, 1)
        x = x.reshape((-1, self.gru_hidden))

        # Run EGNN4 on each frame (N_atom*T, gru_hidden) -> (N_atom*T, gnn_out)
        x, pos, _ = self.conv4(x, edge_index, pos, edge_weight)
        x = self.norm4(x, graphs.batch)
        x = F.relu(x)

        # (N_atom*T, gnn_out) -> (T, N_atom, gnn_out) -> (N_atom, T, gnn_out)
        x = x.reshape((len(graphs), -1, self.gnn_out))
        x = torch.swapaxes(x, 0, 1)

        # Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # (N_atom, T, gru_hidden) -> (T, N_atom, gru_hidden) -> (N_atom*T, gru_hidden)
        x = torch.swapaxes(x, 0, 1)
        x = x.reshape((-1, self.gru_hidden))

        # Run EGNN5 on each frame (N_atom*T, gru_hidden) -> (N_atom*T, gnn_out)
        x, pos, _ = self.conv5(x, edge_index, pos, edge_weight)
        x = self.norm5(x, graphs.batch)
        x = F.relu(x)

        # (N_atom*T, gnn_out) -> (T, N_atom, gnn_out) -> (N_atom, T, gnn_out)
        x = x.reshape((len(graphs), -1, self.gnn_out))
        x = torch.swapaxes(x, 0, 1)

        # Temporal GRU per atom (N_atom, T, gnn_out) -> (N_atom, T, gru_hidden)
        x, _ = self.gru(x)

        # Final hidden state for each atom (N_atom, T, gru_hidden) -> (N_atom, gru_hidden)
        x = x[:, -1, :]

        # 4) Per-atom prediction (N_atom, gru_hidden) -> (N_atom, 1)
        return self.readout(x), x