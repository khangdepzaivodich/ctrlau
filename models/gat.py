"""
Graph Attention Network module for constructing AU-AU and AU-Expression graphs.
Uses torch_geometric's GATConv layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class AUGraphModule(nn.Module):
    """
    Constructs two graphs via GAT:
    1. AU-AU graph: models relationships between AU embeddings
    2. AU-Expression graph: models relationships between AU and Expression embeddings
    
    Each graph produces a learned adjacency matrix (edge weights).
    """
    
    def __init__(self, embed_dim, num_aus, num_emotions, 
                 gat_hidden_dim=256, gat_num_heads=4, gat_num_layers=2, dropout=0.1):
        super().__init__()
        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.embed_dim = embed_dim
        
        # ============================================================
        # AU-AU Graph
        # ============================================================
        # Learnable adjacency matrix for AU-AU (soft, continuous)
        self.au_au_adj = nn.Parameter(torch.randn(num_aus, num_aus) * 0.01)
        
        # GAT layers for AU-AU message passing
        self.au_au_gat_layers = nn.ModuleList()
        in_dim = embed_dim
        for layer_idx in range(gat_num_layers):
            out_dim = gat_hidden_dim // gat_num_heads
            self.au_au_gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=gat_num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            in_dim = out_dim * gat_num_heads
        
        # Project back to embed_dim
        self.au_au_proj = nn.Linear(in_dim, embed_dim)
        
        # ============================================================
        # AU-Expression Graph
        # ============================================================
        num_nodes_ae = num_aus + num_emotions
        
        # Learnable adjacency matrix for AU-Exp
        self.au_exp_adj = nn.Parameter(torch.randn(num_nodes_ae, num_nodes_ae) * 0.01)
        
        # GAT layers for AU-Exp message passing
        self.au_exp_gat_layers = nn.ModuleList()
        in_dim = embed_dim
        for layer_idx in range(gat_num_layers):
            out_dim = gat_hidden_dim // gat_num_heads
            self.au_exp_gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=gat_num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            in_dim = out_dim * gat_num_heads
        
        self.au_exp_proj = nn.Linear(in_dim, embed_dim)
    
    def _build_edge_index_from_adj(self, adj, threshold=0.0):
        """
        Convert a soft adjacency matrix to edge_index and edge_weight
        for torch_geometric.
        
        Args:
            adj: (N, N) learnable adjacency parameters
            threshold: only include edges above this threshold (after sigmoid)
        Returns:
            edge_index: (2, E) edge indices
            edge_weight: (E,) edge weights
        """
        # Apply sigmoid to get [0, 1] weights
        weights = torch.sigmoid(adj)
        
        # Build fully connected edge_index (we let GAT attention handle sparsity)
        N = adj.size(0)
        src = torch.arange(N, device=adj.device).unsqueeze(1).expand(N, N).reshape(-1)
        dst = torch.arange(N, device=adj.device).unsqueeze(0).expand(N, N).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)  # (2, N*N)
        edge_weight = weights.reshape(-1)            # (N*N,)
        
        return edge_index, edge_weight
    
    def forward_au_au(self, au_embeddings_stacked):
        """
        Forward pass for AU-AU graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) stacked AU embeddings
        Returns:
            updated_au: (B, N_AU, D) updated AU embeddings after graph reasoning
            au_au_adj_sigmoid: (N_AU, N_AU) learned adjacency weights
        """
        B, N, D = au_embeddings_stacked.shape
        device = au_embeddings_stacked.device
        
        edge_index, edge_weight = self._build_edge_index_from_adj(self.au_au_adj)
        au_au_adj_sigmoid = torch.sigmoid(self.au_au_adj)
        
        # Process each sample in the batch
        outputs = []
        for b in range(B):
            x = au_embeddings_stacked[b]  # (N_AU, D)
            for gat_layer in self.au_au_gat_layers:
                x = gat_layer(x, edge_index)
                x = F.elu(x)
            x = self.au_au_proj(x)  # (N_AU, D)
            outputs.append(x)
        
        updated_au = torch.stack(outputs, dim=0)  # (B, N_AU, D)
        
        return updated_au, au_au_adj_sigmoid
    
    def forward_au_exp(self, au_embeddings_stacked, emotion_embed):
        """
        Forward pass for AU-Expression graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) AU embeddings
            emotion_embed: (B, D_emo) emotion embedding
        Returns:
            updated_nodes: (B, N_AU + N_EMO, D) updated node embeddings
            au_exp_adj_sigmoid: (N_AU + N_EMO, N_AU + N_EMO) adjacency weights
        """
        B, N_AU, D = au_embeddings_stacked.shape
        device = au_embeddings_stacked.device
        
        # Expand emotion embedding to match AU embedding dim if needed
        # emotion_embed: (B, D_emo), we need (B, N_EMO, D)
        # Replicate emotion embedding for each emotion node
        emo_expanded = emotion_embed.unsqueeze(1).expand(B, self.num_emotions, D)
        
        # Concatenate AU and Emotion node features
        node_features = torch.cat([au_embeddings_stacked, emo_expanded], dim=1)  # (B, N_AU+N_EMO, D)
        
        edge_index, edge_weight = self._build_edge_index_from_adj(self.au_exp_adj)
        au_exp_adj_sigmoid = torch.sigmoid(self.au_exp_adj)
        
        outputs = []
        for b in range(B):
            x = node_features[b]  # (N_AU+N_EMO, D)
            for gat_layer in self.au_exp_gat_layers:
                x = gat_layer(x, edge_index)
                x = F.elu(x)
            x = self.au_exp_proj(x)
            outputs.append(x)
        
        updated_nodes = torch.stack(outputs, dim=0)
        
        return updated_nodes, au_exp_adj_sigmoid
