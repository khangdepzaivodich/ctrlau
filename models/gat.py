"""
Graph Attention Network module for constructing AU-AU and AU-Expression graphs.
Uses torch_geometric's GATConv layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DenseDynamicGraphConv(nn.Module):
    """
    Custom Dense Graph Convolution that implements Sample-Adaptive Causal Routing (Idea 2).
    Combines a dynamic sample-adaptive affinity matrix (via Q-K self-attention) with a 
    globally learned, invariant adjacency graph.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
    def forward(self, x, global_adj_logits):
        """
        Args:
            x: (B, N, D) node features
            global_adj_logits: (N, N) learned global adjacency logits
        Returns:
            out: (B, N, D) updated node features
        """
        Q = self.W_q(x)  # (B, N, D')
        K = self.W_k(x)  # (B, N, D')
        V = self.W_v(x)  # (B, N, D')
        
        # 1. Sample-Adaptive Dynamic Affinity (Scaled Dot-Product)
        d_k = Q.size(-1)
        dynamic_logits = torch.matmul(Q, K.transpose(1, 2)) / (d_k ** 0.5)  # (B, N, N)
        dynamic_adj = torch.sigmoid(dynamic_logits)  # Sample-specific edge probabilities
        
        # 2. Global Invariant Graph
        global_adj = torch.sigmoid(global_adj_logits)  # (N, N)
        
        # 3. Combine Static and Dynamic Adjacencies
        # We element-wise multiply the global DAG probabilities with the dynamic affinities
        combined_adj = global_adj.unsqueeze(0) * dynamic_adj  # (B, N, N)
        
        # Row-normalize (like standard attention/GCN)
        combined_adj = combined_adj / (combined_adj.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Message Passing
        out = torch.matmul(combined_adj, V)  # (B, N, D')
        out = F.elu(self.out_proj(out))
        return out


class AUGraphModule(nn.Module):
    """
    Constructs two graphs via DenseDynamicGraphConv:
    1. AU-AU graph: models relationships between AU embeddings
    2. AU-Expression graph: models relationships between AU and Expression embeddings
    
    Each graph produces a learned global adjacency matrix (edge weights) combined with 
    sample-adaptive routing.
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
        # Learnable global adjacency matrix for AU-AU
        self.au_au_adj = nn.Parameter(torch.randn(num_aus, num_aus) * 0.01)
        
        self.au_au_layers = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(gat_num_layers):
            self.au_au_layers.append(DenseDynamicGraphConv(in_dim, gat_hidden_dim))
            in_dim = gat_hidden_dim
            
        self.au_au_proj = nn.Linear(in_dim, embed_dim)
        
        # ============================================================
        # AU-Expression Graph
        # ============================================================
        num_nodes_ae = num_aus + num_emotions
        
        # Learnable global adjacency matrix for AU-Exp
        self.au_exp_adj = nn.Parameter(torch.randn(num_nodes_ae, num_nodes_ae) * 0.01)
        
        self.au_exp_layers = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(gat_num_layers):
            self.au_exp_layers.append(DenseDynamicGraphConv(in_dim, gat_hidden_dim))
            in_dim = gat_hidden_dim
            
        self.au_exp_proj = nn.Linear(in_dim, embed_dim)
    
    def forward_au_au(self, au_embeddings_stacked):
        """
        Forward pass for AU-AU graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) stacked AU embeddings
        Returns:
            x: (B, N_AU, D) updated AU embeddings after graph reasoning
            au_au_adj_sigmoid: (N_AU, N_AU) global learned adjacency weights
        """
        x = au_embeddings_stacked
        for layer in self.au_au_layers:
            x = layer(x, self.au_au_adj)
            
        x = self.au_au_proj(x)
        au_au_adj_sigmoid = torch.sigmoid(self.au_au_adj)
        
        return x, au_au_adj_sigmoid
    
    def forward_au_exp(self, au_embeddings_stacked, emotion_embeddings_stacked):
        """
        Forward pass for AU-Expression graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) AU embeddings
            emotion_embeddings_stacked: (B, N_EMO, D) emotion embeddings
        Returns:
            x: (B, N_AU + N_EMO, D) updated node embeddings
            au_exp_adj_sigmoid: (N_AU + N_EMO, N_AU + N_EMO) global adjacency weights
        """
        node_features = torch.cat([au_embeddings_stacked, emotion_embeddings_stacked], dim=1)
        x = node_features
        for layer in self.au_exp_layers:
            x = layer(x, self.au_exp_adj)
            
        x = self.au_exp_proj(x)
        au_exp_adj_sigmoid = torch.sigmoid(self.au_exp_adj)
        
        return x, au_exp_adj_sigmoid
