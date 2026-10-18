"""
Graph Module for constructing AU-AU and AU-Expression graphs.
Uses custom DenseDynamicGraphConv with sample-adaptive causal routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, global_adj_logits, return_routing=False):
        """
        Args:
            x: (B, N, D) node features
            global_adj_logits: (N, N) learned global adjacency logits
            return_routing: also return the signed receiver-first weights
                actually used for message passing, without changing outputs
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
        
        # 2. Global signed graph.  The convention is source -> target:
        # global_adj_logits[source, target].  Matmul rows are receivers, so
        # transpose before message passing.
        global_adj = torch.tanh(global_adj_logits).transpose(0, 1)
        
        # 3. Combine signed global edges with positive dynamic affinities.
        combined_adj = global_adj.unsqueeze(0) * dynamic_adj  # (B, N, N)
        
        # Normalize by absolute mass so inhibitory edges retain their sign.
        combined_adj = combined_adj / (combined_adj.abs().sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Message Passing with Residual Connection
        out = torch.matmul(combined_adj, V)  # (B, N, D')
        out = self.residual_proj(x) + F.elu(self.out_proj(out))
        if return_routing:
            # Receiver-first weights actually used by this factual pass.
            return out, combined_adj
        return out


class AUGraphModule(nn.Module):
    """
    Constructs two graphs via DenseDynamicGraphConv:
    1. AU-AU graph: models relationships between AU embeddings
    2. AU-Expression graph: a strict one-way AU -> Expression graph
    
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
        nn.init.normal_(self.au_au_proj.weight, std=0.01)
        nn.init.zeros_(self.au_au_proj.bias)
        
        # ============================================================
        # AU-Expression Graph
        # ============================================================
        # Only AU -> Expression edges are learnable.  The complete matrix
        # used by DenseDynamicGraphConv is constructed as:
        #
        #   [ 0          AU->Expression ]
        #   [ 0                    0    ]
        #
        # Since DenseDynamicGraphConv transposes source->target adjacency
        # before message passing, only expression nodes receive AU messages.
        # Expressions can never send messages back to AUs.
        self.au_to_expr_adj = nn.Parameter(
            torch.randn(num_aus, num_emotions) * 0.01
        )
        
        self.au_exp_layers = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(gat_num_layers):
            self.au_exp_layers.append(DenseDynamicGraphConv(in_dim, gat_hidden_dim))
            in_dim = gat_hidden_dim
            
        self.au_exp_proj = nn.Linear(in_dim, embed_dim)
        nn.init.normal_(self.au_exp_proj.weight, std=0.01)
        nn.init.zeros_(self.au_exp_proj.bias)

    def _build_au_to_expr_adjacency(self):
        """Build the fixed-topology source->target adjacency matrix."""
        zero_au_au = self.au_to_expr_adj.new_zeros(self.num_aus, self.num_aus)
        zero_expr_au = self.au_to_expr_adj.new_zeros(self.num_emotions, self.num_aus)
        zero_expr_expr = self.au_to_expr_adj.new_zeros(
            self.num_emotions, self.num_emotions
        )
        top = torch.cat([zero_au_au, self.au_to_expr_adj], dim=1)
        bottom = torch.cat([zero_expr_au, zero_expr_expr], dim=1)
        return torch.cat([top, bottom], dim=0)
    
    def forward_au_au(self, au_embeddings_stacked, return_routing=False):
        """
        Forward pass for AU-AU graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) stacked AU embeddings
            return_routing: additionally return (B, target, source) absolute
                factual routing, averaged across layers, for CF masks
        Returns:
            x: (B, N_AU, D) updated AU embeddings after graph reasoning
            au_au_adj_logits: (N_AU, N_AU) signed source -> target logits
        """
        x = au_embeddings_stacked
        routing = []
        for layer in self.au_au_layers:
            if return_routing:
                x, weights = layer(x, self.au_au_adj, return_routing=True)
                routing.append(weights.abs())
            else:
                x = layer(x, self.au_au_adj)
            
        x = au_embeddings_stacked + self.au_au_proj(x)
        if return_routing:
            return x, self.au_au_adj, torch.stack(routing).mean(dim=0)
        return x, self.au_au_adj
    
    def forward_au_exp(self, au_embeddings_stacked, emotion_embeddings_stacked, return_routing=False):
        """
        Forward pass for AU-Expression graph.
        
        Args:
            au_embeddings_stacked: (B, N_AU, D) AU embeddings
            emotion_embeddings_stacked: (B, N_EMO, D) emotion embeddings
            return_routing: additionally return absolute factual routing in
                (B, receiver, sender) order, averaged across layers
        Returns:
            x: (B, N_AU + N_EMO, D) updated node embeddings
            au_exp_adj_logits: (N_AU + N_EMO, N_AU + N_EMO) signed source -> target logits
                with only the AU -> Expression block learnable/nonzero
        """
        node_features = torch.cat([au_embeddings_stacked, emotion_embeddings_stacked], dim=1)
        au_exp_adj = self._build_au_to_expr_adjacency()
        x = node_features
        routing = []
        for layer in self.au_exp_layers:
            if return_routing:
                x, weights = layer(x, au_exp_adj, return_routing=True)
                routing.append(weights.abs())
            else:
                x = layer(x, au_exp_adj)
            
        x = node_features + self.au_exp_proj(x)
        if return_routing:
            return x, au_exp_adj, torch.stack(routing).mean(dim=0)
        return x, au_exp_adj
