"""
Mask module: learnable importance masks and polarity masks
extracted from graph adjacency matrices.

Magnitude -> Threshold -> Binary Mask
"""
import torch
import torch.nn as nn


class MaskModule(nn.Module):
    """
    Generates importance masks and polarity masks from graph adjacency matrices.
    
    - Importance mask: which edges/regions are important.
      Computed by: sigmoid(adjacency) -> compare against learnable threshold -> hard mask
      (Uses straight-through estimator for differentiability)
    
    - Polarity mask: +1 (excitatory) or -1 (inhibitory) per edge.
      Computed from the sign of the raw adjacency weights.
    """
    
    def __init__(self, num_aus, num_emotions):
        super().__init__()
        self.num_aus = num_aus
        self.num_emotions = num_emotions
        
        # Learnable threshold for AU-AU graph (pre-sigmoid, initialized to 0.5)
        self.au_au_threshold = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        # Learnable threshold for AU-Exp graph
        num_nodes_ae = num_aus + num_emotions
        self.au_exp_threshold = nn.Parameter(torch.tensor(0.0))
    
    def _threshold_mask(self, magnitude, threshold_param):
        """
        Apply learnable threshold with straight-through estimator.
        
        Args:
            magnitude: (N, N) edge magnitudes in [0, 1]
            threshold_param: scalar learnable threshold parameter
        Returns:
            hard_mask: (N, N) binary mask {0, 1} (differentiable via STE)
        """
        threshold = torch.sigmoid(threshold_param)
        
        # Soft mask (differentiable)
        # Use a steep sigmoid to approximate step function
        temperature = 10.0
        soft_mask = torch.sigmoid(temperature * (magnitude - threshold))
        
        # Hard mask with straight-through estimator
        hard_mask = (magnitude > threshold).float()
        mask = hard_mask + soft_mask - soft_mask.detach()  # STE trick
        
        return mask
    
    def forward_au_au(self, au_au_adj):
        """
        Generate masks for AU-AU graph.
        
        Args:
            au_au_adj: (N_AU, N_AU) sigmoid adjacency weights
        Returns:
            importance_mask: (N_AU, N_AU) binary importance mask
            polarity_mask: (N_AU, N_AU) +1/-1 polarity mask
        """
        importance_mask = self._threshold_mask(au_au_adj, self.au_au_threshold)
        
        # Polarity: sign of the raw adjacency (before sigmoid)
        # Since we receive sigmoid(adj), we use: > 0.5 means positive, < 0.5 means negative
        polarity_mask = torch.where(au_au_adj > 0.5,
                                     torch.ones_like(au_au_adj),
                                     -torch.ones_like(au_au_adj))
        
        return importance_mask, polarity_mask
    
    def forward_au_exp(self, au_exp_adj):
        """
        Generate masks for AU-Expression graph.
        
        Args:
            au_exp_adj: (N_nodes, N_nodes) sigmoid adjacency weights
        Returns:
            importance_mask: (N_nodes, N_nodes) binary importance mask
            polarity_mask: (N_nodes, N_nodes) +1/-1 polarity mask
        """
        importance_mask = self._threshold_mask(au_exp_adj, self.au_exp_threshold)
        
        polarity_mask = torch.where(au_exp_adj > 0.5,
                                     torch.ones_like(au_exp_adj),
                                     -torch.ones_like(au_exp_adj))
        
        return importance_mask, polarity_mask
