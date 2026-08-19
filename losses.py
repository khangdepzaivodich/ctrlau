"""
All loss functions for the CtrlAU architecture.
- HSIC-based disentanglement (L_ib, L_align, L_decorr)
- Contrastive loss (InfoNCE)
- DAG constraint loss
- Rule violation loss
- Counterfactual intervention losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# AU Classification: Asymmetric / Weighted Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-label classification with optional positive class weighting.
    Down-weights easy examples and focuses the gradients on hard examples.
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        # We use PyTorch's built-in pos_weight to handle the raw data imbalance,
        # and Focal Loss to dynamically handle prediction confidence.
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, inputs, targets):
        # Compute standard BCE loss (per element, unreduced)
        bce_loss = self.bce_with_logits(inputs, targets)
        
        # Calculate pt (probability of true class) using the trick: pt = exp(-BCE)
        pt = torch.exp(-bce_loss)
        
        # Focal Loss formula: (1 - pt)^gamma * BCE
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        return focal_loss.mean()


# ============================================================
# HSIC (Hilbert-Schmidt Independence Criterion)
# ============================================================

def rbf_kernel(X, sigma=None):
    """
    Compute RBF (Gaussian) kernel matrix.
    Args:
        X: (B, D) tensor
        sigma: bandwidth. If None, use median heuristic.
    Returns:
        K: (B, B) kernel matrix
    """
    # Pairwise squared distances
    XXT = X @ X.t()
    diag = XXT.diag().unsqueeze(1)
    dists = diag + diag.t() - 2.0 * XXT  # (B, B)
    dists = dists.clamp(min=0.0)
    
    if sigma is None:
        # Median heuristic
        median_dist = dists.median()
        sigma = (median_dist / (2.0 * torch.log(torch.tensor(X.size(0) + 1.0, device=X.device)))).clamp(min=1e-5)
    
    K = torch.exp(-dists / (2.0 * sigma))
    return K


def hsic(X, Y, sigma_x=None, sigma_y=None):
    """
    Compute HSIC between two sets of representations.
    Args:
        X: (B, D1)
        Y: (B, D2)
    Returns:
        scalar HSIC value
    """
    B = X.size(0)
    if B < 4:
        return torch.tensor(0.0, device=X.device)
    
    K = rbf_kernel(X, sigma_x)
    L = rbf_kernel(Y, sigma_y)
    
    # Center the kernel matrices
    H = torch.eye(B, device=X.device) - (1.0 / B) * torch.ones(B, B, device=X.device)
    
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    # HSIC = (1 / (B-1)^2) * tr(Kc @ Lc)
    hsic_val = (Kc * Lc).sum() / ((B - 1) ** 2)
    return hsic_val


class HSICDisentanglementLoss(nn.Module):
    """
    Information Bottleneck via HSIC-Based Disentanglement.
    
    Three objectives:
    (i)   L_ib = (1/N_AU) * sum_i HSIC(f_AU^i, z_img)        -> minimize
    (ii)  L_align = -(1/N_AU) * sum_i HSIC(f_AU^i, y_AU^i)   -> maximize (negate to minimize)
    (iii) L_decorr = (1/N_AU^2) * sum_{i!=j} HSIC(f_AU^i, f_AU^j) -> minimize
    """
    
    def __init__(self, num_aus):
        super().__init__()
        self.num_aus = num_aus
    
    def forward(self, au_embeddings, z_img, au_labels):
        """
        Args:
            au_embeddings: list of (B, D) tensors, one per AU, length N_AU
            z_img: (B, D_img) global image feature
            au_labels: (B, N_AU) binary AU labels
        Returns:
            L_ib, L_align, L_decorr
        """
        B = z_img.size(0)
        device = z_img.device
        N = self.num_aus
        
        # L_ib: minimize dependence between each AU repr and z_img
        l_ib = torch.tensor(0.0, device=device)
        for i in range(N):
            l_ib = l_ib + hsic(au_embeddings[i], z_img)
        l_ib = l_ib / N
        
        # L_align: maximize dependence between each AU repr and its label
        # We treat the label as a 1D signal, expand to (B, 1) for kernel computation
        l_align = torch.tensor(0.0, device=device)
        for i in range(N):
            label_i = au_labels[:, i].unsqueeze(1)  # (B, 1)
            l_align = l_align + hsic(au_embeddings[i], label_i)
        l_align = -l_align / N  # negate because we want to maximize
        
        # L_decorr: minimize dependence between different AU reprs
        l_decorr = torch.tensor(0.0, device=device)
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                l_decorr = l_decorr + hsic(au_embeddings[i], au_embeddings[j])
                count += 1
        if count > 0:
            l_decorr = l_decorr / count
        
        return l_ib, l_align, l_decorr


# ============================================================
# Contrastive Loss (InfoNCE)
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss to align visual AU embeddings
    with textual AU description embeddings in a shared space.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, visual_embeds, text_embeds):
        """
        Args:
            visual_embeds: (N_AU, D) AU visual embeddings (averaged over batch)
            text_embeds: (N_AU, D) AU text description embeddings
        Returns:
            contrastive loss scalar
        """
        # L2 normalize
        visual_embeds = F.normalize(visual_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Similarity matrix (N_AU x N_AU)
        logits = visual_embeds @ text_embeds.t() / self.temperature
        
        # Labels: each AU should match itself
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Symmetric InfoNCE
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2.0


# ============================================================
# DAG Constraint Loss
# ============================================================

class DAGLoss(nn.Module):
    """
    Enforces the learned adjacency matrix to be a DAG.
    L_DAG = tr(exp(E_g ⊙ E_g)) - N_AU
    
    where E_g is the weighted adjacency matrix of the graph.
    """
    
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
    
    def forward(self, adjacency):
        """
        Args:
            adjacency: (N, N) weighted adjacency matrix
        Returns:
            DAG constraint loss scalar
        """
        # E ⊙ E (Hadamard product = element-wise square)
        E_sq = adjacency * adjacency
        
        # Matrix exponential via eigendecomposition for stability
        # exp(E_sq) and then trace
        # For small matrices, direct computation is fine
        exp_E = torch.matrix_exp(E_sq)
        dag_loss = torch.trace(exp_E) - self.num_nodes
        
        return dag_loss


# ============================================================
# Rule Violation Loss
# ============================================================

class ViolationLoss(nn.Module):
    """
    Computes violation of learned causal edges against AU predictions.
    For edge A -> B with positive polarity:
        violation = p(A) * (1 - p(B))
    For edge A -> B with negative polarity:
        violation = p(A) * p(B)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, au_probs, adjacency, polarity_mask, importance_mask):
        """
        Args:
            au_probs: (B, N_AU) predicted AU probabilities
            adjacency: (N_AU, N_AU) learned adjacency matrix
            polarity_mask: (N_AU, N_AU) polarity: +1 for excitatory, -1 for inhibitory
            importance_mask: (N_AU, N_AU) binary mask from thresholding
        Returns:
            violation loss scalar
        """
        B, N = au_probs.shape
        
        # Compute violation for each edge
        # p_i: (B, N, 1), p_j: (B, 1, N)
        p_i = au_probs.unsqueeze(2)  # (B, N, 1)
        p_j = au_probs.unsqueeze(1)  # (B, 1, N)
        
        # Positive polarity: A->B means A activates B => violation = p(A) * (1 - p(B))
        # Negative polarity: A->B means A inhibits B => violation = p(A) * p(B)
        positive_violation = p_i * (1.0 - p_j)  # (B, N, N)
        negative_violation = p_i * p_j           # (B, N, N)
        
        # Select based on polarity
        is_positive = (polarity_mask > 0).float().unsqueeze(0)  # (1, N, N)
        is_negative = (polarity_mask < 0).float().unsqueeze(0)
        
        violation = positive_violation * is_positive + negative_violation * is_negative
        
        # Weight by adjacency magnitude and importance mask
        edge_weight = adjacency.abs().unsqueeze(0) * importance_mask.unsqueeze(0)  # (1, N, N)
        weighted_violation = violation * edge_weight
        
        # Average over batch and edges
        loss = weighted_violation.sum() / (B * edge_weight.sum().clamp(min=1.0))
        
        return loss


# ============================================================
# Counterfactual Intervention Losses
# ============================================================

class CounterfactualLoss(nn.Module):
    """
    Two counterfactual losses based on perturbation:
    
    1. Perturb important regions (mask=1):
       Expected behavior: predictions degrade.
       Loss = -||pred_original - pred_perturbed||  (we want degradation, so maximize difference)
       Equivalently minimize: similarity between original and perturbed.
    
    2. Perturb unimportant regions (mask=0):
       Expected behavior: predictions unchanged.
       Loss = ||pred_original - pred_perturbed||  (we want invariance, so minimize difference)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_original, pred_important_perturbed, pred_unimportant_perturbed):
        """
        Args:
            pred_original: (B, N) original predictions
            pred_important_perturbed: (B, N) predictions after perturbing important regions
            pred_unimportant_perturbed: (B, N) predictions after perturbing unimportant regions
        Returns:
            loss_important, loss_unimportant
        """
        # Important perturbation: should cause degradation
        # We want pred_important_perturbed to be DIFFERENT from pred_original
        # Loss = -MSE => minimize negative MSE => maximize MSE
        diff_important = F.mse_loss(pred_important_perturbed, pred_original)
        loss_important = -diff_important  # negative because we WANT large difference
        
        # Unimportant perturbation: should NOT cause degradation
        # We want pred_unimportant_perturbed to be SAME as pred_original
        loss_unimportant = F.mse_loss(pred_unimportant_perturbed, pred_original)
        
        return loss_important, loss_unimportant
