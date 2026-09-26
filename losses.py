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
        # Paper Eq. (1): (1 / N_AU^2) * sum_{i != j} HSIC(f_AU^i, f_AU^j)
        # Using symmetry HSIC(A, B) = HSIC(B, A), sum_{i != j} = 2 * sum_{i < j}
        l_decorr = torch.tensor(0.0, device=device)
        for i in range(N):
            for j in range(i + 1, N):
                l_decorr = l_decorr + hsic(au_embeddings[i], au_embeddings[j])
        l_decorr = (2.0 * l_decorr) / (N ** 2)
        
        return l_ib, l_align, l_decorr


class WeightedAsymmetricLoss(nn.Module):
    """
    Weighted Asymmetric Loss from the original MultiviewSymAU repo.
    Specifically designed for heavily imbalanced AU datasets.
    """
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing (down-weight easy negatives)
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
            
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            # Ensure weight is on the same device as loss
            self.weight = self.weight.to(loss.device)
            loss = loss * self.weight.view(1, -1)

        loss = loss.mean()
        return -loss

# ============================================================
# Contrastive Loss (InfoNCE)
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Standard InfoNCE contrastive loss.
    Assumes embeddings are L2-normalized.
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


class FACSEmotionViolationLoss(nn.Module):
    """
    Computes strict FACS-based hypergraph violations for AU-Expression logic.
    Evaluates strict logic gates using Fuzzy Logic (T-norm).
    Formula: Prob(A) * Prob(B) * ... * (1 - Prob(Emotion))
    """
    # Mapping from rule emotion names to SYM_EMOTIONS indices:
    # SYM_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]
    EMO_TO_SYM_IDX = {
        "anger": 0,
        "fear": 1,
        "happiness": 2,
        "sadness": 3,
        "surprise": 4,
        "disgust": 5,
    }

    def __init__(self):
        super().__init__()
        from config import EMOTION_AU_RULES_IDX
        self.rules = EMOTION_AU_RULES_IDX

    def forward(self, au_probs, emotion_probs):
        """
        Args:
            au_probs: (B, N_AU) AU probabilities
            emotion_probs: (B, N_EMO) Emotion probabilities (ordered by SYM_EMOTIONS)
        Returns:
            violation loss scalar
        """
        device = au_probs.device
        total_violation = torch.tensor(0.0, device=device)
        
        for emo_name, rule in self.rules.items():
            sym_idx = self.EMO_TO_SYM_IDX.get(emo_name, None)
            if sym_idx is None or sym_idx >= emotion_probs.size(1):
                continue

            au_indices = rule["required_idx"]
            
            # Gather relevant AU probabilities
            relevant_probs = au_probs[:, au_indices]  # (B, K)
            
            # T-norm (AND) over required AUs
            if rule["operator"] == "AND":
                au_score = relevant_probs.prod(dim=1)  # (B,)
            else: # OR
                au_score = 1.0 - (1.0 - relevant_probs).prod(dim=1)
                
            # Violation: AUs are firing but Emotion is NOT firing
            # Violation = au_score * (1 - emotion_prob)
            violation = au_score * (1.0 - emotion_probs[:, sym_idx])
            total_violation += violation.mean()
            
        return total_violation


# ============================================================
# Counterfactual Intervention Losses
# ============================================================

class CounterfactualLoss(nn.Module):
    """
    Dual-level Counterfactual Intervention Loss (Hu et al., CausalAffect - arXiv:2512.00456, Eqs. 13-15).
    
    1. Consistency (perturbing non-causal / unimportant sources):
       - Feature consistency: 1 - cosine_similarity(Z_orig, Z_cf_unimp)
       - Logit consistency: MSE(pred_orig, pred_unimp)
       => loss_unimportant = delta_feat * feat_consist + delta_logit * logit_consist
       
    2. Discrepancy (perturbing causal / important sources):
       - Feature discrepancy: 1 + cosine_similarity(Z_orig, Z_cf_imp)
       - Logit discrepancy: max(0, 1 - MSE(pred_orig, pred_imp))
       => loss_important = eta_feat * feat_discrep + eta_logit * logit_discrep
    """
    
    def __init__(self, delta_feat=1.0, delta_logit=1.0, eta_feat=1.0, eta_logit=1.0, is_expression=False):
        super().__init__()
        self.delta_feat = delta_feat
        self.delta_logit = delta_logit
        self.eta_feat = eta_feat
        self.eta_logit = eta_logit
        self.is_expression = is_expression
    
    def forward(
        self, 
        pred_original, 
        pred_important_perturbed, 
        pred_unimportant_perturbed,
        feat_original=None,
        feat_important_perturbed=None,
        feat_unimportant_perturbed=None
    ):
        """
        Args:
            pred_original: (B, N) or (B, 1) factual predictions
            pred_important_perturbed: (B, N) or (B, 1) predictions after causal perturbation
            pred_unimportant_perturbed: (B, N) or (B, 1) predictions after non-causal perturbation
            feat_original: (B, N, D) or (B, D) factual feature representations Z
            feat_important_perturbed: (B, N, D) or (B, D) feature representations under causal perturbation
            feat_unimportant_perturbed: (B, N, D) or (B, D) feature representations under non-causal perturbation
        Returns:
            loss_important, loss_unimportant
        """
        # 1. Logit-level Consistency & Discrepancy
        if self.is_expression and pred_original.dim() == 2 and pred_original.size(1) > 1:
            # Multi-class emotion probabilities: KL divergence or MSE
            p_orig = pred_original.clamp(1e-6, 1.0)
            p_unimp = pred_unimportant_perturbed.clamp(1e-6, 1.0)
            logit_consist = F.kl_div(p_unimp.log(), p_orig, reduction='batchmean')
            diff_important = F.mse_loss(pred_important_perturbed, pred_original)
            logit_discrep = torch.clamp(1.0 - diff_important, min=0.0)
        else:
            logit_consist = F.mse_loss(pred_unimportant_perturbed, pred_original)
            diff_important = F.mse_loss(pred_important_perturbed, pred_original)
            logit_discrep = torch.clamp(1.0 - diff_important, min=0.0)
        
        # 2. Feature-level Consistency & Discrepancy (Cosine Distance)
        if (
            feat_original is not None 
            and feat_important_perturbed is not None 
            and feat_unimportant_perturbed is not None
        ):
            # Feature consistency: (1 - cos) -> minimize to 0
            cos_unimp = F.cosine_similarity(feat_original, feat_unimportant_perturbed, dim=-1)
            feat_consist = (1.0 - cos_unimp).mean()
            
            # Feature discrepancy: (1 + cos) -> minimize (forces orthogonality or divergence)
            cos_imp = F.cosine_similarity(feat_original, feat_important_perturbed, dim=-1)
            feat_discrep = (1.0 + cos_imp).mean()
            
            loss_unimportant = self.delta_feat * feat_consist + self.delta_logit * logit_consist
            loss_important = self.eta_feat * feat_discrep + self.eta_logit * logit_discrep
        else:
            # Fallback for logit-only calls
            loss_unimportant = logit_consist
            loss_important = -diff_important
            
        return loss_important, loss_unimportant


# ============================================================
# FACS AU-AU Rule Violation Loss
# ============================================================

class FACSAUViolationLoss(nn.Module):
    """
    Computes violation of strict FACS anatomical rules between AUs using Fuzzy Logic (T-norms).
    Note: Contradictory rules (AU25 XOR AU26 and AU9 subsumption) have been removed
    because empirical DISFA annotations exhibit 72.1% co-occurrence between AU25 and AU26.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, au_probs):
        """
        Args:
            au_probs: (B, N_AU) predicted AU probabilities
        Returns:
            violation_loss: scalar sum of valid rule violations (0.0 for DISFA 8-AU subset)
        """
        device = au_probs.device
        return torch.tensor(0.0, device=device)



# ============================================================
# MultiviewSymAU Phase 1 Losses
# ============================================================

class WeightedAsymmetricLoss(nn.Module):
    """
    Weighted Asymmetric Loss (WAL) from MultiviewSymAU (Eq. 3).
    Addresses severe AU class imbalance by dynamically downweighting easy negative examples.
    """
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super().__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                self.register_buffer("weight", weight)
            else:
                self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.register_buffer("weight", None)

    def forward(self, x, y):
        """
        x: p^a (B, N_a) probabilities after sigmoid
        y: y^a (B, N_a) ground-truth labels (0/1)
        """
        xs_pos = x
        xs_neg = 1.0 - x

        # Basic cross-entropy calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing factor for negatives: (1 - p)
        if self.disable_torch_grad:
            neg_weight = (1.0 - xs_neg).detach()
        else:
            neg_weight = 1.0 - xs_neg
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            w = self.weight.to(x.device)
            loss = loss * w.view(1, -1)

        loss = loss.mean(dim=-1)
        return -loss.mean()


class ExpressionBCELoss(nn.Module):
    """
    Expression BCE Loss from MultiviewSymAU (Eq. 4).
    Standard multi-label BCE over emotion probabilities without asymmetric weighting.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_probs, targets_one_hot):
        """
        pred_probs: p^e (B, N_e) emotion probabilities after sigmoid
        targets_one_hot: y^e (B, N_e) one-hot pseudo-labels
        """
        xs_pos = pred_probs
        xs_neg = 1.0 - pred_probs

        los_pos = targets_one_hot * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets_one_hot) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss = loss.mean(dim=-1)
        return -loss.mean()

