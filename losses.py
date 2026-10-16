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


class AUExpressionSupportLoss(nn.Module):
    """Penalize supported AUs that fail to support the matching expression.

    ``support_matrix`` is the verified SymGraphAU relation mask with shape
    ``(N_expression, N_AU)``.  This is a soft implication regularizer for the
    CtrlAU graph; it is deliberately separate from Phase 1 and is not a claim
    that a FACS rule can uniquely identify an expression.
    """
    def __init__(self, support_matrix):
        super().__init__()
        support_matrix = torch.as_tensor(support_matrix, dtype=torch.float32)
        if support_matrix.dim() != 2:
            raise ValueError("support_matrix must have shape (N_expression, N_AU)")
        if support_matrix.sum() <= 0:
            raise ValueError("support_matrix must contain at least one verified AU-expression relation")
        self.register_buffer("support_matrix", support_matrix)

    def forward(self, au_probs, emotion_probs):
        if au_probs.dim() != 2 or emotion_probs.dim() != 2:
            raise ValueError("AU and expression probabilities must be rank-2 tensors")
        if au_probs.size(1) != self.support_matrix.size(1):
            raise ValueError("AU probability dimension does not match support_matrix")
        if emotion_probs.size(1) != self.support_matrix.size(0):
            raise ValueError("expression probability dimension does not match support_matrix")
        violations = (
            au_probs.unsqueeze(1)
            * (1.0 - emotion_probs.unsqueeze(-1))
        )
        return (violations * self.support_matrix.unsqueeze(0)).sum() / (
            au_probs.size(0) * self.support_matrix.sum()
        )


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
        num_targets = pred_original.size(1) if (pred_original.dim() >= 2 and pred_original.size(1) > 1) else 1.0
        # AU and expression heads both use independent sigmoid probabilities;
        # KL divergence would incorrectly assume a normalized categorical
        # distribution for expressions. MSE is the matching multi-label metric.
        logit_consist = F.mse_loss(pred_unimportant_perturbed, pred_original)
        diff_important = F.mse_loss(pred_important_perturbed, pred_original) * num_targets
        logit_discrep = torch.clamp(1.0 - diff_important, min=0.0)
        
        # 2. Feature-level Consistency & Discrepancy (Cosine Distance)
        if (
            feat_original is not None 
            and feat_important_perturbed is not None 
            and feat_unimportant_perturbed is not None
        ):
            # Feature consistency: (1 - cos) -> minimize to 0
            cos_unimp = F.cosine_similarity(feat_original, feat_unimportant_perturbed, dim=-1)
            feat_consist = (1.0 - cos_unimp).clamp_min(0.0).mean()
            
            # Feature discrepancy: (1 + cos) -> minimize (forces orthogonality or divergence)
            cos_imp = F.cosine_similarity(feat_original, feat_important_perturbed, dim=-1)
            feat_discrep = (1.0 + cos_imp).clamp_min(0.0).mean()
            
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
    Computes violation of strict FACS anatomical and semantic rules between AUs using Fuzzy Logic (T-norms).
    Based on FACS 2002 standard rules in facs_au_rules.md:
    
    1. Additive Hierarchy / Implication:
       - Jaw Drop implies Lips Part: AU 26 -> AU 25
         Violation = p(AU 26) * (1 - p(AU 25))
       - Mouth Stretch implies Lips Part & Jaw Drop: AU 27 -> AU 25, AU 27 -> AU 26
         Violation = p(AU 27) * (1 - p(AU 25)) + p(AU 27) * (1 - p(AU 26))
         
    2. Mutually Exclusive Rules (XOR):
       - AU 24 XOR AU 25 (Lip Pressor vs Lips Part)
         Violation = p(AU 24) * p(AU 25)
       - AU 45 XOR AU 46 (Blink vs Wink)
       - AU 41 XOR AU 42 XOR AU 43 (Lid Droop / Slit / Closed)
       - Opposing head rotations (AU 51/52, 53/54, 55/56)
       - Opposing conjugate gaze (AU 61/62, 63/64)
    """
    def __init__(self, au_ids):
        super().__init__()
        self.au_index = {int(au): i for i, au in enumerate(au_ids)}
        
    def forward(self, au_probs):
        """
        Args:
            au_probs: (B, N_AU) predicted AU probabilities
        Returns:
            violation_loss: scalar sum of all rule violations
        """
        device = au_probs.device
        loss = torch.tensor(0.0, device=device)
        def get_p(au_num):
            if au_num in self.au_index:
                return au_probs[:, self.au_index[au_num]]
            return None
            
        p_25 = get_p(25)
        p_26 = get_p(26)
        p_24 = get_p(24)
        p_27 = get_p(27)
        
        # 1. Additive Hierarchy / Implication (Jaw Drop physically parts lips)
        # Rule: AU 26 -> AU 25 (p(AU25) >= p(AU26))
        # Fuzzy violation: p(AU26) AND NOT p(AU25)
        if p_26 is not None and p_25 is not None:
            loss = loss + (p_26 * (1.0 - p_25)).mean()
            
        # Mouth Stretch implies Lips Part & Jaw Drop
        if p_27 is not None and p_25 is not None:
            loss = loss + (p_27 * (1.0 - p_25)).mean()
        if p_27 is not None and p_26 is not None:
            loss = loss + (p_27 * (1.0 - p_26)).mean()
            
        # 2. Mutually Exclusive Rules (XOR)
        # AU 24 XOR AU 25
        if p_24 is not None and p_25 is not None:
            loss = loss + (p_24 * p_25).mean()
            
        # AU 45 XOR AU 46
        p_45, p_46 = get_p(45), get_p(46)
        if p_45 is not None and p_46 is not None:
            loss = loss + (p_45 * p_46).mean()
            
        # AU 41, 42, 43
        p_41, p_42, p_43 = get_p(41), get_p(42), get_p(43)
        if p_41 is not None and p_42 is not None:
            loss = loss + (p_41 * p_42).mean()
        if p_41 is not None and p_43 is not None:
            loss = loss + (p_41 * p_43).mean()
        if p_42 is not None and p_43 is not None:
            loss = loss + (p_42 * p_43).mean()
            
        # Opposing head and eye pairs
        for a, b in [(51, 52), (53, 54), (55, 56), (61, 62), (63, 64)]:
            pa, pb = get_p(a), get_p(b)
            if pa is not None and pb is not None:
                loss = loss + (pa * pb).mean()
                
        return loss



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
        if x.dim() != 2 or y.shape != x.shape:
            raise ValueError(f"WAL expects matching rank-2 tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
        if self.weight is not None and self.weight.numel() != x.size(1):
            raise ValueError(
                f"WAL weight count {self.weight.numel()} does not match {x.size(1)} AU outputs"
            )
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
    Eq. (4) from SymGraphAU. The negative term is weighted by the predicted
    probability, exactly as written in the paper.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_probs, targets_one_hot):
        """
        pred_probs: p^e (B, N_e) emotion probabilities after sigmoid
        targets_one_hot: y^e (B, N_e) one-hot pseudo-labels
        """
        if pred_probs.shape != targets_one_hot.shape:
            raise ValueError(
                f"expression predictions and targets must have the same shape; "
                f"got {tuple(pred_probs.shape)} and {tuple(targets_one_hot.shape)}"
            )
        xs_pos = pred_probs
        xs_neg = 1.0 - pred_probs

        los_pos = targets_one_hot * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (
            (1.0 - targets_one_hot)
            * pred_probs
            * torch.log(xs_neg.clamp(min=self.eps))
        )
        loss = los_pos + los_neg
        loss = loss.mean(dim=-1)
        return -loss.mean()


# ============================================================
# Causal Cycle Consistency Loss (Idea 4)
# ============================================================

class CausalCycleLoss(nn.Module):
    """
    Idea 4: Cross-Task Bidirectional Causal Cycle Consistency.
    Projects predicted expressions back to expected AU activations via the
    canonical expression-to-AU support matrix.
    Adds a top-down cycle loss to ensure the hierarchical predictions are consistent.
    """
    def __init__(self, rule_matrix):
        super().__init__()
        rule_matrix = torch.as_tensor(rule_matrix, dtype=torch.float32)
        if rule_matrix.dim() != 2:
            raise ValueError("rule_matrix must have shape (N_expression, N_AU)")
        self.register_buffer("rule_matrix", rule_matrix)

    def forward(self, emotion_probs, au_probs):
        """
        Args:
            emotion_probs: (B, N_EMO)
            au_probs: (B, N_AU)
        """
        if emotion_probs.size(1) != self.rule_matrix.size(0):
            raise ValueError("emotion probability dimension does not match rule_matrix")
        if au_probs.size(1) != self.rule_matrix.size(1):
            raise ValueError("AU probability dimension does not match rule_matrix")
        # rule_matrix is (N_EMO, N_AU), where a positive value means the
        # expression has a verified primary or secondary AU relation.
        # To project from Emotion down to AU, we say an AU is expected to be active
        # if ANY of the active emotions require it.
        # Probabilistic OR: P(A U B) = 1 - (1 - P(A))(1 - P(B))
        # Expected AU prob: 1 - \prod_e (1 - M_AE[e, a] * P_e)
        
        # Expand dims for broadcasting
        # emotion_probs: (B, N_EMO, 1)
        e_probs = emotion_probs.unsqueeze(-1)
        # rule_matrix: (1, N_EMO, N_AU)
        m_ae = self.rule_matrix.unsqueeze(0)
        
        # Effective prob contribution from each emotion to each AU
        effective_p = e_probs * m_ae  # (B, N_EMO, N_AU)
        
        # Prob of AU NOT being activated by each emotion
        not_active_p = 1.0 - effective_p  # (B, N_EMO, N_AU)
        
        # Prob of AU NOT being activated by ANY emotion
        not_active_any_p = not_active_p.prod(dim=1)  # (B, N_AU)
        
        # Expected AU prob
        expected_au_probs = 1.0 - not_active_any_p  # (B, N_AU)
        
        # Cycle Consistency Loss: force predicted AUs to match Expected AUs
        # We use MSE
        return F.mse_loss(au_probs, expected_au_probs)

