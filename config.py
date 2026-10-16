"""
Configuration for CtrlAU architecture.
Contains all hyperparameters, AU definitions, and prior knowledge matrices.
"""
import torch

# ============================================================
# Dataset AU vocabularies
# ============================================================
DISFA_AUS = [1, 2, 4, 6, 9, 12, 25, 26]
BP4D_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
DATASET_AUS = {
    "disfa": tuple(DISFA_AUS),
    "bp4d": tuple(BP4D_AUS),
}

# Kept for legacy DISFA-only callers. CtrlAU uses cfg.au_ids instead.
NUM_AUS = len(DISFA_AUS)
AU_INDEX = {au: i for i, au in enumerate(DISFA_AUS)}

# ============================================================
# Textual AU Descriptions (for CLIP text encoder)
# ============================================================
AU_DESCRIPTIONS = {
    1:  "Inner Brow Raiser: pulls the inner corners of the eyebrows upwards, controlled by the frontalis pars medialis muscle.",
    2:  "Outer Brow Raiser: pulls the outer corners of the eyebrows upwards, controlled by the frontalis pars lateralis muscle.",
    4:  "Brow Lowerer: lowers and draws the eyebrows together, controlled by the corrugator supercilii and depressor supercilii muscles.",
    5:  "Upper Lid Raiser: raises the upper eyelid widening the eye, controlled by the levator palpebrae superioris muscle.",
    6:  "Cheek Raiser: raises the cheeks causing crow's feet wrinkles around the eyes, controlled by the orbicularis oculi pars orbitalis muscle.",
    7:  "Lid Tightener: tightens the eyelids, controlled by the orbicularis oculi muscle.",
    9:  "Nose Wrinkler: wrinkles the nose often raising the upper lip, controlled by the levator labii superioris alaquae nasi muscle.",
    10: "Upper Lip Raiser: raises the upper lip, controlled by the levator labii superioris muscle.",
    12: "Lip Corner Puller: pulls the corners of the lips upwards and outwards in a smile, controlled by the zygomaticus major muscle.",
    14: "Dimpler: dimples the cheeks by pulling the corners of the mouth laterally, controlled by the buccinator muscle.",
    15: "Lip Corner Depressor: pulls the corners of the lips downwards, controlled by the depressor anguli oris muscle.",
    17: "Chin Raiser: pushes the chin upwards wrinkling chin skin and pushing up the lower lip, controlled by the mentalis muscle.",
    20: "Lip Stretcher: stretches the lips horizontally, controlled by the risorius and platysma muscles.",
    23: "Lip Tightener: tightens and narrows the lips, controlled by the orbicularis oris muscle.",
    24: "Lip Pressor: presses the lips together, controlled by the orbicularis oris muscle.",
    25: "Lips Part: parts the lips separating them slightly, controlled by relaxation of the orbicularis oris or depressor labii inferioris.",
    26: "Jaw Drop: drops the lower jaw parting the teeth, controlled by relaxation of the masseter and temporalis muscles.",
}

# ============================================================
# Emotion definitions
# ============================================================
EMOTIONS = ["happiness", "sadness", "surprise", "fear", "anger", "disgust"]
NUM_EMOTIONS = len(EMOTIONS)

# Legacy seven-class order used by the copied MultiviewSymAU wrapper. Do not
# use this order for CtrlAU; the published SymGraphAU prior has eight classes.
SYM_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]
NUM_SYM_EMOTIONS = len(SYM_EMOTIONS)
# Published SymGraphAU expression order and FACS relationships. The order is
# kept explicit because it is the column order of M_AE.
CTRL_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Contempt", "Neutral"]
CTRL_NUM_EMOTIONS = len(CTRL_EMOTIONS)
CTRL_EMOTION_INDEX = {name: i for i, name in enumerate(CTRL_EMOTIONS)}
PAPER_BETA = 0.1

# Source: SymGraphAU supplementary Table 2. Each relation is either primary
# (1-beta), secondary (0.5), or absent (beta). Neutral has no positive AU
# relations. This is the paper's prior, not a hard emotion detector.
PAPER_AU_EXPRESSION_RELATIONS = {
    "Surprise": {"primary": [1, 2, 25, 26], "secondary": []},
    "Fear": {"primary": [1, 2, 4, 7], "secondary": [23, 25]},
    "Disgust": {"primary": [9, 10], "secondary": [25]},
    "Happy": {"primary": [6, 12], "secondary": [7, 14, 25]},
    "Sad": {"primary": [1, 7, 15], "secondary": [4, 6, 17]},
    "Angry": {"primary": [4, 23, 24], "secondary": [7, 17]},
    "Contempt": {"primary": [12, 14, 23], "secondary": []},
    "Neutral": {"primary": [], "secondary": []},
}

PAPER_AU_IDS = sorted({
    au
    for relation in PAPER_AU_EXPRESSION_RELATIONS.values()
    for level in ("primary", "secondary")
    for au in relation[level]
})


def build_au_expression_prior(au_ids, beta=PAPER_BETA):
    """Build the paper's M_A-E matrix in the requested AU order."""
    au_ids = tuple(int(au) for au in au_ids)
    if len(set(au_ids)) != len(au_ids):
        raise ValueError(f"AU ids must be unique; got {au_ids}")
    if not 0.0 < float(beta) < 0.5:
        raise ValueError(f"beta must be in (0, 0.5); got {beta}")
    unknown = sorted(set(au_ids) - set(PAPER_AU_IDS))
    if unknown:
        raise ValueError(f"No verified SymGraphAU prior exists for AU(s): {unknown}")

    matrix = torch.full((len(au_ids), CTRL_NUM_EMOTIONS), float(beta))
    au_to_row = {au: i for i, au in enumerate(au_ids)}
    for emotion, relation in PAPER_AU_EXPRESSION_RELATIONS.items():
        emotion_idx = CTRL_EMOTION_INDEX[emotion]
        for au in relation["primary"]:
            if au in au_to_row:
                matrix[au_to_row[au], emotion_idx] = 1.0 - beta
        for au in relation["secondary"]:
            if au in au_to_row:
                matrix[au_to_row[au], emotion_idx] = 0.5
    return matrix


DISFA_M_AE = build_au_expression_prior(DISFA_AUS)
BP4D_M_AE = build_au_expression_prior(BP4D_AUS)


def get_dataset_aus(dataset_name):
    key = dataset_name.lower()
    if key not in DATASET_AUS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {sorted(DATASET_AUS)}")
    return tuple(DATASET_AUS[key])


def get_dataset_au_expression_prior(dataset_name):
    key = dataset_name.lower()
    get_dataset_aus(key)
    return build_au_expression_prior(DATASET_AUS[key])


def compute_expression_pseudo_labels(au_values, au_expression_prior):
    """Implement SymGraphAU Eq. (1)-(2): AU prior scores -> one-hot labels."""
    if au_values.dim() != 2:
        raise ValueError(f"au_values must have shape (B, N_AU), got {tuple(au_values.shape)}")
    if au_values.size(1) != au_expression_prior.size(0):
        raise ValueError(
            f"AU/prior mismatch: got {au_values.size(1)} AU values but prior has "
            f"{au_expression_prior.size(0)} rows"
        )
    prior = au_expression_prior.to(device=au_values.device, dtype=au_values.dtype)
    scores = au_values.float() @ prior.float()
    labels = scores.argmax(dim=1)
    neutral_idx = CTRL_EMOTION_INDEX["Neutral"]
    labels[au_values.float().sum(dim=1) == 0] = neutral_idx
    targets = torch.zeros(
        au_values.size(0), prior.size(1), device=au_values.device, dtype=au_values.dtype
    )
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    return targets


def prior_support_matrix(au_expression_prior, beta=PAPER_BETA):
    """Convert primary/secondary relations to an emotion-by-AU support mask."""
    return (au_expression_prior > float(beta)).t().float()


def build_emotion_descriptions():
    descriptions = {}
    for emotion in CTRL_EMOTIONS:
        relation = PAPER_AU_EXPRESSION_RELATIONS[emotion]
        primary = ", ".join(f"AU{au}" for au in relation["primary"]) or "none"
        secondary = ", ".join(f"AU{au}" for au in relation["secondary"]) or "none"
        descriptions[emotion] = (
            f"A facial expression of {emotion.lower()}; primary facial action units: "
            f"{primary}; secondary facial action units: {secondary}."
        )
    return descriptions


CTRL_EMOTION_DESCRIPTIONS = build_emotion_descriptions()

# Backward-compatible names used by older callers. These names now refer to
# the verified DISFA prior rather than the removed hard-rule approximation.
CTRL_AU_M_AE = DISFA_M_AE.t().contiguous()
EMOTION_DESCRIPTIONS = {
    key.lower(): value for key, value in CTRL_EMOTION_DESCRIPTIONS.items()
}

# Legacy numpy matrix retained for callers that expect emotion-by-AU layout.
import numpy as np
M_AE = CTRL_AU_M_AE.numpy().copy()


# ============================================================
# Model hyperparameters
# ============================================================
class ModelConfig:
    # Explicit default benchmark protocol. Training overrides these fields
    # only after selecting --dataset.
    dataset_name = "disfa"
    au_ids = tuple(DISFA_AUS)
    au_expression_prior = DISFA_M_AE
    phase = 1

    # Image
    img_size = 224
    
    # Backbone
    backbone = "resnet50"
    backbone_feat_dim = 2048       # ResNet50 final feature dim
    
    # Embedding dims
    au_embed_dim = 256             # per-AU embedding dimension
    text_embed_dim = 512           # CLIP text embedding dim
    shared_embed_dim = 256         # shared projection space for contrastive
    emotion_embed_dim = 256        # emotion head embedding dim
    
    # GAT
    gat_hidden_dim = 256
    gat_num_heads = 4
    gat_num_layers = 2
    gat_dropout = 0.1
    
    # Learnable threshold
    threshold_init = 0.5           # initial sigmoid pre-activation for mask threshold
    
    # Counterfactual perturbation (CausalAffect Eq. 11-15)
    noise_std = 0.1                # Gaussian noise std for perturbation
    cf_gamma = 5.0                 # Sharpness factor for soft mask sigmoid
    cf_intervention_scale = 5.0    # Scale factor alpha for counterfactual subspace erasure (Option A)
    
    
    # Loss weights: Feature Extraction & Regularizers (Phase 1, 2, 3)
    lambda_au = 1.0                # Base AU Weighted Asymmetric Loss
    lambda_emotion = 0.05          # Expression BCE loss weight (MultiviewSymAU gamma=0.05)
    lambda_ib = 1e-2               # HSIC information bottleneck
    lambda_align = 1e-2            # HSIC label alignment
    lambda_decorr = 1e-2           # HSIC cross-AU decorrelation
    lambda_contrastive = 0.05      # Text-visual contrastive loss for AUs (masked to y=1)
    lambda_emo_contrastive = 0.05  # Text-visual contrastive loss for Emotions (masked to y=1)
    lambda_facs_au = 0.1           # FACS anatomical rules on AU probabilities (AU26 -> AU25, etc.)
    lambda_cycle = 0.1             # Cross-Task Bidirectional Causal Cycle Consistency (Idea 4)
    
    # Loss weights: Graph, Causal DAG & Counterfactual Reasoning (Phase 2 & 3)
    lambda_au_au = 1.0             # Intermediate AU-AU graph classification loss
    lambda_graph_au = 1.0          # Final Graph AU classification loss
    lambda_graph_emo = 1.0         # Final Graph Emotion classification loss
    lambda_dag = 0.1               # DAG constraint on AU-AU adjacency matrix
    lambda_causal_au = 0.1         # Causal structural rule on AU-AU graph
    lambda_causal_exp = 0.1        # Causal structural rule on AU-Exp graph
    lambda_au_expression_prior = 0.1  # Verified AU-expression support regularizer (Phase 2/3)
    # Counterfactual Intervention Weights (Level 1: AU-AU, Level 2: AU-Exp)
    lambda_cf_au_imp = 0.1         # Level 1 AU-AU causal discrepancy
    lambda_cf_au_unimp = 0.1       # Level 1 AU-AU non-causal consistency
    lambda_cf_exp_imp = 0.1        # Level 2 AU-Exp causal discrepancy
    lambda_cf_exp_unimp = 0.1      # Level 2 AU-Exp non-causal consistency
    
    # Training (Differential Learning Rate to prevent overfitting on DISFA video frames)
    lr = 0.0001                    # Default / heads learning rate
    lr_backbone = 1e-5             # Backbone learning rate (preserves ImageNet representations)
    lr_head = 1e-4                 # Heads learning rate (learns AU and Emotion classifiers fast)
    weight_decay = 5e-4            # Weight decay for AdamW (Original repo used 5e-4)
    batch_size = 64
    num_epochs = 20
    clip_model_name = "openai/clip-vit-base-patch32"
