"""
Configuration for CtrlAU architecture.
Contains all hyperparameters, AU definitions, and prior knowledge matrices.
"""
import torch

# ============================================================
# DISFA Dataset AUs (the 8 AUs used in standard benchmarks)
# ============================================================
DISFA_AUS = [1, 2, 4, 6, 9, 12, 25, 26]
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
    9:  "Nose Wrinkler: wrinkles the nose often raising the upper lip, controlled by the levator labii superioris alaquae nasi muscle.",
    12: "Lip Corner Puller: pulls the corners of the lips upwards and outwards in a smile, controlled by the zygomaticus major muscle.",
    15: "Lip Corner Depressor: pulls the corners of the lips downwards, controlled by the depressor anguli oris muscle.",
    17: "Chin Raiser: pushes the chin upwards wrinkling chin skin and pushing up the lower lip, controlled by the mentalis muscle.",
    20: "Lip Stretcher: stretches the lips horizontally, controlled by the risorius and platysma muscles.",
    25: "Lips Part: parts the lips separating them slightly, controlled by relaxation of the orbicularis oris or depressor labii inferioris.",
    26: "Jaw Drop: drops the lower jaw parting the teeth, controlled by relaxation of the masseter and temporalis muscles.",
}

# ============================================================
# Emotion definitions
# ============================================================
EMOTIONS = ["happiness", "sadness", "surprise", "fear", "anger", "disgust"]
NUM_EMOTIONS = len(EMOTIONS)

# ============================================================
# EMFACS AU-Emotion prior rules (using DISFA AU indices)
# Each emotion maps to a list of (operator, au_indices) tuples.
# operator: "AND" uses fuzzy t-norm (min), "OR" uses fuzzy co-norm (max)
# The overall emotion score is computed by chaining these.
#
# Happiness:  AU6 AND AU12
# Sadness:    AU1 AND AU4 AND AU15
# Surprise:   AU1 AND AU2 AND AU5 AND AU26
# Fear:       AU1 AND AU2 AND AU4 AND AU5 AND AU20 AND AU26
# Anger:      AU4 AND AU5 AND AU17
# Disgust:    AU9 AND AU15
# ============================================================
EMOTION_AU_RULES = {
    "happiness": {
        "required_aus": [6, 12],   # t-norm (AND) over these
        "operator": "AND",
    },
    "sadness": {
        "required_aus": [1, 4],    # Dropped missing AU15
        "operator": "AND",
    },
    "surprise": {
        "required_aus": [1, 2, 26], # Dropped missing AU5
        "operator": "AND",
    },
    "fear": {
        "required_aus": [1, 2, 4, 26], # Dropped missing AU5, AU20
        "operator": "AND",
    },
    "anger": {
        "required_aus": [4],        # Dropped missing AU5, AU17
        "operator": "AND",
    },
    "disgust": {
        "required_aus": [9],        # Dropped missing AU15
        "operator": "AND",
    },
}

# Convert AU numbers to indices for fast lookup
EMOTION_AU_RULES_IDX = {}
for emo, rule in EMOTION_AU_RULES.items():
    EMOTION_AU_RULES_IDX[emo] = {
        "required_idx": [AU_INDEX[au] for au in rule["required_aus"]],
        "operator": rule["operator"],
    }

# Dynamically construct Emotion textual descriptions based on their AU components
EMOTION_DESCRIPTIONS = {}
for emo, rule in EMOTION_AU_RULES.items():
    # Join the textual descriptions of all required AUs with " and "
    desc_list = [AU_DESCRIPTIONS[au] for au in rule["required_aus"]]
    EMOTION_DESCRIPTIONS[emo] = f"A facial expression of {emo}, characterized by: " + " and ".join(desc_list)


# ============================================================
# Model hyperparameters
# ============================================================
class ModelConfig:
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
    
    # Counterfactual perturbation
    noise_std = 0.1                # Gaussian noise std for perturbation
    
    # Loss weights
    lambda_au = 1.0                # Base AU BCE loss
    lambda_ib = 1e-2               # HSIC information bottleneck
    lambda_align = 1e-2            # HSIC alignment
    lambda_decorr = 1e-2           # HSIC decorrelation
    lambda_contrastive = 0.1       # Text-visual contrastive loss for AUs
    lambda_emo_contrastive = 0.1   # Text-visual contrastive loss for Emotions
    
    lambda_au_au = 1.0             # Phase 2: graph AU prediction
    lambda_dag = 0.1               # DAG constraint on AU-AU graph
    lambda_causal_au = 0.1         # Causal structural rule on AU-AU graph
    lambda_causal_exp = 0.1        # Causal structural rule on AU-Exp graph
    lambda_facs_au = 0.1           # FACS rules on AU-AU graph (XOR, subsume)
    lambda_facs_exp = 0.1          # FACS emotion rules on AU-Exp graph loss
    lambda_cf_important = 0.1      # counterfactual important perturbation
    lambda_cf_unimportant = 0.1    # counterfactual unimportant perturbation
    lambda_emotion = 0.1           # emotion weak supervision
    lambda_au_au = 1.0             # intermediate AU-AU graph loss
    lambda_graph_au = 1.0          # Graph AU classification loss
    lambda_graph_emo = 1.0         # Graph Emotion classification loss
    
    # Training
    lr = 1e-5                      # Peak learning rate (Original repo used 1e-5)
    weight_decay = 5e-4            # Weight decay for AdamW (Original repo used 5e-4)
    batch_size = 64
    num_epochs = 20
    clip_model_name = "openai/clip-vit-base-patch32"
