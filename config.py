"""
Configuration for CtrlAU architecture.
Contains all hyperparameters, AU definitions, and prior knowledge matrices.
"""
import torch

# ============================================================
# DISFA Dataset AUs (the 12 AUs labeled in DISFA)
# ============================================================
DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
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
        "required_aus": [1, 4, 15],
        "operator": "AND",
    },
    "surprise": {
        "required_aus": [1, 2, 5, 26],
        "operator": "AND",
    },
    "fear": {
        "required_aus": [1, 2, 4, 5, 20, 26],
        "operator": "AND",
    },
    "anger": {
        "required_aus": [4, 5, 17],
        "operator": "AND",
    },
    "disgust": {
        "required_aus": [9, 15],
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

# ============================================================
# Model hyperparameters
# ============================================================
class ModelConfig:
    # Image
    img_size = 224
    
    # Backbone
    backbone = "resnet18"
    backbone_feat_dim = 512        # ResNet18 final feature dim
    
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
    lambda_au = 1.0                # AU detection loss
    lambda_ib = 0.1                # information bottleneck (minimize dep with zimg)
    lambda_align = 0.1             # align AU repr with labels
    lambda_decorr = 0.1            # decorrelate different AU reprs
    lambda_contrastive = 0.1       # contrastive text-visual alignment
    lambda_dag = 0.1               # DAG constraint
    lambda_violation = 0.1         # rule violation loss
    lambda_cf_important = 0.1      # counterfactual important perturbation
    lambda_cf_unimportant = 0.1    # counterfactual unimportant perturbation
    lambda_emotion = 0.1           # emotion weak supervision
    
    # Training
    lr = 1e-4
    weight_decay = 1e-5
    batch_size = 64
    num_epochs = 50
    clip_model_name = "openai/clip-vit-base-patch32"
