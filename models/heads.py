"""
AU Head and Emotion Head.
- AU Head: produces per-AU embeddings and predictions from z_img.
- Emotion Head: weakly supervised via fuzzy logic on AU predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMOTION_AU_RULES_IDX, EMOTIONS, NUM_EMOTIONS


class AUHead(nn.Module):
    """
    AU detection head.
    Takes z_img (B, D_backbone) and produces:
    - per-AU embeddings: list of N_AU tensors, each (B, au_embed_dim)
    - AU predictions: (B, N_AU) probabilities
    """
    
    def __init__(self, backbone_dim, num_aus, au_embed_dim=256):
        super().__init__()
        self.num_aus = num_aus
        self.au_embed_dim = au_embed_dim
        
        # Shared feature transform
        self.shared_fc = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # Per-AU embedding heads
        self.au_embed_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, au_embed_dim),
                nn.BatchNorm1d(au_embed_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_aus)
        ])
        
        # Per-AU classifiers (binary)
        self.au_classifiers = nn.ModuleList([
            nn.Linear(au_embed_dim, 1)
            for _ in range(num_aus)
        ])
    
    def forward(self, z_img):
        """
        Args:
            z_img: (B, D_backbone)
        Returns:
            au_embeddings: list of (B, au_embed_dim) tensors
            au_logits: (B, N_AU) raw logits
            au_probs: (B, N_AU) sigmoid probabilities
        """
        shared = self.shared_fc(z_img)  # (B, 512)
        
        au_embeddings = []
        au_logits_list = []
        
        for i in range(self.num_aus):
            emb = self.au_embed_heads[i](shared)      # (B, au_embed_dim)
            logit = self.au_classifiers[i](emb)         # (B, 1)
            au_embeddings.append(emb)
            au_logits_list.append(logit)
        
        au_logits = torch.cat(au_logits_list, dim=1)    # (B, N_AU)
        au_probs = torch.sigmoid(au_logits)
        
        return au_embeddings, au_logits, au_probs


class EmotionHead(nn.Module):
    """
    Emotion head that is weakly supervised via fuzzy logic operators
    applied to AU predictions.
    
    Uses fuzzy t-norm (min for AND) and t-conorm (max for OR)
    to compute emotion pseudo-probabilities from AU predictions.
    
    Also has a learnable emotion embedding branch.
    """
    
    def __init__(self, backbone_dim, num_emotions=NUM_EMOTIONS, emotion_embed_dim=256):
        super().__init__()
        self.num_emotions = num_emotions
        
        # Learnable emotion embedding from visual features
        self.emotion_fc = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emotion_embed_dim),
            nn.BatchNorm1d(emotion_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Emotion classifier
        self.emotion_classifier = nn.Linear(emotion_embed_dim, num_emotions)
    
    def compute_fuzzy_emotions(self, au_probs):
        """
        Compute emotion pseudo-labels using fuzzy logic on AU predictions.
        
        Fuzzy AND (t-norm) = element-wise min (product t-norm is also an option)
        Fuzzy OR (t-conorm) = element-wise max
        
        We use product t-norm for differentiability:
            AND(a, b) = a * b (product t-norm)
        
        Args:
            au_probs: (B, N_AU) AU probabilities
        Returns:
            emotion_pseudo: (B, N_EMOTIONS) pseudo emotion probabilities
        """
        B = au_probs.size(0)
        device = au_probs.device
        emotion_pseudo = torch.zeros(B, self.num_emotions, device=device)
        
        for emo_idx, emo_name in enumerate(EMOTIONS):
            rule = EMOTION_AU_RULES_IDX[emo_name]
            au_indices = rule["required_idx"]
            
            # Gather the relevant AU probabilities
            relevant_probs = au_probs[:, au_indices]  # (B, K)
            
            if rule["operator"] == "AND":
                # Product t-norm: multiply all probabilities
                emo_score = relevant_probs.prod(dim=1)  # (B,)
            else:  # "OR"
                # Probabilistic co-norm: 1 - prod(1 - p_i)
                emo_score = 1.0 - (1.0 - relevant_probs).prod(dim=1)
            
            emotion_pseudo[:, emo_idx] = emo_score
        
        return emotion_pseudo
    
    def forward(self, z_img, au_probs):
        """
        Args:
            z_img: (B, D_backbone) image features
            au_probs: (B, N_AU) AU probabilities from AU head
        Returns:
            emotion_embed: (B, emotion_embed_dim)
            emotion_logits: (B, N_EMOTIONS)
            emotion_probs: (B, N_EMOTIONS)
            emotion_pseudo: (B, N_EMOTIONS) fuzzy logic pseudo-labels
        """
        emotion_embed = self.emotion_fc(z_img)
        emotion_logits = self.emotion_classifier(emotion_embed)
        emotion_probs = torch.sigmoid(emotion_logits)
        
        # Compute pseudo-labels from AU predictions
        emotion_pseudo = self.compute_fuzzy_emotions(au_probs)
        
        return emotion_embed, emotion_logits, emotion_probs, emotion_pseudo
