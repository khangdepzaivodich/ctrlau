"""
CtrlAU: Full model integration.
Ties together all modules into a single forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DISFA_AUS, NUM_AUS, AU_DESCRIPTIONS, NUM_EMOTIONS,
    ModelConfig,
)
from losses import (
    HSICDisentanglementLoss, ContrastiveLoss, DAGLoss,
    ViolationLoss, CounterfactualLoss,
)
from .backbone import VisualBackbone
from .text_encoder import TextEncoder
from .heads import AUHead, EmotionHead
from .gat import AUGraphModule
from .masks import MaskModule


class CtrlAUModel(nn.Module):
    """
    Full CtrlAU architecture.
    
    Pipeline:
    1. Text descriptions -> CLIP -> text_emb (frozen)
    2. Image -> ResNet50 -> z_img
    3. z_img -> AU Head -> per-AU embeddings + AU predictions
    4. z_img + AU probs -> Emotion Head -> emotion embeddings + fuzzy pseudo-labels
    5. HSIC disentanglement on AU embeddings
    6. Contrastive alignment between AU embeddings and text embeddings
    7. AU embeddings -> GAT -> AU-AU graph + AU-Exp graph
    8. Graphs -> Masks (importance + polarity)
    9. Masks -> Violation loss
    10. Masks -> Counterfactual perturbation -> re-forward -> CF losses
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        
        # ---- Modules ----
        self.backbone = VisualBackbone(
            feat_dim=cfg.backbone_feat_dim,
            pretrained=True,
        )
        
        self.text_encoder = TextEncoder(
            clip_model_name=cfg.clip_model_name,
        )
        
        self.au_head = AUHead(
            backbone_dim=cfg.backbone_feat_dim,
            num_aus=NUM_AUS,
            au_embed_dim=cfg.au_embed_dim,
        )
        
        self.emotion_head = EmotionHead(
            backbone_dim=cfg.backbone_feat_dim,
            num_emotions=NUM_EMOTIONS,
            emotion_embed_dim=cfg.emotion_embed_dim,
        )
        
        self.graph_module = AUGraphModule(
            embed_dim=cfg.au_embed_dim,
            num_aus=NUM_AUS,
            num_emotions=NUM_EMOTIONS,
            gat_hidden_dim=cfg.gat_hidden_dim,
            gat_num_heads=cfg.gat_num_heads,
            gat_num_layers=cfg.gat_num_layers,
            dropout=cfg.gat_dropout,
        )
        
        self.mask_module = MaskModule(
            num_aus=NUM_AUS,
            num_emotions=NUM_EMOTIONS,
        )
        
        # ---- Projection heads for contrastive alignment ----
        self.visual_proj = nn.Sequential(
            nn.Linear(cfg.au_embed_dim, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.embed_dim, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        # ---- Loss modules ----
        from losses import HSICDisentanglementLoss, ContrastiveLoss, DAGLoss, ViolationLoss, CounterfactualLoss, FocalLoss
        self.hsic_loss = HSICDisentanglementLoss(num_aus=NUM_AUS)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.dag_loss = DAGLoss(num_nodes=NUM_AUS)
        self.violation_loss = ViolationLoss()
        self.cf_loss = CounterfactualLoss()
        
        # ---- AU detection loss (Focal Loss with Dataset Pos Weights) ----
        # Pos weights calculated from DISFA dataset to penalize rare classes heavily
        # [AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26]
        au_pos_weights = torch.tensor([
            19.11, 22.18, 5.56, 112.75, 11.67, 22.90, 
            6.76, 47.77, 18.86, 43.48, 2.61, 10.34
        ])
        self.au_bce_loss = FocalLoss(gamma=2.0, pos_weight=au_pos_weights)
        
        # ---- Emotion weak supervision loss ----
        self.emotion_bce_loss = nn.BCELoss()
        
        # ---- Precompute text embeddings (descriptions are fixed) ----
        self._text_descriptions = [AU_DESCRIPTIONS[au] for au in DISFA_AUS]
        self._cached_text_emb = None
    
    def _get_text_embeddings(self):
        """Get or compute cached text embeddings."""
        if self._cached_text_emb is None:
            self._cached_text_emb = self.text_encoder(self._text_descriptions)
        return self._cached_text_emb
    
    def _perturb_features(self, z_img, mask_vector, mode="important"):
        """
        Perturb z_img features based on importance mask.
        
        For 'important': add noise to important dimensions (mask=1)
        For 'unimportant': add noise to unimportant dimensions (mask=0)
        
        Args:
            z_img: (B, D) feature vector
            mask_vector: (D,) importance mask (values in {0, 1})
            mode: 'important' or 'unimportant'
        Returns:
            perturbed_z: (B, D)
        """
        noise = torch.randn_like(z_img) * self.cfg.noise_std
        
        if mode == "important":
            # Perturb where mask = 1
            perturbed_z = z_img + noise * mask_vector.unsqueeze(0)
        else:
            # Perturb where mask = 0
            perturbed_z = z_img + noise * (1.0 - mask_vector.unsqueeze(0))
        
        return perturbed_z
    
    def forward(self, images, au_labels=None):
        """
        Full forward pass.
        
        Args:
            images: (B, 3, H, W) input images
            au_labels: (B, N_AU) binary AU labels (for training)
        Returns:
            dict with predictions and losses
        """
        device = images.device
        B = images.size(0)
        
        # ============================================================
        # 1. Visual backbone
        # ============================================================
        z_img = self.backbone(images)  # (B, 2048)
        
        # ============================================================
        # 2. AU Head
        # ============================================================
        au_embeddings, au_logits, au_probs = self.au_head(z_img)
        # au_embeddings: list of N_AU tensors, each (B, au_embed_dim)
        # au_logits: (B, N_AU)
        # au_probs: (B, N_AU)
        
        # ============================================================
        # 3. Emotion Head (weak supervision via fuzzy logic)
        # ============================================================
        emotion_embed, emotion_logits, emotion_probs, emotion_pseudo = self.emotion_head(z_img, au_probs)
        
        # ============================================================
        # 4. Stack AU embeddings for graph processing
        # ============================================================
        au_emb_stacked = torch.stack(au_embeddings, dim=1)  # (B, N_AU, D)
        
        # ============================================================
        # 5. GAT: AU-AU graph
        # ============================================================
        updated_au, au_au_adj = self.graph_module.forward_au_au(au_emb_stacked)
        # updated_au: (B, N_AU, D)
        # au_au_adj: (N_AU, N_AU) sigmoid weights
        
        # ============================================================
        # 6. GAT: AU-Expression graph
        # ============================================================
        updated_nodes, au_exp_adj = self.graph_module.forward_au_exp(au_emb_stacked, emotion_embed)
        # updated_nodes: (B, N_AU + N_EMO, D)
        
        # ============================================================
        # 7. Masks from graphs
        # ============================================================
        au_au_imp_mask, au_au_pol_mask = self.mask_module.forward_au_au(au_au_adj)
        au_exp_imp_mask, au_exp_pol_mask = self.mask_module.forward_au_exp(au_exp_adj)
        
        # ============================================================
        # Compute losses (only during training)
        # ============================================================
        losses = {}
        
        if au_labels is not None:
            # --- AU detection loss ---
            loss_au = self.au_bce_loss(au_logits, au_labels)
            losses["loss_au"] = loss_au
            
            # --- HSIC disentanglement ---
            l_ib, l_align, l_decorr = self.hsic_loss(au_embeddings, z_img, au_labels)
            losses["loss_ib"] = l_ib
            losses["loss_align"] = l_align
            losses["loss_decorr"] = l_decorr
            
            # --- Contrastive loss (text-visual alignment) ---
            text_emb = self._get_text_embeddings()  # (N_AU, text_dim)
            
            # Average AU embeddings over batch for contrastive alignment
            au_emb_mean = torch.stack([e.mean(dim=0) for e in au_embeddings])  # (N_AU, D)
            
            visual_proj = self.visual_proj(au_emb_mean)   # (N_AU, shared_dim)
            text_proj = self.text_proj(text_emb)           # (N_AU, shared_dim)
            
            loss_contrastive = self.contrastive_loss(visual_proj, text_proj)
            losses["loss_contrastive"] = loss_contrastive
            
            # --- DAG constraint on AU-AU graph ---
            loss_dag = self.dag_loss(au_au_adj)
            losses["loss_dag"] = loss_dag
            
            # --- Violation loss ---
            loss_violation = self.violation_loss(
                au_probs, au_au_adj, au_au_pol_mask, au_au_imp_mask
            )
            losses["loss_violation"] = loss_violation
            
            # --- Emotion weak supervision ---
            # Use fuzzy pseudo-labels as targets for emotion head
            loss_emotion = self.emotion_bce_loss(emotion_probs, emotion_pseudo.detach())
            losses["loss_emotion"] = loss_emotion
            
            # --- Counterfactual intervention ---
            # Derive a z_img-level mask from AU importance.
            # Each AU's importance (diagonal of AU-AU importance mask) weights
            # how much that AU's corresponding shared_fc features matter.
            # We create a (D_backbone,) mask by thresholding z_img feature magnitudes.
            z_img_magnitude = z_img.detach().mean(dim=0).abs()  # (D_backbone,)
            cf_mask = (z_img_magnitude > z_img_magnitude.median()).float()  # (D_backbone,)
            
            # Important perturbation
            z_img_imp_perturbed = self._perturb_features(z_img, cf_mask, mode="important")
            _, au_logits_imp, au_probs_imp = self.au_head(z_img_imp_perturbed)
            
            # Unimportant perturbation
            z_img_unimp_perturbed = self._perturb_features(z_img, cf_mask, mode="unimportant")
            _, au_logits_unimp, au_probs_unimp = self.au_head(z_img_unimp_perturbed)
            
            loss_cf_imp, loss_cf_unimp = self.cf_loss(
                au_probs, au_probs_imp, au_probs_unimp
            )
            losses["loss_cf_important"] = loss_cf_imp
            losses["loss_cf_unimportant"] = loss_cf_unimp
            
            # --- Total loss ---
            cfg = self.cfg
            total_loss = (
                cfg.lambda_au * loss_au +
                cfg.lambda_ib * l_ib +
                cfg.lambda_align * l_align +
                cfg.lambda_decorr * l_decorr +
                cfg.lambda_contrastive * loss_contrastive +
                cfg.lambda_dag * loss_dag +
                cfg.lambda_violation * loss_violation +
                cfg.lambda_emotion * loss_emotion +
                cfg.lambda_cf_important * loss_cf_imp +
                cfg.lambda_cf_unimportant * loss_cf_unimp
            )
            losses["total_loss"] = total_loss
        
        return {
            "au_logits": au_logits,
            "au_probs": au_probs,
            "au_embeddings": au_embeddings,
            "emotion_probs": emotion_probs,
            "emotion_pseudo": emotion_pseudo,
            "emotion_embed": emotion_embed,
            "au_au_adj": au_au_adj,
            "au_exp_adj": au_exp_adj,
            "au_au_importance_mask": au_au_imp_mask,
            "au_au_polarity_mask": au_au_pol_mask,
            "au_exp_importance_mask": au_exp_imp_mask,
            "au_exp_polarity_mask": au_exp_pol_mask,
            "z_img": z_img,
            "losses": losses,
        }
