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
    ModelConfig, EMOTION_DESCRIPTIONS, EMOTIONS
)
from losses import (
    HSICDisentanglementLoss, ContrastiveLoss, DAGLoss,
    ViolationLoss, FACSEmotionViolationLoss, CounterfactualLoss, FocalLoss,
    FACSAUViolationLoss, WeightedAsymmetricLoss
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
            name=cfg.backbone,
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
        
        self.emotion_visual_proj = nn.Sequential(
            nn.Linear(cfg.au_embed_dim, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        self.emotion_text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.embed_dim, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        # ---- Graph Classifiers ----
        # Takes the updated embeddings from GAT and outputs graph-based predictions
        self.au_au_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(NUM_AUS)
        ])
        self.graph_au_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(NUM_AUS)
        ])
        self.graph_emo_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(NUM_EMOTIONS)
        ])
        
        # ---- Loss modules ----
        self.hsic_loss = HSICDisentanglementLoss(num_aus=NUM_AUS)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.dag_loss = DAGLoss(num_nodes=NUM_AUS)
        self.violation_loss = ViolationLoss()
        self.facs_emotion_violation_loss = FACSEmotionViolationLoss()
        self.facs_au_violation_loss = FACSAUViolationLoss()
        self.cf_loss = CounterfactualLoss()
        
        # ---- AU detection loss (Focal Loss with Dataset Pos Weights) ----
        # Original exact dataset weights (Negative / Positive samples):
        # [AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26]
        # We apply Square Root smoothing to prevent extreme imbalances 
        # from destroying precision by over-predicting positive classes.
        raw_au_pos_weights = torch.tensor([
            19.11, 22.18, 5.56, 11.67, 22.90, 6.76, 2.61, 10.34
        ])
        au_pos_weights = torch.sqrt(raw_au_pos_weights)
        self.register_buffer("au_pos_weights", au_pos_weights)
        
        self.au_bce_loss = FocalLoss(gamma=2.0, pos_weight=au_pos_weights)
        
        # ---- Emotion weak supervision loss ----
        self.emotion_bce_loss = nn.BCELoss()
        
        # ---- Counterfactual mask threshold ----
        self.cf_threshold = nn.Parameter(torch.tensor(0.5))
        
        # ---- Precompute text embeddings (descriptions are fixed) ----
        self._text_descriptions = [AU_DESCRIPTIONS[au] for au in DISFA_AUS]
        self._cached_text_emb = None
        
        from config import EMOTIONS
        self._emotion_text_descriptions = [EMOTION_DESCRIPTIONS[emo] for emo in EMOTIONS]
        self._cached_emotion_text_emb = None
    
    def _get_text_embeddings(self):
        """Get or compute cached text embeddings."""
        if self._cached_text_emb is None:
            self._cached_text_emb = self.text_encoder(self._text_descriptions)
        return self._cached_text_emb
        
    def _get_emotion_text_embeddings(self):
        """Get or compute cached text embeddings for emotions."""
        if self._cached_emotion_text_emb is None:
            self._cached_emotion_text_emb = self.text_encoder(self._emotion_text_descriptions)
        return self._cached_emotion_text_emb
    

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
        
        # Ground emotion pseudo-labels on real labels during training (100% clean targets)
        au_for_pseudo = au_labels.float() if (self.training and au_labels is not None) else au_probs
        emotion_embed_list, emotion_logits, emotion_probs, emotion_pseudo = self.emotion_head(z_img, au_for_pseudo)
        
        # ============================================================
        # 4. Stack embeddings for graph processing
        # ============================================================
        au_emb_stacked = torch.stack(au_embeddings, dim=1)           # (B, N_AU, D)
        emotion_emb_stacked = torch.stack(emotion_embed_list, dim=1) # (B, N_EMO, D)
        
        # ============================================================
        # 5. GAT: AU-AU graph
        # ============================================================
        updated_au, au_au_adj = self.graph_module.forward_au_au(au_emb_stacked)
        # updated_au: (B, N_AU, D)
        # au_au_adj: (N_AU, N_AU) sigmoid weights
        
        au_au_logits = []
        for i in range(NUM_AUS):
            au_au_logits.append(self.au_au_classifiers[i](updated_au[:, i, :]))
        au_au_logits = torch.cat(au_au_logits, dim=1) # (B, N_AU)
        au_au_probs = torch.sigmoid(au_au_logits)
        
        # ============================================================
        # 6. GAT: AU-Expression graph
        # ============================================================
        updated_nodes, au_exp_adj = self.graph_module.forward_au_exp(au_emb_stacked, emotion_emb_stacked)
        # updated_nodes: (B, N_AU + N_EMO, D)
        
        # Split updated_nodes back into AU and Emo
        updated_au_final = updated_nodes[:, :NUM_AUS, :]  # (B, N_AU, D)
        updated_emo_final = updated_nodes[:, NUM_AUS:, :] # (B, N_EMO, D)
        
        # Apply Graph Classifiers
        graph_au_logits = []
        for i in range(NUM_AUS):
            graph_au_logits.append(self.graph_au_classifiers[i](updated_au_final[:, i, :]))
        graph_au_logits = torch.cat(graph_au_logits, dim=1) # (B, N_AU)
        graph_au_probs = torch.sigmoid(graph_au_logits)
        
        graph_emo_logits = []
        for i in range(NUM_EMOTIONS): # 7 emotions
            graph_emo_logits.append(self.graph_emo_classifiers[i](updated_emo_final[:, i, :]))
        graph_emo_logits = torch.cat(graph_emo_logits, dim=1) # (B, N_EMO)
        graph_emo_probs = torch.sigmoid(graph_emo_logits)
        
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
            # Pool the spatial dimension (dim=2) of z_img for global representation
            z_img_global = z_img.mean(dim=2)
            l_ib, l_align, l_decorr = self.hsic_loss(au_embeddings, z_img_global, au_labels)
            losses["loss_ib"] = l_ib
            losses["loss_align"] = l_align
            losses["loss_decorr"] = l_decorr
            
            # --- AU-AU graph loss ---
            loss_au_au = self.au_bce_loss(au_au_logits, au_labels)
            losses["loss_au_au"] = loss_au_au
            
            # --- Contrastive loss (text-visual alignment) ---
            text_emb = self._get_text_embeddings()  # (N_AU, text_dim)
            
            # Average AU embeddings over batch for contrastive alignment
            au_emb_mean = torch.stack([e.mean(dim=0) for e in au_embeddings])  # (N_AU, D)
            
            visual_proj = self.visual_proj(au_emb_mean)   # (N_AU, shared_dim)
            text_proj = self.text_proj(text_emb)           # (N_AU, shared_dim)
            
            loss_contrastive = self.contrastive_loss(visual_proj, text_proj)
            losses["loss_contrastive"] = loss_contrastive
            
            # --- Emotion Contrastive loss (text-visual alignment) ---
            emo_text_emb = self._get_emotion_text_embeddings() # (N_EMO, text_dim)
            emo_emb_mean = emotion_emb_stacked.mean(dim=0)     # (N_EMO, D)
            
            emo_visual_proj = self.emotion_visual_proj(emo_emb_mean) # (N_EMO, shared_dim)
            emo_text_proj = self.emotion_text_proj(emo_text_emb)     # (N_EMO, shared_dim)
            
            loss_emo_contrastive = self.contrastive_loss(emo_visual_proj, emo_text_proj)
            losses["loss_emo_contrastive"] = loss_emo_contrastive
            
            # --- DAG constraint on AU-AU graph ---
            loss_dag = self.dag_loss(au_au_adj)
            losses["loss_dag"] = loss_dag
            
            # --- Violation loss (Symmetrical Causal + FACS for both graphs) ---
            # 1. AU-AU Causal Discovery Violation (grounded on real co-activations)
            causal_au_target = au_labels.float() if au_labels is not None else graph_au_probs
            loss_causal_au = self.violation_loss(
                causal_au_target, au_au_adj, au_au_pol_mask, au_au_imp_mask
            )
            losses["loss_causal_au"] = loss_causal_au
            
            # 2. AU-AU FACS Violation (Dual-Stage: guides CNN in Phase 1 & 3, Graph in Phase 2 & 3)
            loss_facs_cnn = self.facs_au_violation_loss(au_probs)
            loss_facs_graph = self.facs_au_violation_loss(graph_au_probs)
            loss_facs_au = loss_facs_cnn + loss_facs_graph
            losses["loss_facs_au"] = loss_facs_au
            
            # 3. AU-Expression Causal Discovery Violation
            causal_exp_target = torch.cat([causal_au_target, emotion_pseudo], dim=1)
            loss_causal_exp = self.violation_loss(
                causal_exp_target, au_exp_adj, au_exp_pol_mask, au_exp_imp_mask
            )
            losses["loss_causal_exp"] = loss_causal_exp
            
            # 4. AU-Expression FACS Hypergraph Violation
            loss_facs_exp = self.facs_emotion_violation_loss(graph_au_probs, graph_emo_probs)
            losses["loss_facs_exp"] = loss_facs_exp
            
            # --- Emotion weak supervision ---
            # Use fuzzy pseudo-labels as targets for emotion head
            loss_emotion = self.emotion_bce_loss(emotion_probs, emotion_pseudo.detach())
            losses["loss_emotion"] = loss_emotion
            
            # --- Graph prediction losses ---
            loss_graph_au = self.au_bce_loss(graph_au_logits, au_labels)
            losses["loss_graph_au"] = loss_graph_au
            
            loss_graph_emo = self.emotion_bce_loss(graph_emo_probs, emotion_pseudo.detach())
            losses["loss_graph_emo"] = loss_graph_emo
            
            # --- Counterfactual intervention (HiMod Dynamic Sparsity-Aware Perturbation) ---
            # Derive mask from graph's learned au_au_imp_mask (diagonal is self-importance)
            cf_importance = au_au_imp_mask.diagonal()
            cf_mask = (cf_importance > self.cf_threshold).float()  # (N_AU,)
            
            # HiMod: Scale perturbation variance adaptively based on AU class sparsity
            # Rare AUs receive stronger exploration noise to prevent identity memorization
            sparsity_scale = (self.au_pos_weights / self.au_pos_weights.mean()).view(1, -1, 1)
            
            # Perturb the AU node embeddings (graph input)
            noise_imp = torch.randn_like(au_emb_stacked) * (self.cfg.noise_std * sparsity_scale)
            au_embeddings_imp = au_emb_stacked + noise_imp * cf_mask.view(1, -1, 1)
            
            noise_unimp = torch.randn_like(au_emb_stacked) * (self.cfg.noise_std * sparsity_scale)
            au_embeddings_unimp = au_emb_stacked + noise_unimp * (1.0 - cf_mask.view(1, -1, 1))
            
            # Forward perturbed embeddings through AU-Exp graph to get new predictions
            updated_nodes_imp, _ = self.graph_module.forward_au_exp(au_embeddings_imp, emotion_emb_stacked)
            updated_au_imp = updated_nodes_imp[:, :NUM_AUS, :]
            
            graph_au_logits_imp = []
            for i in range(NUM_AUS):
                graph_au_logits_imp.append(self.graph_au_classifiers[i](updated_au_imp[:, i, :]))
            graph_au_probs_imp = torch.sigmoid(torch.cat(graph_au_logits_imp, dim=1))
            
            updated_nodes_unimp, _ = self.graph_module.forward_au_exp(au_embeddings_unimp, emotion_emb_stacked)
            updated_au_unimp = updated_nodes_unimp[:, :NUM_AUS, :]
            
            graph_au_logits_unimp = []
            for i in range(NUM_AUS):
                graph_au_logits_unimp.append(self.graph_au_classifiers[i](updated_au_unimp[:, i, :]))
            graph_au_probs_unimp = torch.sigmoid(torch.cat(graph_au_logits_unimp, dim=1))
            
            # Compute CF loss on the graph predictions
            loss_cf_imp, loss_cf_unimp = self.cf_loss(
                graph_au_probs.detach(), graph_au_probs_imp, graph_au_probs_unimp
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
                cfg.lambda_emo_contrastive * loss_emo_contrastive +
                cfg.lambda_dag * loss_dag +
                cfg.lambda_causal_au * loss_causal_au +
                cfg.lambda_causal_exp * loss_causal_exp +
                cfg.lambda_facs_au * loss_facs_au +
                cfg.lambda_facs_exp * loss_facs_exp +
                cfg.lambda_emotion * loss_emotion +
                cfg.lambda_au_au * loss_au_au +
                cfg.lambda_graph_au * loss_graph_au +
                cfg.lambda_graph_emo * loss_graph_emo +
                cfg.lambda_cf_important * loss_cf_imp +
                cfg.lambda_cf_unimportant * loss_cf_unimp
            )
            losses["total_loss"] = total_loss
        
        return {
            "au_embeddings": au_embeddings,
            "au_logits": au_logits,
            "au_probs": au_probs,
            "au_au_probs": au_au_probs,
            "emotion_embed": emotion_emb_stacked,
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "emotion_pseudo": emotion_pseudo,
            "graph_au_probs": graph_au_probs,
            "graph_emo_probs": graph_emo_probs,
            "au_au_adj": au_au_adj,
            "au_exp_adj": au_exp_adj,
            "au_au_importance_mask": au_au_imp_mask,
            "au_au_polarity_mask": au_au_pol_mask,
            "au_exp_importance_mask": au_exp_imp_mask,
            "au_exp_polarity_mask": au_exp_pol_mask,
            "z_img": z_img,
            "losses": losses,
        }
