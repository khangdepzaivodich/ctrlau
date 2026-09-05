"""
SymGraphAU: Architecture integrating MultiviewSymAU Phase 1 (Stage 1)
with Graph Relational Reasoning (Stage 2).
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
    FACSAUViolationLoss, WeightedAsymmetricLoss, ExpressionBCELoss
)
from .backbone import VisualBackbone
from .text_encoder import TextEncoder
from .heads import AUHead, Conv1DExtractor
from .gat import AUGraphModule
from .masks import MaskModule


# ============================================================
# MultiviewSymAU Definitions for Phase 1
# ============================================================
SYM_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]
NUM_SYM_EMOTIONS = len(SYM_EMOTIONS)  # 7 emotions in MultiviewSymAU

# M_AE matrix from MultiviewSymAU matrixMAE/M_AE_DISFA.npy (8 AUs x 7 Emotions)
# Rows: AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26
# Cols: Angry, Fear, Happy, Sad, Surprise, Disgust, Neutral
M_AE_DISFA = torch.tensor([
    [0.5, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1],  # AU1
    [0.5, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1],  # AU2
    [0.9, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1],  # AU4
    [0.1, 0.5, 0.9, 0.5, 0.5, 0.1, 0.1],  # AU6
    [0.5, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],  # AU9
    [0.1, 0.1, 0.9, 0.1, 0.5, 0.5, 0.1],  # AU12
    [0.5, 0.9, 0.9, 0.5, 0.9, 0.5, 0.1],  # AU25
    [0.1, 0.5, 0.5, 0.1, 0.9, 0.1, 0.1],  # AU26
], dtype=torch.float32)


class SymExprHead(nn.Module):
    """
    Expression Head from MultiviewSymAU Stage 1 (JFL).
    7 independent 1D CNN branches producing per-emotion embeddings and probabilities.
    Emotions: [Angry, Fear, Happy, Sad, Surprise, Disgust, Neutral]
    """
    def __init__(self, backbone_dim, num_emotions=NUM_SYM_EMOTIONS, emotion_embed_dim=256):
        super().__init__()
        self.num_emotions = num_emotions
        self.embed_dim = emotion_embed_dim
        
        self.extractors = nn.ModuleList([
            Conv1DExtractor(backbone_dim, emotion_embed_dim)
            for _ in range(num_emotions)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(emotion_embed_dim, 1)
            for _ in range(num_emotions)
        ])
    
    def forward(self, z_img):
        """
        Args:
            z_img: (B, 2048, 49) image patch features
        Returns:
            emb_list: list of 7 tensors (B, emotion_embed_dim)
            logits: (B, 7)
            probs: (B, 7)
        """
        emb_list = []
        logit_list = []
        for i in range(self.num_emotions):
            emb = self.extractors[i](z_img)
            logit = self.classifiers[i](emb)
            emb_list.append(emb)
            logit_list.append(logit)
        logits = torch.cat(logit_list, dim=1)
        probs = torch.sigmoid(logits)
        return emb_list, logits, probs


class SymGraphAUModel(nn.Module):
    """
    SymGraphAU architecture:
    
    Phase 1 (Identical to MultiviewSymAU Stage 1):
    1. Image -> ResNet-50 -> z_img (2048-dim, 49 patches)
    2. z_img -> AU Head (8 branches) -> V_a (embeddings) + p_a (AU probabilities)
    3. z_img -> SymExprHead (7 branches) -> V_e (embeddings) + p_e (Emotion probabilities)
    4. Y_a -> au_to_expr_pseudo(M_AE) -> Y_e (one-hot expression target with Neutral=6)
    5. Loss L_wa = WeightedAsymmetricLoss(p_a, Y_a)
    6. Loss L_we = ExpressionBCELoss(p_e, Y_e)
    7. Phase 1 Total Loss = L_wa + 0.05 * L_we
    
    Phase 2 (Stage 2 Relational Reasoning):
    - GAT on AU-AU graph and AU-Expression graph
    - Causal DAG & FACS Violation Loss
    - Counterfactual Intervention
    
    Phase 3: End-to-End Joint Fine-Tuning
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        self.phase = getattr(cfg, "phase", 1)
        self.num_emotions = NUM_SYM_EMOTIONS
        
        # Prior AU-Expression Matrix from MultiviewSymAU
        self.register_buffer("M_AE", M_AE_DISFA)
        
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
        
        # 7-class Emotion Head (matches MultiviewSymAU)
        self.emotion_head = SymExprHead(
            backbone_dim=cfg.backbone_feat_dim,
            num_emotions=self.num_emotions,
            emotion_embed_dim=cfg.emotion_embed_dim,
        )
        
        self.graph_module = AUGraphModule(
            embed_dim=cfg.au_embed_dim,
            num_aus=NUM_AUS,
            num_emotions=self.num_emotions,
            gat_hidden_dim=cfg.gat_hidden_dim,
            gat_num_heads=cfg.gat_num_heads,
            gat_num_layers=cfg.gat_num_layers,
            dropout=cfg.gat_dropout,
        )
        
        self.mask_module = MaskModule(
            num_aus=NUM_AUS,
            num_emotions=self.num_emotions,
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
        self.au_au_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(NUM_AUS)
        ])
        self.graph_au_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(NUM_AUS)
        ])
        self.graph_emo_classifiers = nn.ModuleList([
            nn.Linear(cfg.au_embed_dim, 1) for _ in range(self.num_emotions)
        ])
        
        # ---- MultiviewSymAU Stage 1 Losses ----
        raw_au_pos_weights = torch.tensor([
            19.11, 22.18, 5.56, 11.67, 22.90, 6.76, 2.61, 10.34
        ])
        wal_weights = raw_au_pos_weights / raw_au_pos_weights.sum() * NUM_AUS
        self.register_buffer("wal_weights", wal_weights)
        self.register_buffer("au_pos_weights", torch.sqrt(raw_au_pos_weights))
        
        # (1) L_wa: Weighted Asymmetric Loss on AU probabilities
        self.wal_loss = WeightedAsymmetricLoss(weight=self.wal_weights)
        self.au_bce_loss = FocalLoss(gamma=2.0, pos_weight=torch.sqrt(raw_au_pos_weights))
        
        # (2) L_we: Expression BCE Loss on Emotion probabilities
        self.expression_bce_loss = ExpressionBCELoss()
        self.emotion_bce_loss = nn.BCELoss()
        
        # ---- Relational / Graph Loss modules ----
        self.hsic_loss = HSICDisentanglementLoss(num_aus=NUM_AUS)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.dag_loss = DAGLoss(num_nodes=NUM_AUS)
        self.violation_loss = ViolationLoss()
        self.facs_emotion_violation_loss = FACSEmotionViolationLoss()
        self.facs_au_violation_loss = FACSAUViolationLoss()
        self.cf_loss = CounterfactualLoss()
        
        # ---- Counterfactual mask threshold ----
        self.cf_threshold = nn.Parameter(torch.tensor(0.5))
        
        # ---- Precompute text embeddings ----
        self._text_descriptions = [AU_DESCRIPTIONS[au] for au in DISFA_AUS]
        self._cached_text_emb = None
        self._emotion_text_descriptions = [EMOTION_DESCRIPTIONS[emo] for emo in EMOTIONS]
        self._cached_emotion_text_emb = None
        
        # Initialize default phase freeze/unfreeze
        self.set_phase(self.phase)
    
    def set_phase(self, phase: int):
        """Configure trainable modules and loss weights for Phase 1, 2, or 3."""
        self.phase = phase
        if phase == 1:
            # Phase 1: Feature Extraction (MultiviewSymAU Stage 1)
            # Only train Backbone and CNN heads
            for param in self.parameters():
                param.requires_grad = True
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.graph_module.parameters():
                param.requires_grad = False
            for param in self.mask_module.parameters():
                param.requires_grad = False
            for param in self.visual_proj.parameters():
                param.requires_grad = False
            for param in self.text_proj.parameters():
                param.requires_grad = False
            for param in self.emotion_visual_proj.parameters():
                param.requires_grad = False
            for param in self.emotion_text_proj.parameters():
                param.requires_grad = False
            for param in self.au_au_classifiers.parameters():
                param.requires_grad = False
            for param in self.graph_au_classifiers.parameters():
                param.requires_grad = False
            for param in self.graph_emo_classifiers.parameters():
                param.requires_grad = False
        elif phase == 2:
            # Phase 2: Graph Learning
            # Freeze Backbone and CNN heads, train Graph reasoning
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.au_head.parameters():
                param.requires_grad = False
            for param in self.emotion_head.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.visual_proj.parameters():
                param.requires_grad = False
            for param in self.text_proj.parameters():
                param.requires_grad = False
            for param in self.emotion_visual_proj.parameters():
                param.requires_grad = False
            for param in self.emotion_text_proj.parameters():
                param.requires_grad = False
            for param in self.graph_module.parameters():
                param.requires_grad = True
            for param in self.mask_module.parameters():
                param.requires_grad = True
            for param in self.au_au_classifiers.parameters():
                param.requires_grad = True
            for param in self.graph_au_classifiers.parameters():
                param.requires_grad = True
            for param in self.graph_emo_classifiers.parameters():
                param.requires_grad = True
        elif phase == 3:
            # Phase 3: End-to-end fine tuning
            for param in self.parameters():
                param.requires_grad = True
            for param in self.text_encoder.parameters():
                param.requires_grad = False

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
    
    def au_to_expr_pseudo(self, Y_a):
        """
        MultiviewSymAU Eq. (1) & (2):
        1. scores = Y_a @ M_AE
        2. ke = argmax(scores)
        3. neutral samples (sum(Y_a) == 0) -> class 6 (Neutral)
        4. One-hot scatter
        """
        scores = Y_a.float() @ self.M_AE  # (B, 7)
        ke = scores.argmax(dim=1)         # (B,)
        neutral_mask = (Y_a.float().sum(dim=1) == 0)
        ke[neutral_mask] = 6              # Neutral index = 6
        Y_e = torch.zeros(Y_a.size(0), self.num_emotions, device=Y_a.device)
        Y_e.scatter_(1, ke.unsqueeze(1), 1.0)
        return Y_e

    def forward(self, images, au_labels=None, phase=None):
        """
        Forward pass supporting Phase 1 (Stage 1 MultiviewSymAU), Phase 2, and Phase 3.
        
        Args:
            images: (B, 3, H, W) input images
            au_labels: (B, N_AU) binary AU ground-truth labels
            phase: optional int (1, 2, or 3). Defaults to self.phase (1).
        """
        if phase is None:
            phase = getattr(self.cfg, "phase", None) or self.phase
            
        device = images.device
        B = images.size(0)
        
        # ============================================================
        # 1. Visual Backbone (ResNet-50)
        # ============================================================
        z_img = self.backbone(images)  # (B, 2048, 49)
        
        # ============================================================
        # 2. AU Head (8 branches -> V_a and p_a)
        # ============================================================
        au_embeddings, au_logits, au_probs = self.au_head(z_img)
        # au_embeddings: list of 8 tensors, each (B, 256)
        # au_logits: (B, 8)
        # au_probs: (B, 8)
        
        # ============================================================
        # 3. Expression Head (7 branches -> V_e and p_e)
        # ============================================================
        emotion_embed_list, emotion_logits, emotion_probs = self.emotion_head(z_img)
        # emotion_embed_list: list of 7 tensors, each (B, 256)
        # emotion_logits: (B, 7)
        # emotion_probs: (B, 7)
        
        # ============================================================
        # Phase 1 Losses: Identical to MultiviewSymAU Stage 1
        # ============================================================
        losses = {}
        emotion_pseudo = None
        
        if au_labels is not None:
            # Pseudo-label expression via M_AE dot product (MultiviewSymAU Eq. 1 & 2)
            emotion_pseudo = self.au_to_expr_pseudo(au_labels)  # (B, 7) one-hot
            
            # (1) L_wa: Weighted Asymmetric Loss on AU probabilities (Eq. 3)
            loss_wa = self.wal_loss(au_probs, au_labels.float())
            losses["loss_wa"] = loss_wa
            losses["loss_au"] = loss_wa
            
            # (2) L_we: Expression BCE Loss on Emotion probabilities (Eq. 4)
            loss_we = self.expression_bce_loss(emotion_probs, emotion_pseudo)
            losses["loss_we"] = loss_we
            losses["loss_emotion"] = loss_we
            
            # (3) Joint Feature Learning (JFL) Phase 1 Loss: L_jf = L_wa + gamma * L_we
            # (Section 4.1 in MultiviewSymAU: gamma = 0.05)
            loss_phase1 = loss_wa + 0.05 * loss_we
            losses["loss_phase1"] = loss_phase1
            
            if phase == 1:
                losses["total_loss"] = loss_phase1
        else:
            if phase == 1:
                # Inference during Phase 1
                emotion_pseudo = self.au_to_expr_pseudo((au_probs > 0.5).float())

        # ============================================================
        # Early return for Phase 1: Pure backbone & CNN testing
        # ============================================================
        if phase == 1:
            return {
                "au_embeddings": au_embeddings,
                "au_logits": au_logits,
                "au_probs": au_probs,
                "emotion_embed": emotion_embed_list,
                "emotion_logits": emotion_logits,
                "emotion_probs": emotion_probs,
                "emotion_pseudo": emotion_pseudo,
                "z_img": z_img,
                "losses": losses,
            }

        # ============================================================
        # Phase 2 & 3: Graph Relational Reasoning & Interventions
        # ============================================================
        au_emb_stacked = torch.stack(au_embeddings, dim=1)           # (B, N_AU, D)
        emotion_emb_stacked = torch.stack(emotion_embed_list, dim=1) # (B, N_EMO, D)
        
        # AU-AU graph
        updated_au, au_au_adj = self.graph_module.forward_au_au(au_emb_stacked)
        au_au_logits = []
        for i in range(NUM_AUS):
            au_au_logits.append(self.au_au_classifiers[i](updated_au[:, i, :]))
        au_au_logits = torch.cat(au_au_logits, dim=1)
        au_au_probs = torch.sigmoid(au_au_logits)
        
        # AU-Expression graph
        updated_nodes, au_exp_adj = self.graph_module.forward_au_exp(au_emb_stacked, emotion_emb_stacked)
        updated_au_final = updated_nodes[:, :NUM_AUS, :]
        updated_emo_final = updated_nodes[:, NUM_AUS:, :]
        
        graph_au_logits = []
        for i in range(NUM_AUS):
            graph_au_logits.append(self.graph_au_classifiers[i](updated_au_final[:, i, :]))
        graph_au_logits = torch.cat(graph_au_logits, dim=1)
        graph_au_probs = torch.sigmoid(graph_au_logits)
        
        graph_emo_logits = []
        for i in range(self.num_emotions):
            graph_emo_logits.append(self.graph_emo_classifiers[i](updated_emo_final[:, i, :]))
        graph_emo_logits = torch.cat(graph_emo_logits, dim=1)
        graph_emo_probs = torch.sigmoid(graph_emo_logits)
        
        # Graph masks
        au_au_imp_mask, au_au_pol_mask = self.mask_module.forward_au_au(au_au_adj)
        au_exp_imp_mask, au_exp_pol_mask = self.mask_module.forward_au_exp(au_exp_adj)
        
        # Losses for Phase 2 / Phase 3
        if au_labels is not None:
            # HSIC Disentanglement
            z_img_global = z_img.mean(dim=2)
            l_ib, l_align, l_decorr = self.hsic_loss(au_embeddings, z_img_global, au_labels)
            losses["loss_ib"] = l_ib
            losses["loss_align"] = l_align
            losses["loss_decorr"] = l_decorr
            
            # AU-AU graph loss
            loss_au_au = self.au_bce_loss(au_au_logits, au_labels)
            losses["loss_au_au"] = loss_au_au
            
            # Contrastive text-visual alignment
            text_emb = self._get_text_embeddings()
            au_emb_mean = torch.stack([e.mean(dim=0) for e in au_embeddings])
            visual_proj = self.visual_proj(au_emb_mean)
            text_proj = self.text_proj(text_emb)
            loss_contrastive = self.contrastive_loss(visual_proj, text_proj)
            losses["loss_contrastive"] = loss_contrastive
            
            # Emotion contrastive
            emo_text_emb = self._get_emotion_text_embeddings()
            emo_emb_mean = emotion_emb_stacked[:, :len(self._emotion_text_descriptions)].mean(dim=0)
            emo_visual_proj = self.emotion_visual_proj(emo_emb_mean)
            emo_text_proj = self.emotion_text_proj(emo_text_emb)
            loss_emo_contrastive = self.contrastive_loss(emo_visual_proj, emo_text_proj)
            losses["loss_emo_contrastive"] = loss_emo_contrastive
            
            # DAG loss
            loss_dag = self.dag_loss(au_au_adj)
            losses["loss_dag"] = loss_dag
            
            # Causal & FACS Violation losses
            causal_au_target = au_labels.float()
            loss_causal_au = self.violation_loss(
                causal_au_target, au_au_adj, au_au_pol_mask, au_au_imp_mask
            )
            losses["loss_causal_au"] = loss_causal_au
            
            loss_facs_cnn = self.facs_au_violation_loss(au_probs)
            loss_facs_graph = self.facs_au_violation_loss(graph_au_probs)
            loss_facs_au = loss_facs_cnn + loss_facs_graph
            losses["loss_facs_au"] = loss_facs_au
            
            causal_exp_target = torch.cat([causal_au_target, emotion_pseudo], dim=1)
            loss_causal_exp = self.violation_loss(
                causal_exp_target, au_exp_adj, au_exp_pol_mask, au_exp_imp_mask
            )
            losses["loss_causal_exp"] = loss_causal_exp
            
            loss_facs_exp = self.facs_emotion_violation_loss(
                graph_au_probs, graph_emo_probs[:, :NUM_EMOTIONS]
            )
            losses["loss_facs_exp"] = loss_facs_exp
            
            # Graph prediction losses
            loss_graph_au = self.au_bce_loss(graph_au_logits, au_labels)
            losses["loss_graph_au"] = loss_graph_au
            
            loss_graph_emo = self.expression_bce_loss(graph_emo_probs, emotion_pseudo)
            losses["loss_graph_emo"] = loss_graph_emo
            
            # HiMod Counterfactual Intervention
            cf_importance = au_au_imp_mask.diagonal()
            cf_mask = (cf_importance > self.cf_threshold).float()
            sparsity_scale = (self.au_pos_weights / self.au_pos_weights.mean()).view(1, -1, 1)
            
            noise_imp = torch.randn_like(au_emb_stacked) * (self.cfg.noise_std * sparsity_scale)
            au_embeddings_imp = au_emb_stacked + noise_imp * cf_mask.view(1, -1, 1)
            
            noise_unimp = torch.randn_like(au_emb_stacked) * (self.cfg.noise_std * sparsity_scale)
            au_embeddings_unimp = au_emb_stacked + noise_unimp * (1.0 - cf_mask.view(1, -1, 1))
            
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
            
            loss_cf_imp, loss_cf_unimp = self.cf_loss(
                graph_au_probs.detach(), graph_au_probs_imp, graph_au_probs_unimp
            )
            losses["loss_cf_important"] = loss_cf_imp
            losses["loss_cf_unimportant"] = loss_cf_unimp
            
            # Combined Loss for Phase 2 / Phase 3
            cfg = self.cfg
            gamma_emo = 0.05 if cfg.lambda_emotion == 0.1 else cfg.lambda_emotion
            total_loss = (
                cfg.lambda_au * loss_wa +
                gamma_emo * loss_we +
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


# Backward-compatible alias
CtrlAUModel = SymGraphAUModel
