"""
SymGraphAU: Architecture integrating MultiviewSymAU Phase 1 (Stage 1)
with Graph Relational Reasoning (Stage 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
from .gat import AUGraphModule
from .masks import MaskModule


# ============================================================
# MultiviewSymAU Definitions for Phase 1
# ============================================================
SYM_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]
NUM_SYM_EMOTIONS = len(SYM_EMOTIONS)  # 7 emotions in MultiviewSymAU
EMB_DIM = 256

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


# ============================================================
# MultiviewSymAU Stage 1 Head Architecture (Exact Replication)
# ============================================================

class LinearBlock(nn.Module):
    """
    Linear projection block mapping backbone channels (2048 -> 512).
    Matches MultiviewSymAU/model/basic_block.py LinearBlock.
    """
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2.0 / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        # x: (B, D, C_in) e.g. (B, 49, 2048)
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)            # (B, C_mid, D)
        x = self.relu(self.bn(x)).permute(0, 2, 1) # (B, D, C_mid)
        return x


class Conv1DExtractor(nn.Module):
    """
    1D Convolutional extractor for a single AU / Expression branch.
    Matches MultiviewSymAU/model/SymStage1.py Conv1DExtractor.
    Input: (B, D, C_in) e.g. (B, 49, 512)
    Output: (B, C_emb=256)
    """
    def __init__(self, in_channels: int = 512, hid_channels: int = 512, emb_channels: int = EMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hid_channels)
        self.conv2 = nn.Conv1d(hid_channels, emb_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(emb_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        # (B, D, C_in) -> (B, C_in, D) for Conv1d
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        emb = x.mean(dim=-1)  # Global average pool across spatial patches D
        return emb


class SymAUHead(nn.Module):
    """
    Stage-1 AU Head from MultiviewSymAU.
    Has N_a independent Conv1DExtractor branches + linear classifiers.
    """
    def __init__(self, in_channels: int = 512, num_aus: int = NUM_AUS, hid_channels: int = 512, emb_channels: int = EMB_DIM):
        super().__init__()
        self.num_aus = num_aus
        self.extractors = nn.ModuleList([
            Conv1DExtractor(in_channels, hid_channels, emb_channels)
            for _ in range(num_aus)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(emb_channels, 1)
            for _ in range(num_aus)
        ])

    def forward(self, feat):
        """
        feat: (B, D, C_mid) e.g. (B, 49, 512)
        Returns:
            emb_list: list of N_a tensors (B, 256)
            logits: (B, N_a)
            probs: (B, N_a)
        """
        emb_list = []
        logit_list = []
        for i in range(self.num_aus):
            emb = self.extractors[i](feat)
            logit = self.classifiers[i](emb)
            emb_list.append(emb)
            logit_list.append(logit)
        logits = torch.cat(logit_list, dim=1)
        probs = torch.sigmoid(logits)
        return emb_list, logits, probs


class SymExprHead(nn.Module):
    """
    Stage-1 Expression Head from MultiviewSymAU.
    Has N_e independent Conv1DExtractor branches + linear classifiers.
    """
    def __init__(self, in_channels: int = 512, num_expr: int = NUM_SYM_EMOTIONS, hid_channels: int = 512, emb_channels: int = EMB_DIM):
        super().__init__()
        self.num_expr = num_expr
        self.extractors = nn.ModuleList([
            Conv1DExtractor(in_channels, hid_channels, emb_channels)
            for _ in range(num_expr)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(emb_channels, 1)
            for _ in range(num_expr)
        ])

    def forward(self, feat):
        """
        feat: (B, D, C_mid) e.g. (B, 49, 512)
        Returns:
            emb_list: list of N_e tensors (B, 256)
            logits: (B, N_e)
            probs: (B, N_e)
        """
        emb_list = []
        logit_list = []
        for i in range(self.num_expr):
            emb = self.extractors[i](feat)
            logit = self.classifiers[i](emb)
            emb_list.append(emb)
            logit_list.append(logit)
        logits = torch.cat(logit_list, dim=1)
        probs = torch.sigmoid(logits)
        return emb_list, logits, probs


# ============================================================
# SymGraphAU Full Model
# ============================================================

class SymGraphAUModel(nn.Module):
    """
    SymGraphAU architecture:
    
    Phase 1 (Identical to MultiviewSymAU Stage 1):
    1. Image -> ResNet-50 -> z_img (2048-dim, 49 patches)
    2. z_img -> LinearBlock (2048 -> 512) -> feat (49, 512)
    3. feat -> SymAUHead (8 branches) -> V_a (embeddings) + p_a (AU probabilities)
    4. feat -> SymExprHead (7 branches) -> V_e (embeddings) + p_e (Emotion probabilities)
    5. Y_a -> au_to_expr_pseudo(M_AE) -> Y_e (one-hot expression target with Neutral=6)
    6. Loss L_wa = WeightedAsymmetricLoss(p_a, Y_a)
    7. Loss L_we = ExpressionBCELoss(p_e, Y_e)
    8. Phase 1 Total Loss = L_wa + 0.05 * L_we
    
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
        
        # MultiviewSymAU Stage 1 Shared Linear Projection (2048 -> 512)
        mid_channels = cfg.backbone_feat_dim // 4  # 2048 // 4 = 512
        self.global_linear = LinearBlock(
            in_features=cfg.backbone_feat_dim,
            out_features=mid_channels,
        )
        
        self.text_encoder = TextEncoder(
            clip_model_name=cfg.clip_model_name,
        )
        
        # MultiviewSymAU Stage 1 8-branch AU Head
        self.au_head = SymAUHead(
            in_channels=mid_channels,
            num_aus=NUM_AUS,
            hid_channels=mid_channels,
            emb_channels=EMB_DIM,
        )
        
        # MultiviewSymAU Stage 1 7-branch Emotion Head
        self.emotion_head = SymExprHead(
            in_channels=mid_channels,
            num_expr=self.num_emotions,
            hid_channels=mid_channels,
            emb_channels=EMB_DIM,
        )
        
        self.graph_module = AUGraphModule(
            embed_dim=EMB_DIM,
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
            nn.Linear(EMB_DIM, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.embed_dim, cfg.shared_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.shared_embed_dim, cfg.shared_embed_dim),
        )
        
        self.emotion_visual_proj = nn.Sequential(
            nn.Linear(EMB_DIM, cfg.shared_embed_dim),
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
            nn.Linear(EMB_DIM, 1) for _ in range(NUM_AUS)
        ])
        self.graph_au_classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(NUM_AUS)
        ])
        self.graph_emo_classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(self.num_emotions)
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
            for param in self.global_linear.parameters():
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

    def update_class_weights(self, weights: torch.Tensor):
        """Update AU class weights dynamically (e.g. for specific folds)."""
        w = weights / weights.sum() * NUM_AUS
        self.wal_weights.copy_(w)
        self.wal_loss.weight = self.wal_weights

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
        # 1. Visual Backbone (ResNet-50) -> (B, 49, 2048)
        # ============================================================
        z_img = self.backbone(images)  # (B, 2048, 49)
        feat = z_img.permute(0, 2, 1)  # (B, 49, 2048)
        
        # ============================================================
        # 2. Global Linear (LinearBlock 2048 -> 512)
        # ============================================================
        feat = self.global_linear(feat)  # (B, 49, 512)
        
        # ============================================================
        # 3. AU Head (8 branches -> V_a and p_a)
        # ============================================================
        au_embeddings, au_logits, au_probs = self.au_head(feat)
        # au_embeddings: list of 8 tensors, each (B, 256)
        # au_logits: (B, 8)
        # au_probs: (B, 8)
        
        # ============================================================
        # 4. Expression Head (7 branches -> V_e and p_e)
        # ============================================================
        emotion_embed_list, emotion_logits, emotion_probs = self.emotion_head(feat)
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
