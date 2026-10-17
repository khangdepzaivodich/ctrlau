"""
CtrlAU: Full Model Architecture.
Integrates MultiviewSymAU Stage 1 Backbone (ResNet-50 + LinearBlock + SymAUHead + SymExprHead + M_AE matrix)
with CtrlAU Relational & Causal Regularizations in Phase 1, Phase 2, and Phase 3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AU_DESCRIPTIONS, ModelConfig, CTRL_EMOTIONS, CTRL_NUM_EMOTIONS,
    CTRL_EMOTION_DESCRIPTIONS, get_dataset_aus,
    get_dataset_au_expression_prior, prior_support_matrix,
    compute_expression_pseudo_labels,
)
from losses import (
    HSICDisentanglementLoss, ContrastiveLoss, DAGLoss,
    ViolationLoss, CounterfactualLoss,
    FACSAUViolationLoss, WeightedAsymmetricLoss, ExpressionBCELoss
)
from .text_encoder import TextEncoder
from .gat import AUGraphModule
from .masks import MaskModule
from .symgraphau import (
    LinearBlock, SymAUHead, SymExprHead,
    EMB_DIM,
    resnet50
)


class CtrlAUModel(nn.Module):
    """
    CtrlAU Architecture combining the MultiviewSymAU Stage 1 backbone with CtrlAU regularizations:
    
    Phase 1 (Feature Extraction with Combined Regularizations):
    1. Image -> ResNet-50 -> z_img (2048-dim, 49 spatial patches)
    2. z_img -> LinearBlock (2048 -> 512) -> feat (49, 512)
    3. feat -> SymAUHead (dataset AU branches) -> V_a + p_a
    4. feat -> SymExprHead (8 expression branches) -> V_e + p_e
    5. Y_a -> published M_A-E prior -> one-hot expression target
    6. Loss L_wa = WeightedAsymmetricLoss(p_a, Y_a)
    7. Loss L_we = ExpressionBCELoss(p_e, Y_e)
    8. CtrlAU Regularizers:
       - HSIC Disentanglement: L_ib, L_align, L_decorr
       - CLIP Text-Visual Contrastive: L_contrastive (AUs) + L_emo_contrastive (Emotions)
       - No FACS loss (introduced only after graph construction)
    9. Phase 1 Total Loss = L_wa + 0.05 * L_we + CtrlAU regularizers
    
    Phase 2 (Graph Learning & Relational Reasoning):
    - GAT on AU-AU graph and AU-Expression graph
    - Causal DAG & FACS Violation Loss
    - No counterfactual passes
    
    Phase 3: Counterfactual training with the visual model frozen
    Phase 4: Optional end-to-end refinement
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        self.phase = cfg.phase
        self.dataset_name = cfg.dataset_name.lower()
        self.au_ids = tuple(cfg.au_ids)
        verified_au_ids = get_dataset_aus(self.dataset_name)
        if self.au_ids != verified_au_ids:
            raise ValueError(
                f"cfg.au_ids {self.au_ids} do not match verified {self.dataset_name} AU order {verified_au_ids}"
            )
        self.num_aus = len(self.au_ids)
        self.emotion_names = tuple(CTRL_EMOTIONS)
        self.num_emotions = CTRL_NUM_EMOTIONS
        self.au_expression_prior = get_dataset_au_expression_prior(self.dataset_name)
        self.au_expression_support = prior_support_matrix(self.au_expression_prior)
        supplied_prior = torch.as_tensor(cfg.au_expression_prior, dtype=torch.float32)
        if not torch.equal(supplied_prior, self.au_expression_prior):
            raise ValueError("cfg.au_expression_prior does not match the verified dataset prior")
        self.register_buffer("paper_au_expression_prior", self.au_expression_prior)
        
        # MultiviewSymAU Stage 1 ResNet-50 Backbone (ImageNet V1 / resnet50-19c8e357)
        self.backbone = resnet50(pretrained=True)
        
        # MultiviewSymAU Stage 1 Shared Linear Projection (2048 -> 512)
        mid_channels = cfg.backbone_feat_dim // 4  # 2048 // 4 = 512
        self.global_linear = LinearBlock(
            in_features=cfg.backbone_feat_dim,
            out_features=mid_channels,
        )
        
        self.text_encoder = TextEncoder(
            clip_model_name=cfg.clip_model_name,
        )
        
        # MultiviewSymAU Stage 1 AU head; branch count follows the dataset.
        self.au_head = SymAUHead(
            in_channels=mid_channels,
            num_aus=self.num_aus,
            hid_channels=mid_channels,
        )
        
        # Published SymGraphAU expression head (8 classes, including Contempt).
        self.emotion_head = SymExprHead(
            in_channels=mid_channels,
            num_expr=self.num_emotions,
            hid_channels=mid_channels,
        )
        
        self.graph_module = AUGraphModule(
            embed_dim=EMB_DIM,
            num_aus=self.num_aus,
            num_emotions=self.num_emotions,
            gat_hidden_dim=cfg.gat_hidden_dim,
            gat_num_heads=cfg.gat_num_heads,
            gat_num_layers=cfg.gat_num_layers,
            dropout=cfg.gat_dropout,
        )
        
        self.mask_module = MaskModule(
            num_aus=self.num_aus,
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
            nn.Linear(EMB_DIM, 1) for _ in range(self.num_aus)
        ])
        self.graph_au_classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(self.num_aus)
        ])
        self.graph_emo_classifiers = nn.ModuleList([
            nn.Linear(EMB_DIM, 1) for _ in range(self.num_emotions)
        ])
        # Initialize graph classifiers from trained base heads (Phase 1 calibrated boundaries)
        self.init_graph_classifiers_from_base()
        
        # ---- MultiviewSymAU Stage 1 Losses ----
        # Fold-specific class weights are supplied by the dataset in train.py.
        # Until then, unit weights are the mathematically neutral WAL setting.
        wal_weights = torch.ones(self.num_aus)
        self.register_buffer("wal_weights", wal_weights)
        
        # (1) L_wa: Weighted Asymmetric Loss on AU probabilities
        self.wal_loss = WeightedAsymmetricLoss(weight=self.wal_weights)
        
        # (2) L_we: Expression BCE Loss on Emotion probabilities
        self.expression_bce_loss = ExpressionBCELoss()
        self.emotion_bce_loss = nn.BCELoss()
        
        # ---- Relational / Graph Loss modules ----
        self.hsic_loss = HSICDisentanglementLoss(num_aus=self.num_aus)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.dag_loss = DAGLoss(num_nodes=self.num_aus)
        self.violation_loss = ViolationLoss()
        self.facs_au_violation_loss = FACSAUViolationLoss(self.au_ids)
        # ---- Counterfactual Loss modules (Level 1: AU-AU, Level 2: AU-Exp) ----
        self.cf_loss = CounterfactualLoss() # backward compatibility
        self.cf_loss_au = CounterfactualLoss(is_expression=False)
        self.cf_loss_exp = CounterfactualLoss(is_expression=True)
        
        # ---- Counterfactual mask threshold logits (sigmoid bounds them) ----
        self.cf_threshold = nn.Parameter(torch.tensor(float(cfg.threshold_init)))
        self.cf_threshold_exp = nn.Parameter(torch.tensor(float(cfg.threshold_init)))
        
        # ---- Precompute text descriptions ----
        self._text_descriptions = [AU_DESCRIPTIONS[au] for au in self.au_ids]
        self._cached_text_emb = None
        
        self._emotion_text_descriptions = [CTRL_EMOTION_DESCRIPTIONS[emo] for emo in self.emotion_names]
        self._cached_emotion_text_emb = None
        
        # Initialize default phase freeze/unfreeze
        self.set_phase(self.phase)

    def init_graph_classifiers_from_base(self):
        """Copy base classifiers without reinitializing training state."""
        if hasattr(self, "au_head") and hasattr(self.au_head, "classifiers"):
            for i in range(min(len(self.au_head.classifiers), len(self.graph_au_classifiers))):
                self.graph_au_classifiers[i].weight.data.copy_(self.au_head.classifiers[i].weight.data)
                self.graph_au_classifiers[i].bias.data.copy_(self.au_head.classifiers[i].bias.data)
            for i in range(min(len(self.au_head.classifiers), len(self.au_au_classifiers))):
                self.au_au_classifiers[i].weight.data.copy_(self.au_head.classifiers[i].weight.data)
                self.au_au_classifiers[i].bias.data.copy_(self.au_head.classifiers[i].bias.data)
        if hasattr(self, "emotion_head") and hasattr(self.emotion_head, "classifiers"):
            for i in range(min(len(self.emotion_head.classifiers), len(self.graph_emo_classifiers))):
                self.graph_emo_classifiers[i].weight.data.copy_(self.emotion_head.classifiers[i].weight.data)
                self.graph_emo_classifiers[i].bias.data.copy_(self.emotion_head.classifiers[i].bias.data)
    
    def set_phase(self, phase: int):
        """Configure trainable modules for phases 1 through 4."""
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported phase: {phase}")
        self.phase = phase
        self.cfg.phase = phase
        if phase == 1:
            # Phase 1: Feature Extraction
            # Train Backbone, LinearBlock, AUHead, ExprHead, and contrastive projection heads
            for param in self.parameters():
                param.requires_grad = True
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.graph_module.parameters():
                param.requires_grad = False
            for param in self.mask_module.parameters():
                param.requires_grad = False
            for param in self.au_au_classifiers.parameters():
                param.requires_grad = False
            for param in self.graph_au_classifiers.parameters():
                param.requires_grad = False
            for param in self.graph_emo_classifiers.parameters():
                param.requires_grad = False
                
            # Contrastive projection heads active during Phase 1
            for param in self.visual_proj.parameters():
                param.requires_grad = True
            for param in self.text_proj.parameters():
                param.requires_grad = True
            for param in self.emotion_visual_proj.parameters():
                param.requires_grad = True
            for param in self.emotion_text_proj.parameters():
                param.requires_grad = True
        elif phase in (2, 3):
            # Phase 2: Graph Learning & Phase 3: Counterfactual Interventions
            # Freeze Backbone and CNN heads, train Graph reasoning & Causal Interventions
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
                param.requires_grad = phase == 3
            self.cf_threshold.requires_grad_(phase == 3)
            self.cf_threshold_exp.requires_grad_(phase == 3)
            for param in self.au_au_classifiers.parameters():
                param.requires_grad = True
            for param in self.graph_au_classifiers.parameters():
                param.requires_grad = True
            for param in self.graph_emo_classifiers.parameters():
                param.requires_grad = True
        elif phase == 4:
            # Phase 4: End-to-end fine tuning
            for param in self.parameters():
                param.requires_grad = True
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def update_class_weights(self, weights: torch.Tensor):
        """Update AU class weights dynamically (e.g. for specific folds)."""
        if weights.numel() != self.num_aus or torch.any(weights <= 0):
            raise ValueError(f"expected {self.num_aus} positive AU class weights")
        w = weights / weights.sum() * self.num_aus
        self.wal_weights.copy_(w)
        self.wal_loss.weight = self.wal_weights

    def _get_text_embeddings(self, device=None):
        """Get or compute cached text embeddings."""
        if self._cached_text_emb is None:
            self._cached_text_emb = self.text_encoder(self._text_descriptions)
        if device is not None and self._cached_text_emb.device != device:
            self._cached_text_emb = self._cached_text_emb.to(device)
        return self._cached_text_emb
        
    def _get_emotion_text_embeddings(self, device=None):
        """Get or compute cached text embeddings for emotions."""
        if self._cached_emotion_text_emb is None:
            self._cached_emotion_text_emb = self.text_encoder(self._emotion_text_descriptions)
        if device is not None and self._cached_emotion_text_emb.device != device:
            self._cached_emotion_text_emb = self._cached_emotion_text_emb.to(device)
        return self._cached_emotion_text_emb
    
    def au_to_expr_pseudo(self, Y_a):
        """
        Implement SymGraphAU Eq. (1)-(2): select the highest-scoring
        expression under the verified dataset prior and make it one-hot.
        """
        return compute_expression_pseudo_labels(Y_a.float(), self.paper_au_expression_prior)

    def _semantic_components(self, features, text_directions):
        """Pull projected text scores back into raw AU feature coordinates.

        For source AU i, u_i is the normalized input gradient of
        <visual_proj(h_i), normalize(text_proj(text_i))>. This is a local
        direction, not an inverse of visual_proj or a claim of causal isolation.
        The direction is detached; intervention gradients still flow through
        the source features and the continuous edge masks.
        """
        with torch.enable_grad():
            semantic_input = features.detach().requires_grad_(True)
            projected = self.visual_proj(semantic_input)
            scores = (projected * F.normalize(text_directions.detach(), dim=-1).unsqueeze(0)).sum()
            directions = torch.autograd.grad(scores, semantic_input)[0].detach()
        norms = directions.norm(dim=-1, keepdim=True)
        if not torch.isfinite(directions).all() or torch.any(norms <= 1e-8):
            raise RuntimeError(
                "Cannot construct semantic interventions: non-finite or zero "
                "input gradients from the trained visual projection."
            )
        directions = directions / norms
        return (features * directions).sum(dim=-1, keepdim=True) * directions

    @staticmethod
    def _intervene(features, components, masks, strength):
        """Attenuate raw-space components for each target x source mask."""
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError("cf_intervention_scale must be in [0, 1]; values above 1 overshoot")
        return (
            features.unsqueeze(1)
            - float(strength) * components.unsqueeze(1) * masks.unsqueeze(0).unsqueeze(-1)
        )

    def forward(self, images, au_labels=None, phase=None):
        """
        Forward pass of CtrlAU.
        Phase 1: Feature Extraction combining SymGraphAU backbone with CtrlAU Regularizations.
        Phase 2: Supervised graph learning, without interventions.
        Phase 3: Counterfactual graph training with the visual model frozen.
        Phase 4: End-to-end refinement.
        
        Args:
            images: (B, 3, H, W) input images
            au_labels: (B, N_AU) binary AU ground-truth labels
            phase: optional int (1, 2, 3, or 4). Defaults to self.phase.
        """
        if phase is None:
            phase = self.phase
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported phase: {phase}")
            
        device = images.device
        B = images.size(0)
        
        # ============================================================
        # 1. Visual Backbone (ResNet-50 from MultiviewSymAU) -> (B, 49, 2048)
        # ============================================================
        feat = self.backbone(images)   # (B, 49, 2048)
        z_img = feat.permute(0, 2, 1)  # (B, 2048, 49) sequence of spatial patches
        
        # ============================================================
        # 2. Global Linear (LinearBlock 2048 -> 512)
        # ============================================================
        feat = self.global_linear(feat)  # (B, 49, 512)
        
        # ============================================================
        # 3. AU Head (dataset AU branches -> V_a and p_a)
        # ============================================================
        V_a, au_probs = self.au_head(feat)
        # V_a: (B, N_AU, 256); au_probs: (B, N_AU)
        au_embeddings = [V_a[:, i, :] for i in range(self.num_aus)]
        au_logits = torch.logit(au_probs.clamp(1e-6, 1.0 - 1e-6))
        
        # ============================================================
        # 4. Expression Head (8 branches -> V_e and p_e)
        # ============================================================
        V_e, emotion_probs = self.emotion_head(feat)
        # V_e: (B, 8, 256); emotion_probs: (B, 8)
        emotion_embed_list = [V_e[:, i, :] for i in range(self.num_emotions)]
        emotion_logits = torch.logit(emotion_probs.clamp(1e-6, 1.0 - 1e-6))
        
        # ============================================================
        # Phase 1 Losses: MultiviewSymAU Backbone + CtrlAU Regularizations
        # ============================================================
        losses = {}
        emotion_pseudo = None
        
        if au_labels is not None:
            # One-hot expression target from SymGraphAU Eq. (1)-(2).
            emotion_pseudo = self.au_to_expr_pseudo(au_labels)  # (B, 8)
            
            # (1) L_wa: Weighted Asymmetric Loss on AU probabilities (MultiviewSymAU Eq. 3)
            loss_wa = self.wal_loss(au_probs, au_labels.float())
            losses["loss_wa"] = loss_wa
            losses["loss_au"] = loss_wa
            
            # (2) L_we: Expression BCE Loss on Emotion probabilities (MultiviewSymAU Eq. 4)
            loss_we = self.expression_bce_loss(emotion_probs, emotion_pseudo)
            losses["loss_we"] = loss_we
            losses["loss_emotion"] = loss_we
            
            # (3) Joint Feature Learning (JFL) Phase 1 Base Loss.
            # Keep both configured task weights active; lambda_au was
            # previously ignored here.
            lambda_au = getattr(self.cfg, "lambda_au", 1.0)
            gamma_emo = getattr(self.cfg, "lambda_emotion", 0.05)
            loss_phase1 = lambda_au * loss_wa + gamma_emo * loss_we
            losses["loss_phase1"] = loss_phase1
            
            if phase == 1:
                # ---- CtrlAU Method Regularizations in Phase 1 ----
                # 1. HSIC Disentanglement on AU embeddings
                z_img_global = z_img.mean(dim=2)  # (B, 2048)
                l_ib, l_align, l_decorr = self.hsic_loss(au_embeddings, z_img_global, au_labels)
                losses["loss_ib"] = l_ib
                losses["loss_align"] = l_align
                losses["loss_decorr"] = l_decorr
                
                # 2. Text-Visual Contrastive Alignment (AUs) - ONLY on samples where AU is actually present (y == 1)
                text_emb = self._get_text_embeddings(device=device)
                active_au_vis = []
                active_au_txt = []
                for i in range(self.num_aus):
                    pos_mask = (au_labels[:, i] == 1)
                    if pos_mask.sum() > 0:
                        pos_mean = au_embeddings[i][pos_mask].mean(dim=0)
                        active_au_vis.append(pos_mean)
                        active_au_txt.append(text_emb[i])
                
                if len(active_au_vis) >= 2:
                    visual_proj = self.visual_proj(torch.stack(active_au_vis))
                    text_proj = self.text_proj(torch.stack(active_au_txt))
                    loss_contrastive = self.contrastive_loss(visual_proj, text_proj)
                else:
                    loss_contrastive = torch.tensor(0.0, device=device)
                losses["loss_contrastive"] = loss_contrastive
                
                # 3. Text-Visual Contrastive Alignment (Emotions) - ONLY on samples where Emotion pseudo-label is present (y == 1)
                emo_text_emb = self._get_emotion_text_embeddings(device=device)
                active_emo_vis = []
                active_emo_txt = []
                for k in range(self.num_emotions):
                    emo_pos_mask = (emotion_pseudo[:, k] == 1)
                    if emo_pos_mask.sum() > 0:
                        emo_pos_mean = emotion_embed_list[k][emo_pos_mask].mean(dim=0)
                        active_emo_vis.append(emo_pos_mean)
                        active_emo_txt.append(emo_text_emb[k])
                
                if len(active_emo_vis) >= 2:
                    emo_visual_proj = self.emotion_visual_proj(torch.stack(active_emo_vis))
                    emo_text_proj = self.emotion_text_proj(torch.stack(active_emo_txt))
                    loss_emo_contrastive = self.contrastive_loss(emo_visual_proj, emo_text_proj)
                else:
                    loss_emo_contrastive = torch.tensor(0.0, device=device)
                losses["loss_emo_contrastive"] = loss_emo_contrastive
                
                # Total Combined Phase 1 Loss
                cfg = self.cfg
                total_loss = (
                    loss_phase1 +
                    cfg.lambda_ib * l_ib +
                    cfg.lambda_align * l_align +
                    cfg.lambda_decorr * l_decorr +
                    cfg.lambda_contrastive * loss_contrastive +
                    cfg.lambda_emo_contrastive * loss_emo_contrastive
                )
                losses["total_loss"] = total_loss
        else:
            if phase == 1:
                # Inference during Phase 1
                emotion_pseudo = self.au_to_expr_pseudo((au_probs > 0.5).float())

        # ============================================================
        # Early return for Phase 1: Feature Extraction
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
        
        # AU-AU graph: models cross-AU dependencies via feature-level residual
        updated_au, au_au_adj = self.graph_module.forward_au_au(au_emb_stacked)
        au_au_logits = torch.cat([self.au_au_classifiers[i](updated_au[:, i, :]) for i in range(self.num_aus)], dim=1)
        au_au_probs = torch.sigmoid(au_au_logits)
        
        # Strict one-way AU -> Expression graph.  The graph receives updated
        # AU nodes from the AU-AU graph and cannot send messages back to them.
        updated_nodes, au_exp_adj = self.graph_module.forward_au_exp(updated_au, emotion_emb_stacked)
        updated_au_final = updated_nodes[:, :self.num_aus, :]
        updated_emo_final = updated_nodes[:, self.num_aus:, :]
        
        graph_au_logits = torch.cat([self.graph_au_classifiers[i](updated_au_final[:, i, :]) for i in range(self.num_aus)], dim=1)
        graph_au_probs = torch.sigmoid(graph_au_logits)
        
        graph_emo_logits = torch.cat([self.graph_emo_classifiers[i](updated_emo_final[:, i, :]) for i in range(self.num_emotions)], dim=1)
        graph_emo_probs = torch.sigmoid(graph_emo_logits)
        
        # Graph masks
        au_au_imp_mask, au_au_pol_mask = self.mask_module.forward_au_au(au_au_adj)
        au_exp_imp_mask, au_exp_pol_mask = self.mask_module.forward_au_exp(au_exp_adj)
        
        # ============================================================
        # Losses for Phase 2 / Phase 3 (Graph Learning & Causal Interventions)
        # ============================================================
        if au_labels is not None:
            # 1. AU-AU intermediate graph prediction loss (Weighted Asymmetric Loss)
            loss_au_au = self.wal_loss(au_au_probs, au_labels.float())
            losses["loss_au_au"] = loss_au_au
            
            # 2. Final Graph AU & Emotion prediction losses
            loss_graph_au = self.wal_loss(graph_au_probs, au_labels.float())
            losses["loss_graph_au"] = loss_graph_au
            
            loss_graph_emo = self.expression_bce_loss(graph_emo_probs, emotion_pseudo)
            losses["loss_graph_emo"] = loss_graph_emo
            
            # 3. Soft DAG Acyclicity Loss on AU-AU adjacency
            loss_dag = self.dag_loss(au_au_adj)
            losses["loss_dag"] = loss_dag
            
            # 4. Causal Structure Violation Losses
            causal_au_target = au_labels.float()
            loss_causal_au = self.violation_loss(
                causal_au_target, au_au_adj, au_au_pol_mask, au_au_imp_mask
            )
            losses["loss_causal_au"] = loss_causal_au
            
            causal_exp_target = torch.cat([causal_au_target, emotion_pseudo], dim=1)
            loss_causal_exp = self.violation_loss(
                causal_exp_target, au_exp_adj, au_exp_pol_mask, au_exp_imp_mask
            )
            losses["loss_causal_exp"] = loss_causal_exp
            
            # 5. FACS AU Rule Violation Loss on AU-AU Graph & Final Graph
            loss_facs_au_au = self.facs_au_violation_loss(au_au_probs)
            loss_facs_graph_au = self.facs_au_violation_loss(graph_au_probs)
            loss_facs_au = 0.5 * (loss_facs_au_au + loss_facs_graph_au)
            losses["loss_facs_au"] = loss_facs_au
            losses["loss_facs_au_au"] = loss_facs_au_au
            losses["loss_facs_graph_au"] = loss_facs_graph_au
            
            cf_weights = (
                self.cfg.lambda_cf_au_imp, self.cfg.lambda_cf_au_unimp,
                self.cfg.lambda_cf_exp_imp, self.cfg.lambda_cf_exp_unimp,
            )
            if any(not math.isfinite(w) or w < 0 for w in cf_weights):
                raise ValueError("Counterfactual loss weights must be non-negative")
            run_cf = phase >= 3 and any(w > 0 for w in cf_weights)
            if run_cf:
                # 7. Hierarchical Counterfactual Intervention (4 Losses across Level 1 & Level 2)
                # Raw-space component attenuation along pulled-back CLIP score directions
                gamma_cf = self.cfg.cf_gamma
                alpha_cf = self.cfg.cf_intervention_scale
                if not math.isfinite(gamma_cf) or gamma_cf <= 0:
                    raise ValueError("cf_gamma must be positive and finite")

                # Text scores are aligned with visual_proj, so pull their input
                # gradients back to the raw graph-feature coordinates.
                text_emb_au = self._get_text_embeddings(device=device)
                text_directions = self.text_proj(text_emb_au).detach()
                components_1 = self._semantic_components(au_emb_stacked, text_directions)

                # =========================================================
                # LEVEL 1: AU-AU Graph Counterfactual Intervention (Losses 1 & 2)
                # Source: AU_i -> Target: AU_j
                # =========================================================
                # Soft Mask on AU-AU causal importance: Eq. 11
                m_soft_au = torch.sigmoid(
                    gamma_cf * (torch.tanh(au_au_adj).abs() - torch.sigmoid(self.cf_threshold))
                )  # continuous source -> target strengths
                # Counterfactual batches are indexed target first, source second.
                m_discrep_au = m_soft_au.transpose(0, 1)
                m_consist_au = 1.0 - m_discrep_au

                au_imp_all_1 = self._intervene(
                    au_emb_stacked, components_1, m_discrep_au, alpha_cf
                ).reshape(B * self.num_aus, self.num_aus, -1)
                au_unimp_all_1 = self._intervene(
                    au_emb_stacked, components_1, m_consist_au, alpha_cf
                ).reshape(B * self.num_aus, self.num_aus, -1)

                # Forward perturbed representations through AU-AU GAT
                up_nodes_imp_1, _ = self.graph_module.forward_au_au(au_imp_all_1)
                up_nodes_unimp_1, _ = self.graph_module.forward_au_au(au_unimp_all_1)

                # Reshape to (B, target_AU, source/target_AU_nodes, D)
                up_nodes_imp_1 = up_nodes_imp_1.reshape(B, self.num_aus, self.num_aus, -1)
                up_nodes_unimp_1 = up_nodes_unimp_1.reshape(B, self.num_aus, self.num_aus, -1)

                # For each target AU j, extract node j's representation under perturbation
                updated_au_imp_1 = torch.stack([up_nodes_imp_1[:, j, j, :] for j in range(self.num_aus)], dim=1)
                updated_au_unimp_1 = torch.stack([up_nodes_unimp_1[:, j, j, :] for j in range(self.num_aus)], dim=1)

                probs_imp_1 = torch.sigmoid(torch.cat([self.au_au_classifiers[i](updated_au_imp_1[:, i, :]) for i in range(self.num_aus)], dim=1))
                probs_unimp_1 = torch.sigmoid(torch.cat([self.au_au_classifiers[i](updated_au_unimp_1[:, i, :]) for i in range(self.num_aus)], dim=1))

                # Level 1 CF Losses: Feature Cosine + Logit MSE
                loss_cf_au_imp, loss_cf_au_unimp = self.cf_loss_au(
                    au_au_probs.detach(), probs_imp_1, probs_unimp_1,
                    feat_original=updated_au.detach(),
                    feat_important_perturbed=updated_au_imp_1,
                    feat_unimportant_perturbed=updated_au_unimp_1
                )
                losses["loss_cf_au_imp"] = loss_cf_au_imp
                losses["loss_cf_au_unimp"] = loss_cf_au_unimp

                # =========================================================
                # LEVEL 2: AU-Expression Graph Counterfactual Intervention (Losses 3 & 4)
                # Source: AU_i -> Target: Expression_k
                # =========================================================
                # Extract AU -> Expression edges (source AU, target emotion) and
                # present them as target-emotion x source-AU masks.
                au_exp_strength = torch.tanh(au_exp_adj[:self.num_aus, self.num_aus:]).abs().transpose(0, 1)
                m_soft_exp = torch.sigmoid(
                    gamma_cf * (au_exp_strength - torch.sigmoid(self.cf_threshold_exp))
                )  # (N_expression, N_AU), without hard-mask quantization
                m_discrep_exp = m_soft_exp
                m_consist_exp = 1.0 - m_soft_exp

                components_2 = self._semantic_components(updated_au, text_directions)
                au_imp_all_2 = self._intervene(
                    updated_au, components_2, m_discrep_exp, alpha_cf
                ).reshape(B * self.num_emotions, self.num_aus, -1)
                au_unimp_all_2 = self._intervene(
                    updated_au, components_2, m_consist_exp, alpha_cf
                ).reshape(B * self.num_emotions, self.num_aus, -1)

                emo_repeat_2 = emotion_emb_stacked.unsqueeze(1).expand(-1, self.num_emotions, -1, -1).reshape(B * self.num_emotions, self.num_emotions, -1)

                # Forward perturbed representations through AU-Exp Bipartite GAT
                up_nodes_imp_2, _ = self.graph_module.forward_au_exp(au_imp_all_2, emo_repeat_2)
                up_nodes_unimp_2, _ = self.graph_module.forward_au_exp(au_unimp_all_2, emo_repeat_2)

                up_nodes_imp_2 = up_nodes_imp_2.reshape(B, self.num_emotions, self.num_aus + self.num_emotions, -1)
                up_nodes_unimp_2 = up_nodes_unimp_2.reshape(B, self.num_emotions, self.num_aus + self.num_emotions, -1)

                # For each target expression k, extract expression node k.
                updated_emo_imp_2 = torch.stack([up_nodes_imp_2[:, k, self.num_aus + k, :] for k in range(self.num_emotions)], dim=1)
                updated_emo_unimp_2 = torch.stack([up_nodes_unimp_2[:, k, self.num_aus + k, :] for k in range(self.num_emotions)], dim=1)

                probs_imp_2 = torch.sigmoid(torch.cat([self.graph_emo_classifiers[k](updated_emo_imp_2[:, k, :]) for k in range(self.num_emotions)], dim=1))
                probs_unimp_2 = torch.sigmoid(torch.cat([self.graph_emo_classifiers[k](updated_emo_unimp_2[:, k, :]) for k in range(self.num_emotions)], dim=1))

                # Level 2 CF losses: feature cosine and probability MSE
                loss_cf_exp_imp, loss_cf_exp_unimp = self.cf_loss_exp(
                    graph_emo_probs.detach(), probs_imp_2, probs_unimp_2,
                    feat_original=updated_emo_final.detach(),
                    feat_important_perturbed=updated_emo_imp_2,
                    feat_unimportant_perturbed=updated_emo_unimp_2
                )
                losses["loss_cf_exp_imp"] = loss_cf_exp_imp
                losses["loss_cf_exp_unimp"] = loss_cf_exp_unimp

                # Combined / Global Counterfactual Losses
                loss_cf_imp = 0.5 * (loss_cf_au_imp + loss_cf_exp_imp)
                loss_cf_unimp = 0.5 * (loss_cf_au_unimp + loss_cf_exp_unimp)
                losses["loss_cf_important"] = loss_cf_imp
                losses["loss_cf_unimportant"] = loss_cf_unimp

            # Phase 2 Total Loss: Pure Graph & Hierarchical Causal Reasoning (Eq. 16)
            cfg = self.cfg
            total_loss = (
                cfg.lambda_au_au * loss_au_au +
                cfg.lambda_graph_au * loss_graph_au +
                cfg.lambda_graph_emo * loss_graph_emo +
                cfg.lambda_dag * loss_dag +
                cfg.lambda_causal_au * loss_causal_au +
                cfg.lambda_causal_exp * loss_causal_exp +
                cfg.lambda_facs_au * loss_facs_au
            )
            if run_cf:
                total_loss = total_loss + sum(
                    w * losses[name] for w, name in zip(cf_weights, (
                        "loss_cf_au_imp", "loss_cf_au_unimp",
                        "loss_cf_exp_imp", "loss_cf_exp_unimp",
                    ))
                )
            losses["total_loss"] = total_loss
        # ============================================================
        # Test-Time Symbolic Energy Minimization (Abductive Inference)
        # ============================================================
        # By default, use direct GAT predictions. Enable via cfg.enable_test_time_refinement = True
        raw_graph_au_probs = graph_au_probs
        raw_graph_emo_probs = graph_emo_probs
        if not self.training and getattr(self.cfg, "enable_test_time_refinement", False):
            # Fixed forward AU -> expression pseudo-targets. No expression
            # reconstruction of AUs, and no gradient from this target into AUs.
            prior_targets = self.au_to_expr_pseudo((graph_au_probs.detach() > 0.5).float())
            with torch.enable_grad():
                refined_au_logits = graph_au_logits.detach().clone().requires_grad_(True)
                refined_emo_logits = graph_emo_logits.detach().clone().requires_grad_(True)
                
                optimizer = torch.optim.Adam([refined_au_logits, refined_emo_logits], lr=0.01)
                
                for _ in range(10):
                    optimizer.zero_grad()
                    p_au = torch.sigmoid(refined_au_logits)
                    p_emo = torch.sigmoid(refined_emo_logits)
                    
                    e_au = self.facs_au_violation_loss(p_au)
                    e_exp = self.expression_bce_loss(p_emo, prior_targets)
                    # Strong prior anchor (1.0) so visual evidence is not overridden by rule suppression
                    e_prior = F.mse_loss(refined_au_logits, graph_au_logits.detach()) + \
                              F.mse_loss(refined_emo_logits, graph_emo_logits.detach())
                              
                    energy = 0.05 * (e_au + e_exp) + 1.0 * e_prior
                    energy.backward()
                    optimizer.step()
                    
                graph_au_probs = torch.sigmoid(refined_au_logits).detach()
                graph_emo_probs = torch.sigmoid(refined_emo_logits).detach()

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
            "raw_graph_au_probs": raw_graph_au_probs,
            "raw_graph_emo_probs": raw_graph_emo_probs,
            "au_au_adj": au_au_adj,
            "au_exp_adj": au_exp_adj,
            "au_au_importance_mask": au_au_imp_mask,
            "au_au_polarity_mask": au_au_pol_mask,
            "au_exp_importance_mask": au_exp_imp_mask,
            "au_exp_polarity_mask": au_exp_pol_mask,
            "z_img": z_img,
            "losses": losses,
        }
