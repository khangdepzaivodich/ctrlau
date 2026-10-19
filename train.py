"""
Training script for CtrlAU.
Matches original MultiviewSymAU training pipeline:
  - Per-batch cosine annealing LR schedule
  - No gradient clipping
  - Resize(256) -> CenterCrop(224) + ColorJitter (train) / no jitter (val)
"""
import os
import random
import argparse
import warnings
from math import cos, pi, isfinite
import numpy as np
import torch

# Suppress PyTorch JIT deprecation FutureWarning from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.*")
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import (
    ModelConfig, DISFA_AUS, BP4D_AUS, get_dataset_aus,
    get_dataset_au_expression_prior,
)
from dataset import DISFADataset, BP4DDataset
from models.ctrlau import CtrlAUModel


TRAINING_SCHEMA = "ctrlau_one_way_v2"
CF_MASK_SCHEMA = "incoming_relative_v1"


def load_training_checkpoint(model, path, *, phase, fold, dataset, model_name,
                             device, resume_phase=None):
    """Validate the protocol and every tensor before changing model state.

    Phase-1 checkpoints from the previous implementation remain compatible:
    only the two obsolete, deterministic loss buffers are removed. No learned
    tensor is fabricated or accepted as missing. Old graph-stage checkpoints
    are rejected because their objectives and intervention semantics changed.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    required = {"state_dict", "phase", "fold", "dataset", "au_ids", "model", "num_emotions", "epoch"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint must contain complete model, dataset, fold and phase metadata")
    expected = {
        "dataset": dataset, "fold": fold, "model": model_name,
        "num_emotions": getattr(model, "num_emotions", None),
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(f"Checkpoint {name} mismatch: {payload[name]!r} != {value!r}")
    if tuple(payload["au_ids"]) != get_dataset_aus(dataset):
        raise ValueError("Checkpoint AU vocabulary/order mismatch")
    source_phase = payload["phase"]
    if source_phase not in (1, 2, 3, 4) or source_phase not in (phase, phase - 1):
        raise ValueError(f"Cannot enter phase {phase} from phase {source_phase}")
    if resume_phase is not None and source_phase != resume_phase:
        raise ValueError("--resume_phase disagrees with checkpoint metadata")
    if model_name == "ctrlau":
        if payload.get("graph_topology") != "au_to_expression":
            raise ValueError("Checkpoint does not use the strict AU->Expression topology")
        if source_phase >= 2 and payload.get("training_schema") != TRAINING_SCHEMA:
            raise ValueError("Old graph-stage objectives are incompatible; restart phase 2 from phase 1")
        if source_phase >= 3 and payload.get("cf_mask_schema") != CF_MASK_SCHEMA:
            raise ValueError("Old phase-3 mask calibration is incompatible; restart phase 3 from phase 2")
    if source_phase == phase:
        if not {"optimizer", "best_f1"}.issubset(payload):
            raise ValueError("Same-phase resume requires optimizer and best_f1 metadata")
        if not isinstance(payload["optimizer"], dict) or not {"state", "param_groups"}.issubset(payload["optimizer"]):
            raise ValueError("Same-phase resume requires a complete optimizer state")
        if not isfinite(payload["best_f1"]) or not 0.0 <= payload["best_f1"] <= 1.0:
            raise ValueError("Checkpoint best_f1 must be in [0, 1]")
    if not isinstance(payload["epoch"], int) or payload["epoch"] < 1:
        raise ValueError("Checkpoint must contain a positive epoch")

    cleaned = {}
    for key, value in payload["state_dict"].items():
        name = key.removeprefix("module.")
        if model_name == "ctrlau":
            name = name.removeprefix("stage1.")
        if name in cleaned:
            raise ValueError(f"Duplicate checkpoint tensor after prefix removal: {name}")
        cleaned[name] = value
    if model_name == "ctrlau" and source_phase == 1:
        for name in ("cycle_loss.rule_matrix", "au_expression_support_loss.support_matrix"):
            if name in cleaned:
                value = cleaned.pop(name)
                if not torch.equal(value.to(model.au_expression_support), model.au_expression_support):
                    raise ValueError(f"Obsolete prior buffer does not match the selected dataset: {name}")
    target = model.state_dict()
    missing = sorted(set(target) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(target))
    if missing or unexpected:
        raise ValueError(f"Incomplete checkpoint: missing={missing}, unexpected={unexpected}")
    for name, value in cleaned.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[name].shape:
            raise ValueError(f"Checkpoint tensor shape/type mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    if model_name == "ctrlau":
        if not torch.equal(cleaned["paper_au_expression_prior"].to(model.paper_au_expression_prior),
                           model.paper_au_expression_prior):
            raise ValueError("Checkpoint expression prior mismatch")
        for name in ("wal_weights", "wal_loss.weight"):
            if not torch.allclose(cleaned[name].to(model.wal_weights), model.wal_weights):
                raise ValueError("Checkpoint class weights do not match the selected training split")
    model.load_state_dict(cleaned, strict=True)
    if model_name == "ctrlau" and source_phase == 1 and phase == 2:
        model.init_graph_classifiers_from_base()
    if model_name == "ctrlau" and source_phase == 2 and phase == 3:
        model.initialize_counterfactual_thresholds()
        print("CF masks initialized: cutoff = each target's mean incoming routing strength")
    return payload


def configure_training_phase(model, args):
    """Apply the CLI's phase-specific freezing and objective weights."""
    model.set_phase(args.phase)
    if args.phase == 1:
        print("=== Phase 1: Feature Extraction ===")
        model.cfg.lambda_au = 1.0
        model.cfg.lambda_emotion = 0.05
        model.cfg.lambda_facs_au = 0.0
        if args.model == "ctrlau":
            print("CtrlAU Method: Phase 1 Feature Learning with Core Method Regularizers:")
            print("  - HSIC Disentanglement (L_ib, L_align, L_decorr)")
            print("  - CLIP Text-Visual Contrastive Alignment (AUs & Emotions on active y=1)")
            print("  - FACS AU Rule: Removed (0.0) from Phase 1")
            model.cfg.lambda_ib = 1e-2
            model.cfg.lambda_align = 1e-2
            model.cfg.lambda_decorr = 1e-2
            model.cfg.lambda_contrastive = 0.05
            model.cfg.lambda_emo_contrastive = 0.05
        else:
            print("Pure SymGraphAU: MultiviewSymAU Stage 1 Baseline (No Phase 1 regularizers)")
            model.cfg.lambda_ib = 0.0
            model.cfg.lambda_align = 0.0
            model.cfg.lambda_decorr = 0.0
            model.cfg.lambda_contrastive = 0.0
            model.cfg.lambda_emo_contrastive = 0.0
            model.cfg.lambda_facs_au = 0.0

        model.cfg.lambda_dag = 0.0
        model.cfg.lambda_causal_au = 0.0
        model.cfg.lambda_causal_exp = 0.0
        model.cfg.lambda_au_au = 0.0
        model.cfg.lambda_graph_au = 0.0
        model.cfg.lambda_graph_emo = 0.0

    elif args.phase == 2:
        print("=== Phase 2: Causal Graph Routing ===")
        print("Freezing Backbone and Base Classifiers. Training only the Graph Routing Modules.")
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
        if hasattr(model, "global_linear"):
            for param in model.global_linear.parameters():
                param.requires_grad = False
        if hasattr(model, "au_head"):
            for param in model.au_head.parameters():
                param.requires_grad = False
        if hasattr(model, "emotion_head"):
            for param in model.emotion_head.parameters():
                param.requires_grad = False
        if hasattr(model, "text_encoder"):
            for param in model.text_encoder.parameters():
                param.requires_grad = False
        if hasattr(model, "visual_proj"):
            for param in model.visual_proj.parameters():
                param.requires_grad = False
        if hasattr(model, "text_proj"):
            for param in model.text_proj.parameters():
                param.requires_grad = False
        if hasattr(model, "emotion_visual_proj"):
            for param in model.emotion_visual_proj.parameters():
                param.requires_grad = False
        if hasattr(model, "emotion_text_proj"):
            for param in model.emotion_text_proj.parameters():
                param.requires_grad = False

        # Freeze base CNN classification losses, enable Graph losses ONLY
        model.cfg.lambda_au = 0.0
        model.cfg.lambda_emotion = 0.0
        model.cfg.lambda_ib = 0.0
        model.cfg.lambda_align = 0.0
        model.cfg.lambda_decorr = 0.0
        model.cfg.lambda_contrastive = 0.0
        model.cfg.lambda_emo_contrastive = 0.0
        model.cfg.lambda_facs_au = 0.05 if args.model == "ctrlau" else 0.0

        # Enable supervised graph losses
        model.cfg.lambda_au_au = 1.0
        model.cfg.lambda_graph_au = 1.0
        model.cfg.lambda_graph_emo = 1.0
        model.cfg.lambda_dag = 0.1

        # NO Counterfactual losses in Phase 2
        model.cfg.lambda_causal_au = 0.0
        model.cfg.lambda_causal_exp = 0.0
        model.cfg.lambda_cf_au_imp = 0.0
        model.cfg.lambda_cf_au_unimp = 0.0
        model.cfg.lambda_cf_exp_imp = 0.0
        model.cfg.lambda_cf_exp_unimp = 0.0

    elif args.phase in (3, 4):
        if args.phase == 3:
            print("=== Phase 3: Counterfactual Intervention ===")
            print("Activating Counterfactual Pass and CF Losses on top of the Graph.")
        else:
            print("=== Phase 4: End-to-End Fine-Tuning ===")
            print("Training visual feature extractors and graphs with the phase-3 objective.")
            print("CLIP, semantic projections and phase-1-only classifiers stay frozen.")

        # The objective is identical in phases 3/4; set_phase controls which
        # modules train. Never inherit different weights from config defaults
        # or from a previously configured phase in the same process.
        model.cfg.lambda_au = 0.0
        model.cfg.lambda_emotion = 0.0
        model.cfg.lambda_ib = 0.0
        model.cfg.lambda_align = 0.0
        model.cfg.lambda_decorr = 0.0
        model.cfg.lambda_contrastive = 0.0
        model.cfg.lambda_emo_contrastive = 0.0
        model.cfg.lambda_facs_au = 0.05 if args.model == "ctrlau" else 0.0

        # Supervised graph losses remain active
        model.cfg.lambda_au_au = 1.0
        model.cfg.lambda_graph_au = 1.0
        model.cfg.lambda_graph_emo = 1.0
        model.cfg.lambda_dag = 0.1

        # Balanced CF weights (0.05) so counterfactual regularization doesn't overpower task accuracy
        model.cfg.lambda_causal_au = 0.05
        model.cfg.lambda_causal_exp = 0.05
        model.cfg.lambda_cf_au_imp = 0.05
        model.cfg.lambda_cf_au_unimp = 0.05
        model.cfg.lambda_cf_exp_imp = 0.05
        model.cfg.lambda_cf_exp_unimp = 0.05


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):
    """Per-batch cosine annealing LR schedule matching MultiviewSymAU utils.py verbatim."""
    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def keep_frozen_modules_eval(model):
    """Keep frozen graph-stage modules deterministic without freezing phase-4 features."""
    phase = getattr(model, "phase", None)
    if phase not in (2, 3, 4):
        return

    frozen_modules = [
        "text_encoder", "visual_proj", "text_proj",
        "emotion_visual_proj", "emotion_text_proj",
    ]
    if phase in (2, 3):
        frozen_modules += ["backbone", "global_linear", "au_head", "emotion_head"]
    for module_name in frozen_modules:
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


def build_optimizer(model, lr, weight_decay):
    # Preserve phase-4 optimizer ordering for existing checkpoints. AdamW
    # skips frozen parameters (grad=None), so retaining them does not update
    # the semantic anchors or the phase-1-only classifier weights.
    parameters = [
        p for name, p in model.named_parameters()
        if p.requires_grad or (
            getattr(model, "phase", None) == 4 and not name.startswith("text_encoder.")
        )
    ]
    return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg=None, init_lr=1e-4):
    """Train for one epoch."""
    model.train()
    keep_frozen_modules_eval(model)
    running_losses = {}
    num_batches = 0
    
    train_loader_len = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Per-batch cosine annealing LR
        if cfg is not None:
            adjust_learning_rate(optimizer, epoch - 1, cfg.num_epochs, init_lr,
                                 batch_idx, train_loader_len)
        
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)
        
        # Forward
        outputs = model(images, au_labels=au_labels)
        losses = outputs["losses"]
        total_loss = losses["total_loss"]

        # Probe CF-only gradient signals once per epoch, without changing .grad
        # or the optimizer. Total backward below still runs exactly once.
        if batch_idx == 0 and "loss_cf_au_imp" in losses:
            au_grad = torch.autograd.grad(
                losses["loss_cf_au_imp"], model.cf_threshold, retain_graph=True
            )[0]
            exp_grad = torch.autograd.grad(
                losses["loss_cf_exp_imp"], model.cf_threshold_exp, retain_graph=True
            )[0]
            print(f"  CF-only threshold |grad|: AU={au_grad.abs().item():.3e}, Expr={exp_grad.abs().item():.3e}")
            for name, value in outputs["cf_diagnostics"].items():
                print(f"  {name}: {value.item():.6e}")
        
        # Backward (no gradient clipping, matching original repo)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v.item()
        if "cf_diagnostics" in outputs:
            for k, v in outputs["cf_diagnostics"].items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v.item()
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_losses["total_loss"] / num_batches
            msg = f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, Total Loss: {avg_total:.4f}"
            
            # Phase 1 CNN Losses (if in Phase 1)
            if running_losses.get("loss_graph_au", 0.0) == 0.0:
                if "loss_wa" in running_losses:
                    msg += f", L_wa: {running_losses['loss_wa'] / num_batches:.4f}"
                if "loss_we" in running_losses:
                    msg += f", L_we: {running_losses['loss_we'] / num_batches:.4f}"
                if "loss_ib" in running_losses and abs(running_losses['loss_ib']) > 1e-6:
                    msg += f", L_ib: {running_losses['loss_ib'] / num_batches:.4f}"
                if "loss_align" in running_losses and abs(running_losses['loss_align']) > 1e-6:
                    msg += f", L_align: {running_losses['loss_align'] / num_batches:.4f}"
                if "loss_decorr" in running_losses and abs(running_losses['loss_decorr']) > 1e-6:
                    msg += f", L_decorr: {running_losses['loss_decorr'] / num_batches:.4f}"
                if "loss_contrastive" in running_losses and running_losses['loss_contrastive'] > 0:
                    msg += f", L_clip_au: {running_losses['loss_contrastive'] / num_batches:.4f}"
                if "loss_emo_contrastive" in running_losses and running_losses['loss_emo_contrastive'] > 0:
                    msg += f", L_clip_exp: {running_losses['loss_emo_contrastive'] / num_batches:.4f}"
                if "loss_facs_au" in running_losses and running_losses['loss_facs_au'] > 0:
                    msg += f", L_facs: {running_losses['loss_facs_au'] / num_batches:.4f}"
            else:
                # Phase 2 & 3: Graph, relational and counterfactual losses
                if "loss_graph_au" in running_losses:
                    msg += f", L_g_au: {running_losses['loss_graph_au'] / num_batches:.4f}"
                if "loss_graph_emo" in running_losses:
                    msg += f", L_g_emo: {running_losses['loss_graph_emo'] / num_batches:.4f}"
                if "loss_au_au" in running_losses:
                    msg += f", L_au_au: {running_losses['loss_au_au'] / num_batches:.4f}"
                if "loss_dag" in running_losses and running_losses['loss_dag'] > 0:
                    msg += f", L_dag: {running_losses['loss_dag'] / num_batches:.4f}"
                if ("loss_causal_au" in running_losses and running_losses['loss_causal_au'] > 0) or ("loss_causal_exp" in running_losses and running_losses['loss_causal_exp'] > 0):
                    l_c = (running_losses.get('loss_causal_au', 0.0) + running_losses.get('loss_causal_exp', 0.0)) / num_batches
                    msg += f", L_causal: {l_c:.4f}"
                if "loss_cf_au_imp" in running_losses and running_losses['loss_cf_au_imp'] > 0:
                    msg += f", CF_au: {running_losses['loss_cf_au_imp'] / num_batches:.6f}"
                if "loss_cf_exp_imp" in running_losses and running_losses['loss_cf_exp_imp'] > 0:
                    msg += f", CF_exp: {running_losses['loss_cf_exp_imp'] / num_batches:.6f}"
                if "loss_cf_au_unimp" in running_losses and running_losses['loss_cf_au_unimp'] > 0:
                    cf_unimp = (running_losses['loss_cf_au_unimp'] + running_losses.get('loss_cf_exp_unimp', 0.0)) / (2 * num_batches)
                    msg += f", CF_unimp: {cf_unimp:.4f}"
                if "cf_au_mask_mean" in running_losses:
                    msg += f", Mask_AU: {running_losses['cf_au_mask_mean'] / num_batches:.3f}"
                    msg += f", Mask_E: {running_losses['cf_exp_mask_mean'] / num_batches:.3f}"
                    msg += f", dP_AU: {running_losses['cf_au_prob_mse_imp'] / num_batches:.2e}"
                    msg += f", dP_E: {running_losses['cf_exp_prob_mse_imp'] / num_batches:.2e}"
            print(msg)
    
    # Print epoch summary
    print(f"\n--- Epoch {epoch} Summary ---")
    for k, v in sorted(running_losses.items()):
        if k.startswith("cf_"):
            print(f"  {k}: {v / num_batches:.6e}")
        else:
            print(f"  {k}: {v / num_batches:.4f}")
    print()
    
    return {k: v / num_batches for k, v in running_losses.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, au_ids, phase=1):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)
        
        outputs = model(images)
        
        # In Phase 1, we validate the CNN Backbone.
        # In Phase 2 & 3, we validate the final Graph predictions!
        if phase >= 2:
            au_probs = outputs["graph_au_probs"]
        else:
            au_probs = outputs["au_probs"]
        
        all_preds.append(au_probs.cpu())
        all_labels.append(au_labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute per-AU F1
    preds_binary = (all_preds > 0.5).float()
    
    f1_scores = []
    for i, au in enumerate(au_ids):
        tp = ((preds_binary[:, i] == 1) & (all_labels[:, i] == 1)).sum().float()
        fp = ((preds_binary[:, i] == 1) & (all_labels[:, i] == 0)).sum().float()
        fn = ((preds_binary[:, i] == 0) & (all_labels[:, i] == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
        print(f"  AU{au}: F1={f1:.4f}, Prec={precision:.4f}, Recall={recall:.4f}")
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"  Average F1: {avg_f1:.4f}")
    
    return avg_f1
def main():
    parser = argparse.ArgumentParser(description="Train CtrlAU Model")
    parser.add_argument(
        "--data_root", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), "DISFA_Data"),
        help="Path to the selected dataset root containing its official image/list layout"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="disfa",
        choices=["disfa", "bp4d"],
        help="Benchmark protocol and AU vocabulary",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint (.pth file) to resume training from"
    )
    parser.add_argument(
        "--resume_phase",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Optional source-phase assertion; checkpoints must contain protocol metadata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Directory for phase/fold-specific checkpoints",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Training phase: 1 (Base Classifiers), 2 (Causal Graph), 3 (Counterfactual Intervention), 4 (End-to-End)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,  
        choices=[1, 2, 3],
        help="Which fold to use for validation (1, 2, or 3). The other two will be used for training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ctrlau",
        choices=["symgraphau", "ctrlau"],
        help="Model architecture: 'ctrlau' (default) or 'symgraphau' (explicit baseline only)"
    )
    parser.add_argument(
        "--lr", "-lr",
        type=float,
        default=None,
        help="Learning rate (defaults: phases 1/2=1e-4, phase 3=2e-5, phase 4=1e-5)",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64, matching ynhi's original)",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0, matching ynhi's conf.py)",
    )
    parser.add_argument(
        "--cf-scale",
        type=float,
        default=1.0,
        help="Semantic component attenuation in [0, 1] (default: 1.0; no overshoot)",
    )
    args = parser.parse_args()
    if args.lr is None:
        args.lr = {1: 1e-4, 2: 1e-4, 3: 2e-5, 4: 1e-5}[args.phase]
    if not isfinite(args.cf_scale) or not 0.0 <= args.cf_scale <= 1.0:
        parser.error("--cf-scale must be in [0, 1]; larger values amplify instead of attenuating")
    if args.phase >= 2 and not args.resume:
        parser.error("Phases 2/3/4 require --resume from the preceding phase or the same phase")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    if args.model == "symgraphau" and args.phase != 1:
        raise ValueError("The copied legacy SymGraphAU wrapper supports phase 1 only; use CtrlAU for graph phases")
    if args.model == "symgraphau" and args.dataset != "disfa":
        raise ValueError("The copied legacy SymGraphAU wrapper is DISFA-only; use CtrlAU for BP4D")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg = ModelConfig()
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.cf_intervention_scale = args.cf_scale
    cfg.phase = args.phase
    cfg.dataset_name = args.dataset
    cfg.au_ids = get_dataset_aus(args.dataset)
    cfg.au_expression_prior = get_dataset_au_expression_prior(args.dataset)

    # ---- Seeding (identical to ynhi's conf.py set_env) ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    au_ids = tuple(cfg.au_ids)
    print(f"Dataset: {args.dataset.upper()}, AUs: {list(au_ids)}")
    print(f"Model: {args.model}, Phase: {args.phase}, Fold: {args.fold}")
    print(f"Learning rate: {args.lr}, Batch size: {cfg.batch_size}, Epochs: {cfg.num_epochs}, Seed: {args.seed}")
    
    # Exact DISFA 3-Fold splits from MultiviewSymAU. BP4D uses its prepared
    # train/test list files for the same fold number.
    group_1 = ["SN002", "SN010", "SN001", "SN026", "SN027", "SN032", "SN030", "SN009", "SN016"]
    group_2 = ["SN013", "SN018", "SN011", "SN028", "SN012", "SN006", "SN031", "SN021", "SN024"]
    group_3 = ["SN003", "SN029", "SN023", "SN025", "SN008", "SN005", "SN007", "SN017", "SN004"]
    
    if args.fold == 1:
        train_subjects = group_1 + group_2
        val_subjects = group_3
    elif args.fold == 2:
        train_subjects = group_1 + group_3
        val_subjects = group_2
    else:  # fold 3
        train_subjects = group_2 + group_3
        val_subjects = group_1
        
    print(f"--- FOLD {args.fold} SPLIT ---")
    if args.dataset == "disfa":
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
        train_dataset = DISFADataset(
            data_root=args.data_root, subjects=train_subjects, train=True, fold=args.fold
        )
        val_dataset = DISFADataset(
            data_root=args.data_root, subjects=val_subjects, train=False, fold=args.fold
        )
    else:
        if args.model == "symgraphau":
            raise ValueError("The copied legacy SymGraphAU wrapper is DISFA-only; use --model ctrlau for BP4D")
        train_dataset = BP4DDataset(data_root=args.data_root, train=True, fold=args.fold)
        val_dataset = BP4DDataset(data_root=args.data_root, train=False, fold=args.fold)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"Dataset is empty! Found {len(train_dataset)} train samples and {len(val_dataset)} val samples.\n"
            f"Please check that --data_root points to the official {args.dataset.upper()} layout."
        )
    
    # DataLoader settings matching original repo (no drop_last)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    
    # Model
    if args.model == "symgraphau":
        from models.symgraphau import SymGraphAUModel
        model = SymGraphAUModel(cfg=cfg).to(device)
    else:
        from models.ctrlau import CtrlAUModel
        model = CtrlAUModel(cfg=cfg).to(device)
        
    if hasattr(model, "set_phase"):
        model.set_phase(args.phase)
        
    # Class weights must come from the selected training split.  This avoids
    # silently reusing weights from another dataset, fold, or AU ordering.
    print(f"Calculating AU class weights from the {args.dataset.upper()} training split...")
    computed_weights, occur_rates = train_dataset.calculate_class_weights()
    if computed_weights is None:
        raise ValueError("Dataset did not provide class weights for a non-empty training split")
    computed_weights = computed_weights.to(device)
    if not hasattr(model, "update_class_weights"):
        raise TypeError("Selected model does not implement dataset-specific AU class weights")
    model.update_class_weights(computed_weights)
    print(f"AU weights (sum={computed_weights.sum().item():.2f}):")
    for au_idx, au_id in enumerate(au_ids):
        rate_pct = occur_rates[au_idx].item() * 100
        print(f"  AU{au_id:<2}: weight={computed_weights[au_idx].item():.4f} (occurrence: {rate_pct:.2f}%)")
    
    # ==========================================
    # Phase-specific objectives and freezing
    # ==========================================
    configure_training_phase(model, args)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    resume_payload = None
    checkpoint_phase = None
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        resume_payload = load_training_checkpoint(
            model, args.resume, phase=args.phase, fold=args.fold,
            dataset=args.dataset, model_name=args.model, device=device,
            resume_phase=args.resume_phase,
        )
        checkpoint_phase = resume_payload["phase"]
        print(f"Checkpoint loaded strictly: phase {checkpoint_phase}, fold {args.fold}")

    # Optimizer matching original SymGraphAU verbatim:
    # optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=cfg.weight_decay)
    print(f"Initial learning rate: {args.lr:.2e}")

    start_epoch = 1
    if resume_payload is not None and checkpoint_phase == args.phase:
        optimizer_state = resume_payload.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            start_epoch = int(resume_payload.get("epoch", 0)) + 1
            print(f"Resuming phase {args.phase} at epoch {start_epoch}.")
    
    # No epoch-level scheduler — LR is adjusted per-batch inside train_one_epoch
    # using adjust_learning_rate() (cosine annealing matching original repo)
    
    # Training loop
    best_f1 = resume_payload["best_f1"] if checkpoint_phase == args.phase else 0.0
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.num_epochs}")
        print(f"{'='*60}")
        
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg=cfg, init_lr=args.lr)
        
        # Evaluate
        print("Validation:")
        avg_f1 = evaluate(model, val_loader, device, au_ids=au_ids, phase=args.phase)
        
        # Phase 1 graph heads are frozen, so synchronize them with the trained
        # base heads before writing a checkpoint for the next phase.
        if args.model == "ctrlau" and args.phase == 1:
            model.init_graph_classifiers_from_base()

        improved = avg_f1 > best_f1
        best_f1 = max(best_f1, avg_f1)
        checkpoint = {
            "training_schema": TRAINING_SCHEMA,
            "cf_mask_schema": CF_MASK_SCHEMA if args.phase >= 3 else None,
            "best_f1": best_f1,
            "f1": avg_f1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "phase": args.phase,
            "fold": args.fold,
            "model": args.model,
            "dataset": args.dataset,
            "au_ids": list(au_ids),
            "num_emotions": getattr(model, "num_emotions", None),
            "graph_topology": "au_to_expression",
        }
        epoch_ckpt_path = os.path.join(
            output_dir,
            f"{args.model}_phase{args.phase}_fold{args.fold}_epoch{epoch}.pth",
        )
        torch.save(checkpoint, epoch_ckpt_path)
        print(f"  -> Saved epoch checkpoint: {epoch_ckpt_path}")

        # Save best model
        if improved:
            best_ckpt_path = os.path.join(
                output_dir,
                f"{args.model}_phase{args.phase}_fold{args.fold}_best.pth",
            )
            torch.save(checkpoint, best_ckpt_path)
            print(f"  -> New best F1: {best_f1:.4f}, '{best_ckpt_path}' updated.")
    
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
