"""Validate a CtrlAU graph checkpoint across refinement iteration counts.

K=0 is always included. Read each validation image and run the visual/graph
network once. Refinement snapshots share the exact Adam trajectory from the
model's existing ten-step loop. Only copied logits are optimized, never model
parameters. No validation labels enter prediction or refinement.
"""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, get_dataset_aus, get_dataset_au_expression_prior
from dataset import DISFADataset, BP4DDataset
from diagnose_phase4 import au_metrics
from models.ctrlau import CtrlAUModel
from train import load_training_checkpoint


@torch.no_grad()
def evaluate_k_grid(model, loader, device, k_values):
    k_values = tuple(sorted({0, *k_values}))
    if any(type(k) is not int or k < 0 for k in k_values):
        raise ValueError("K values must be non-negative integers")
    if getattr(model.cfg, "enable_test_time_refinement", False):
        raise ValueError("Disable automatic model refinement before running the K grid")
    model.eval()
    predictions = {k: [] for k in k_values}
    labels = []
    for batch_index, batch in enumerate(loader, 1):
        # Do not pass AU ground truth into the model or the refinement method.
        outputs = model(batch["image"].to(device, non_blocking=True))
        snapshots = model.refine_graph_predictions(
            outputs["graph_au_logits"], outputs["graph_emo_logits"], k_values
        )
        for k in k_values:
            predictions[k].append(snapshots[k][0].cpu())
        labels.append(batch["au_labels"].cpu())
        if batch_index % 100 == 0:
            print(f"Validation: {batch_index}/{len(loader)} batches; all K values evaluated", flush=True)
    if not labels:
        raise ValueError("Validation split is empty")
    labels = torch.cat(labels)
    return {k: au_metrics(torch.cat(predictions[k]), labels) for k in k_values}


def print_results(results, au_ids):
    means = {}
    for k, metrics in sorted(results.items()):
        print(f"\nK = {k}" + (" (direct graph predictions)" if k == 0 else ""))
        for i, au in enumerate(au_ids):
            print(f"  AU{au}: F1={metrics['f1'][i]:.4f}, "
                  f"Prec={metrics['precision'][i]:.4f}, Recall={metrics['recall'][i]:.4f}")
        means[k] = sum(metrics["f1"].tolist()) / len(au_ids)
        print(f"  Average F1: {means[k]:.6f} ({100 * means[k]:.4f}%)")
    print("\nK     Average F1 (%)    Change vs K=0 (pp)")
    for k, mean in sorted(means.items()):
        print(f"{k:<5d} {100 * mean:>12.4f} {100 * (mean - means[0]):>21.4f}")
    best_k = max(sorted(means), key=means.get)
    print(f"Best validation K: {best_k}; Average F1: {100 * means[best_k]:.4f}% "
          "(ties select smaller K)")
    print("K is selected on validation data; this script does not update a model or write checkpoints.")
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", choices=("disfa", "bp4d"), default="disfa")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--phase", type=int, choices=(2, 3, 4), default=3,
                        help="Must match the phase stored in the checkpoint")
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10, 15, 20, 30])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1 or args.num_workers < 0 or any(k < 0 for k in args.ks):
        parser.error("batch_size must be positive; num_workers and K must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = sorted({0, *args.ks})
    print(f"Device: {device}; dataset: {args.dataset}; fold: {args.fold}; K: {k_values}", flush=True)
    print("Refinement: Adam lr=0.01; rule/expression energy weight=0.05; logit anchor weight=1.0.")
    print("Use the same batch size as normal validation to preserve the existing refinement scaling.")

    if args.dataset == "disfa":
        # Same subject order and fold mapping as train.py.
        groups = [
            ["SN002", "SN010", "SN001", "SN026", "SN027", "SN032", "SN030", "SN009", "SN016"],
            ["SN013", "SN018", "SN011", "SN028", "SN012", "SN006", "SN031", "SN021", "SN024"],
            ["SN003", "SN029", "SN023", "SN025", "SN008", "SN005", "SN007", "SN017", "SN004"],
        ]
        held_out = {1: 2, 2: 1, 3: 0}[args.fold]
        subjects = sum([group for i, group in enumerate(groups) if i != held_out], [])
        train_data = DISFADataset(args.data_root, subjects=subjects, train=True, fold=args.fold)
        val_data = DISFADataset(args.data_root, subjects=groups[held_out], train=False, fold=args.fold)
    else:
        train_data = BP4DDataset(args.data_root, train=True, fold=args.fold)
        val_data = BP4DDataset(args.data_root, train=False, fold=args.fold)
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation splits must both be nonempty")
    print(f"Training samples (class-weight verification only): {len(train_data)}; validation: {len(val_data)}")
    weights, _ = train_data.calculate_class_weights()
    cfg = ModelConfig()
    cfg.phase, cfg.dataset_name = args.phase, args.dataset
    cfg.au_ids = get_dataset_aus(args.dataset)
    cfg.au_expression_prior = get_dataset_au_expression_prior(args.dataset)
    cfg.enable_test_time_refinement = False
    model = CtrlAUModel(cfg).to(device)
    model.update_class_weights(weights.to(device))
    payload = load_training_checkpoint(
        model, str(args.checkpoint), phase=args.phase, fold=args.fold,
        dataset=args.dataset, model_name="ctrlau", device="cpu", resume_phase=args.phase,
    )
    print(f"Checkpoint: {args.checkpoint}; phase={payload['phase']}; epoch={payload['epoch']}")
    recorded_f1 = payload.get("f1")
    del payload, train_data
    model.requires_grad_(False)
    loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    means = print_results(evaluate_k_grid(model, loader, device, k_values), cfg.au_ids)
    if recorded_f1 is not None:
        print(f"K=0 vs recorded checkpoint F1: {100 * (means[0] - recorded_f1):+.4f} percentage points")


if __name__ == "__main__":
    main()
