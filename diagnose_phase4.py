"""Read-only phase-4 diagnostics: validation, BN-buffer ablation and gradients.

No optimizer steps, checkpoint writes, threshold tuning or synthetic features.
Temporary in-memory changes to module modes and buffers are always restored.
"""
import argparse
from contextlib import contextmanager
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ModelConfig, get_dataset_aus, get_dataset_au_expression_prior
from dataset import DISFADataset, BP4DDataset
from models.ctrlau import CtrlAUModel
from train import configure_training_phase, evaluate, keep_frozen_modules_eval, load_training_checkpoint


@contextmanager
def preserve_runtime_state(model):
    modes = {module: module.training for module in model.modules()}
    buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
    try:
        yield
    finally:
        with torch.no_grad():
            for name, tensor in model.named_buffers():
                tensor.copy_(buffers[name])
        for module, training in modes.items():
            module.training = training


def bn_buffers(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            names.update(f"{name}.{key}" for key, _ in module.named_buffers(recurse=False))
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers() if name in names}


def gradient_metrics(task, cf):
    task_sq = sum(g.float().square().sum().item() for g in task if g is not None)
    cf_sq = sum(g.float().square().sum().item() for g in cf if g is not None)
    dot = sum((a.float() * b.float()).sum().item() for a, b in zip(task, cf)
              if a is not None and b is not None)
    task_norm, cf_norm = task_sq ** .5, cf_sq ** .5
    # An unused parameter contributes no derivative; undefined ratios/cosines
    # are reported explicitly, never replaced by a fabricated measurement.
    return {
        "task_norm": task_norm, "cf_norm": cf_norm,
        "cf/task": cf_norm / task_norm if task_norm else None,
        "cosine": dot / (task_norm * cf_norm) if task_norm and cf_norm else None,
        "task_unused": sum(g is None for g in task),
        "cf_unused": sum(g is None for g in cf),
    }


def probe_gradients(model, batches, device, seed):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = {"CNN": [], "feature_extractors": [], "graphs_and_graph_heads": []}
    for i, (name, _) in enumerate(named):
        group = "CNN" if name.startswith("backbone.") else (
            "feature_extractors" if name.startswith(("global_linear.", "au_head.", "emotion_head."))
            else "graphs_and_graph_heads"
        )
        groups[group].append(i)
    parameters = tuple(p for _, p in named)
    cfg = model.cfg
    rows = []
    for index, batch in enumerate(batches, 1):
        # Match dropout masks between checkpoints, without accumulating BN drift.
        torch.manual_seed(seed + index)
        torch.cuda.manual_seed_all(seed + index)
        with preserve_runtime_state(model):
            model.train()
            keep_frozen_modules_eval(model)
            output = model(batch["image"].to(device), au_labels=batch["au_labels"].to(device))
            losses = output["losses"]
            task = (cfg.lambda_au_au * losses["loss_au_au"]
                    + cfg.lambda_graph_au * losses["loss_graph_au"]
                    + cfg.lambda_graph_emo * losses["loss_graph_emo"])
            cf = sum(getattr(cfg, weight) * losses[loss] for weight, loss in (
                ("lambda_cf_au_imp", "loss_cf_au_imp"),
                ("lambda_cf_au_unimp", "loss_cf_au_unimp"),
                ("lambda_cf_exp_imp", "loss_cf_exp_imp"),
                ("lambda_cf_exp_unimp", "loss_cf_exp_unimp"),
            ))
            if not torch.isfinite(losses["total_loss"]):
                raise RuntimeError("Non-finite diagnostic loss")
            other = losses["total_loss"] - task - cf
            print(f"Batch {index}: weighted task={task.item():.6f}, CF={cf.item():.6f}, other={other.item():.6f}")
            task_grads = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
            cf_grads = torch.autograd.grad(cf, parameters, allow_unused=True)
            for name, indices in groups.items():
                selected = ([task_grads[i] for i in indices], [cf_grads[i] for i in indices])
                if any(not torch.isfinite(g).all() for grads in selected for g in grads if g is not None):
                    raise RuntimeError(f"Non-finite gradients in {name}")
                metrics = gradient_metrics(*selected)
                values = ", ".join(f"{key}={value:.6g}" if value is not None else f"{key}=undefined (zero norm)"
                                   for key, value in metrics.items())
                print(f"  {name}: {values}")
                rows.append({"batch": index, "group": name, **metrics})
    return rows


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--phase3", required=True, type=Path)
    parser.add_argument("--phase4", required=True, type=Path)
    parser.add_argument("--dataset", choices=("disfa", "bp4d"), default="disfa")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--batch_size", type=positive_int, default=64, help="validation batch size")
    parser.add_argument("--gradient_batch_size", type=positive_int, default=64,
                        help="match training batch size for matched train-mode BN statistics")
    parser.add_argument("--gradient_batches", type=positive_int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cf-scale", type=float, default=1., help="use the value from your training command")
    args = parser.parse_args()
    for path in (args.phase3, args.phase4):
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
    if args.workers < 0 or not np.isfinite(args.cf_scale) or not 0 <= args.cf_scale <= 1:
        parser.error("workers must be nonnegative; cf-scale must be finite and in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; three full validation passes, then {args.gradient_batches} probes per checkpoint.")
    print(f"Gradient batch size: {args.gradient_batch_size}; CF attenuation scale: {args.cf_scale}")

    # Exact subject order and fold mapping used by train.py.
    groups = [
        ["SN002", "SN010", "SN001", "SN026", "SN027", "SN032", "SN030", "SN009", "SN016"],
        ["SN013", "SN018", "SN011", "SN028", "SN012", "SN006", "SN031", "SN021", "SN024"],
        ["SN003", "SN029", "SN023", "SN025", "SN008", "SN005", "SN007", "SN017", "SN004"],
    ]
    if args.dataset == "disfa":
        held_out = {1: 2, 2: 1, 3: 0}[args.fold]
        subjects = sum([group for i, group in enumerate(groups) if i != held_out], [])
        train_data = DISFADataset(args.data_root, subjects=subjects, train=True, fold=args.fold)
        val_data = DISFADataset(args.data_root, subjects=groups[held_out], train=False, fold=args.fold)
    else:
        train_data = BP4DDataset(args.data_root, train=True, fold=args.fold)
        val_data = BP4DDataset(args.data_root, train=False, fold=args.fold)
    if not len(train_data) or not len(val_data):
        raise ValueError("Both train and validation splits must be nonempty")
    print(f"Train samples: {len(train_data)}; validation samples: {len(val_data)}")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    # Cache only the sampled training images, so every checkpoint sees identical
    # subjects, augmentation, labels and batch grouping. No validation labels train anything.
    probe_loader = DataLoader(train_data, batch_size=args.gradient_batch_size, shuffle=True,
                             num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    batches = []
    for batch in probe_loader:
        batches.append({"image": batch["image"], "au_labels": batch["au_labels"]})
        if len(batches) == args.gradient_batches:
            break
    if len(batches) != args.gradient_batches:
        raise ValueError("Requested more gradient batches than the training split contains")

    cfg = ModelConfig()
    cfg.phase, cfg.dataset_name = 4, args.dataset
    cfg.au_ids = get_dataset_aus(args.dataset)
    cfg.au_expression_prior = get_dataset_au_expression_prior(args.dataset)
    cfg.cf_intervention_scale = args.cf_scale
    if getattr(cfg, "enable_test_time_refinement", False):
        raise ValueError("Disable test-time refinement for direct graph-prediction diagnostics")
    model = CtrlAUModel(cfg).to(device)
    weights, _ = train_data.calculate_class_weights()
    model.update_class_weights(weights.to(device))
    phase_args = SimpleNamespace(phase=4, model="ctrlau")
    configure_training_phase(model, phase_args)

    def load(path, source_phase):
        payload = load_training_checkpoint(model, str(path), phase=4, fold=args.fold,
                    dataset=args.dataset, model_name="ctrlau", device="cpu", resume_phase=source_phase)
        model._cached_text_emb = None
        model._cached_emotion_text_emb = None
        print(f"\nCheckpoint: {path}")
        print({key: payload[key] for key in ("phase", "epoch", "fold", "dataset", "f1", "best_f1")})
        # Drop optimizer/state payloads immediately; the model is already loaded.
        return float(payload["f1"])

    saved3 = load(args.phase3, 3)
    phase3_bn = bn_buffers(model)
    model.eval()
    images = batches[0]["image"].to(device)
    with torch.no_grad():
        model.set_phase(3)
        native = model(images)["graph_au_probs"]
        model.set_phase(4)
        zero_update = model(images)["graph_au_probs"]
    difference = (native - zero_update).abs().max().item()
    print(f"Phase-3 vs phase-4 mode, same weights/images, max AU probability difference: {difference:.9g}")
    if not torch.allclose(native, zero_update, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Zero-update phase transition changes inference; investigate before fine-tuning")
    print("\n=== A: Phase-3 weights evaluated as phase 4, zero updates ===")
    baseline = evaluate(model, val_loader, device, cfg.au_ids, phase=4)
    print(f"Compared with recorded checkpoint F1: {(baseline - saved3) * 100:+.4f} percentage points")
    print("\n=== D1: Weighted gradient comparison, phase-3 starting weights ===")
    probe_gradients(model, batches, device, args.seed)

    saved4 = load(args.phase4, 4)
    print("\n=== B: Phase-4 checkpoint, unchanged ===")
    actual = evaluate(model, val_loader, device, cfg.au_ids, phase=4)
    print(f"Compared with recorded checkpoint F1: {(actual - saved4) * 100:+.4f} percentage points")
    current_bn = bn_buffers(model)
    if current_bn.keys() != phase3_bn.keys():
        raise RuntimeError("Checkpoint BatchNorm buffer sets do not match")
    drift = sorted(((name, (tensor.float() - phase3_bn[name].float()).square().mean().sqrt().item())
                    for name, tensor in current_bn.items() if name.endswith("running_mean")),
                   key=lambda item: item[1], reverse=True)
    print("Largest BN running-mean RMS changes:")
    for name, change in drift[:5]:
        print(f"  {name}: {change:.6g}")
    print("\n=== C: Phase-4 weights, ONLY running BN buffers restored from phase 3 ===")
    with preserve_runtime_state(model), torch.no_grad():
        for name, tensor in model.named_buffers():
            if name in phase3_bn:
                tensor.copy_(phase3_bn[name].to(tensor))
        restored = evaluate(model, val_loader, device, cfg.au_ids, phase=4)
    print("\n=== D2: Weighted gradient comparison, unchanged phase-4 checkpoint ===")
    probe_gradients(model, batches, device, args.seed)
    print("\n=== SUMMARY ===")
    print(f"A zero-update baseline: {baseline:.6f}")
    print(f"B phase-4 checkpoint:  {actual:.6f} ({(actual-baseline)*100:+.3f} pp vs A)")
    print(f"C phase-4 + old BN:    {restored:.6f} ({(restored-actual)*100:+.3f} pp vs B)")
    print("C changes inference buffers only; it does not reconstruct training with frozen BN.")
    print("Gradient cosine < 0 means local objective conflict on that batch/group.")
    print("CF/task compares WEIGHTED gradient norms; these probes do not establish a dataset-wide cause.")
    print("No optimizer steps taken and no checkpoints written.")


if __name__ == "__main__":
    main()
