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


@contextmanager
def temporary_model_parts(model, reference, keys):
    """Swap only explicitly selected state tensors; restore even on failure."""
    current = model.state_dict()
    keys = tuple(keys)
    if not keys or any(key not in current or key not in reference for key in keys):
        raise ValueError("Selected checkpoint component is empty or has missing tensors")
    if any(current[key].shape != reference[key].shape or current[key].dtype != reference[key].dtype for key in keys):
        raise ValueError("Checkpoint component shape/dtype mismatch")
    original = {key: current[key].detach().cpu().clone() for key in keys}
    try:
        with torch.no_grad():
            for key in keys:
                current[key].copy_(reference[key])
        yield
    finally:
        with torch.no_grad():
            for key in keys:
                current[key].copy_(original[key])


def au_metrics(probabilities, labels):
    if probabilities.shape != labels.shape or probabilities.ndim != 2 or not len(labels):
        raise ValueError("AU predictions and labels must have the same nonempty (sample, AU) shape")
    if (not torch.isfinite(probabilities).all() or not torch.isfinite(labels).all()
            or torch.any((probabilities < 0) | (probabilities > 1))
            or torch.any((labels != 0) & (labels != 1))):
        raise ValueError("Expected finite probabilities in [0,1] and binary labels")
    truth, prediction = labels.bool(), probabilities > .5
    tp = (prediction & truth).sum(0).float()
    fp = (prediction & ~truth).sum(0).float()
    fn = (~prediction & truth).sum(0).float()
    # Same zero-denominator convention and strict >0.5 threshold as train.evaluate.
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp,
            "positive_rate": prediction.float().mean(0), "true_rate": truth.float().mean(0)}


@torch.no_grad()
def localization_evaluation(model, loader, device):
    model.eval()
    outputs = {"base": [], "AU_graph": [], "final_graph": []}
    labels, subjects = [], []
    subject_metadata = None
    for batch in loader:
        result = model(batch["image"].to(device))
        for name, key in (("base", "au_probs"), ("AU_graph", "au_au_probs"), ("final_graph", "graph_au_probs")):
            outputs[name].append(result[key].cpu())
        labels.append(batch["au_labels"].cpu())
        present = "subject" in batch
        if subject_metadata is not None and present != subject_metadata:
            raise ValueError("Subject metadata is inconsistently provided across batches")
        subject_metadata = present
        if present:
            subjects.extend(batch["subject"])
    if not labels:
        raise ValueError("Empty validation loader")
    labels = torch.cat(labels)
    outputs = {name: torch.cat(values) for name, values in outputs.items()}
    if subject_metadata and len(subjects) != len(labels):
        raise ValueError("Subject metadata/sample count mismatch")
    return {
        "labels": labels, "subjects": subjects if subject_metadata else None,
        "final_probabilities": outputs["final_graph"],
        "heads": {name: au_metrics(values, labels)["f1"].mean().item() for name, values in outputs.items()},
    }


def localize_regression(model, loader, device, au_ids, load_phase3, load_phase4):
    """Five full validation passes; no gradients, optimization or threshold tuning."""
    load_phase3()
    reference = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print("\n=== LOCALIZE A: Original phase-3 checkpoint ===")
    baseline = localization_evaluation(model, loader, device)
    load_phase4()
    print("\n=== LOCALIZE B: Original phase-4 checkpoint ===")
    changed = localization_evaluation(model, loader, device)
    if not torch.equal(baseline["labels"], changed["labels"]) or baseline["subjects"] != changed["subjects"]:
        raise RuntimeError("Validation sample order/labels changed between passes")
    results = [("A: phase 3", baseline), ("B: phase 4", changed)]
    state_keys = tuple(reference)
    bn_keys = tuple(bn_buffers(model))
    visual_keys = tuple(key for key in state_keys if key.startswith(
        ("backbone.", "global_linear.", "au_head.", "emotion_head.")))
    graph_keys = tuple(key for key in state_keys if key.startswith(
        ("graph_module.", "au_au_classifiers.", "graph_au_classifiers.",
         "graph_emo_classifiers.", "mask_module.", "cf_threshold")))
    for title, keys in (
        ("C: phase 4 + phase-3 BN buffers only", bn_keys),
        ("D: phase-3 visual modules + phase-4 graphs", visual_keys),
        ("E: phase-4 visual modules + phase-3 graphs", graph_keys),
    ):
        print(f"\n=== LOCALIZE {title} ===")
        with preserve_runtime_state(model), temporary_model_parts(model, reference, keys):
            result = localization_evaluation(model, loader, device)
        if not torch.equal(result["labels"], baseline["labels"]):
            raise RuntimeError("Validation labels changed during component comparison")
        results.append((title, result))
        print(f"Final graph macro F1: {result['heads']['final_graph']:.6f}")
    print("\n=== LOCALIZATION SUMMARY (F1 percentages, fixed threshold 0.5) ===")
    print("case | base AU heads | intermediate AU graph | final graph")
    for title, result in results:
        print(title + " | " + " | ".join(f"{result['heads'][name]*100:.3f}" for name in ("base", "AU_graph", "final_graph")))
    print("\n=== FINAL-GRAPH PER-AU CHANGE: phase 3 -> phase 4 ===")
    before = au_metrics(baseline["final_probabilities"], baseline["labels"])
    after = au_metrics(changed["final_probabilities"], changed["labels"])
    for i, au in enumerate(au_ids):
        row = [f"{name}={before[name][i].item():.4f}->{after[name][i].item():.4f}"
               for name in ("f1", "precision", "recall", "positive_rate")]
        print(f"AU{au}: " + ", ".join(row) + f", true_positive_rate={before['true_rate'][i].item():.4f}")
    if baseline["subjects"] is None:
        print("Per-subject analysis unavailable: dataset does not provide subject IDs; no IDs inferred.")
    else:
        rows = []
        for subject in sorted(set(baseline["subjects"])):
            selected = torch.tensor([item == subject for item in baseline["subjects"]])
            a = au_metrics(baseline["final_probabilities"][selected], baseline["labels"][selected])
            b = au_metrics(changed["final_probabilities"][selected], changed["labels"][selected])
            for i, au in enumerate(au_ids):
                rows.append((b["f1"][i].item() - a["f1"][i].item(), subject, au,
                             a["f1"][i].item(), b["f1"][i].item(), int(a["fp"][i]), int(b["fp"][i]),
                             int(selected.sum()), int(baseline["labels"][selected, i].sum())))
        print("\n=== 12 LARGEST SUBJECT/AU F1 DROPS (percentage points) ===")
        for delta, subject, au, a, b, fp_a, fp_b, count, positives in sorted(rows)[:12]:
            print(f"{subject} AU{au}: F1={a:.4f}->{b:.4f}, delta={delta*100:+.3f} pp, "
                  f"FP={fp_a}->{fp_b}, samples={count}, GT positives={positives}")
    print("Hybrids localize inference sensitivity, not the cause of the optimization trajectory.")
    print("Visual swaps include visual BN buffers/affine parameters; C swaps running BN buffers only.")
    print("Hybrid feature/graph compatibility can affect scores; do not treat them as trained models.")
    print("No training policy changed, no optimizer steps, no checkpoints written.")
    return results


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


def inspect_finetune_start(model, training_batches, heldout_batches, device, seed):
    """Inspect weighted objective directions at fixed weights; never take a step."""
    if len(training_batches) != len(heldout_batches) or not training_batches:
        raise ValueError("Need the same positive number of training and held-out batches")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = tuple(p for _, p in named)
    groups = {"CNN": [], "feature_extractors": [], "graphs": []}
    for i, (name, _) in enumerate(named):
        group = "CNN" if name.startswith("backbone.") else (
            "feature_extractors" if name.startswith(("global_linear.", "au_head.", "emotion_head.")) else "graphs")
        groups[group].append(i)
    cfg, records = model.cfg, []
    for index, (training, heldout) in enumerate(zip(training_batches, heldout_batches), 1):
        with preserve_runtime_state(model):
            model.eval()
            reference_output = model(heldout["image"].to(device))
            validation_loss = model.wal_loss(reference_output["graph_au_probs"], heldout["au_labels"].to(device).float())
            if not torch.isfinite(validation_loss):
                raise RuntimeError("Non-finite held-out AU loss")
            print(f"\nPair {index}: held-out final-AU WAL={validation_loss.item():.6f}")
            reference_gradients = tuple(None if g is None else g.detach().cpu() for g in torch.autograd.grad(
                validation_loss, parameters, allow_unused=True))
            if any(not torch.isfinite(g).all() for g in reference_gradients if g is not None):
                raise RuntimeError("Non-finite held-out AU gradient")
            with torch.no_grad():
                eval_prediction = model(training["image"].to(device))["graph_au_probs"]
            del reference_output, validation_loss
        for mode in ("original_BN_train", "temporary_BN_eval"):
            torch.manual_seed(seed + index)
            torch.cuda.manual_seed_all(seed + index)
            with preserve_runtime_state(model):
                model.train()
                keep_frozen_modules_eval(model)
                if mode == "temporary_BN_eval":
                    for module in model.modules():
                        if isinstance(module, nn.modules.batchnorm._BatchNorm):
                            module.eval()
                output = model(training["image"].to(device), au_labels=training["au_labels"].to(device))
                losses = output["losses"]
                parts = {
                    "AU": cfg.lambda_au_au * losses["loss_au_au"] + cfg.lambda_graph_au * losses["loss_graph_au"],
                    "expression": cfg.lambda_graph_emo * losses["loss_graph_emo"],
                    "CF": sum(getattr(cfg, weight) * losses[loss] for weight, loss in (
                        ("lambda_cf_au_imp", "loss_cf_au_imp"), ("lambda_cf_au_unimp", "loss_cf_au_unimp"),
                        ("lambda_cf_exp_imp", "loss_cf_exp_imp"), ("lambda_cf_exp_unimp", "loss_cf_exp_unimp"))),
                    "other": (cfg.lambda_dag * losses["loss_dag"]
                              + cfg.lambda_causal_au * losses["loss_causal_au"]
                              + cfg.lambda_causal_exp * losses["loss_causal_exp"]
                              + cfg.lambda_facs_au * losses["loss_facs_au"]),
                }
                torch.testing.assert_close(sum(parts.values()), losses["total_loss"])
                prediction_mse = (output["graph_au_probs"].detach() - eval_prediction).square().mean().item()
                print(f"  {mode}: prediction MSE vs eval={prediction_mse:.6g}; "
                      + ", ".join(f"{name}={loss.item():.6f}" for name, loss in parts.items()))
                combined = {"full": [None] * len(parameters), "without_CF_vector": [None] * len(parameters)}
                for component, loss in parts.items():
                    gradients = tuple(None if g is None else g.detach().cpu() for g in torch.autograd.grad(
                        loss, parameters, retain_graph=True, allow_unused=True))
                    if any(not torch.isfinite(g).all() for g in gradients if g is not None):
                        raise RuntimeError(f"Non-finite {component} gradient")
                    for title, accumulator in combined.items():
                        if title == "without_CF_vector" and component == "CF":
                            continue
                        for i, gradient in enumerate(gradients):
                            if gradient is not None:
                                if accumulator[i] is None:
                                    accumulator[i] = gradient.clone()
                                else:
                                    accumulator[i].add_(gradient)
                    for group, indices in groups.items():
                        metrics = gradient_metrics([reference_gradients[i] for i in indices],
                                                   [gradients[i] for i in indices])
                        records.append((mode, component, group, metrics["cf_norm"], metrics["cosine"]))
                for component, gradients in combined.items():
                    for group, indices in groups.items():
                        metrics = gradient_metrics([reference_gradients[i] for i in indices],
                                                   [gradients[i] for i in indices])
                        records.append((mode, component, group, metrics["cf_norm"], metrics["cosine"]))
                # Drop the final scalar too: retain_graph=True otherwise keeps
                # its CNN activations alive while the next BN-mode pass starts.
                del output, losses, parts, loss, gradients, combined, accumulator
    print("\n=== STARTING-POINT GRADIENT SUMMARY ===")
    print("BN mode | objective | group | mean weighted norm | mean cosine with held-out AU gradient | negative pairs")
    for mode in ("original_BN_train", "temporary_BN_eval"):
        for component in ("AU", "expression", "CF", "other", "full", "without_CF_vector"):
            for group in groups:
                rows = [r for r in records if r[:3] == (mode, component, group)]
                cosines = [r[4] for r in rows if r[4] is not None]
                cosine_text = f"{sum(cosines)/len(cosines):+.6f}" if cosines else "undefined (zero norm)"
                print(f"{mode} | {component} | {group} | {sum(r[3] for r in rows)/len(rows):.6g} | "
                      f"{cosine_text} | {sum(c < 0 for c in cosines)}/{len(cosines)}")
    print("Positive cosine: gradient descent on that training objective locally reduces sampled held-out AU loss.")
    print("Negative cosine: local conflict. This is a first-order gradient test, not an AdamW update or F1 forecast.")
    print("without_CF_vector subtracts the CF derivative only; it does NOT run training with CF disabled.")
    print("BN mode changes are temporary; all running buffers and module modes are restored.")
    print("Held-out gradients are measured only: never applied, optimized or used to select settings automatically.")
    print("No optimizer created, no optimizer steps, no checkpoints written. Training code remains unchanged.")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--phase3", required=True, type=Path)
    parser.add_argument("--phase4", type=Path)
    parser.add_argument("--dataset", choices=("disfa", "bp4d"), default="disfa")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--batch_size", type=positive_int, default=64, help="validation batch size")
    parser.add_argument("--gradient_batch_size", type=positive_int, default=64,
                        help="match training batch size for matched train-mode BN statistics")
    parser.add_argument("--gradient_batches", type=positive_int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cf-scale", type=float, default=1., help="use the value from your training command")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--localize", action="store_true",
                       help="instead run five validation passes comparing visual/graph components and subject errors")
    modes.add_argument("--inspect-start", action="store_true",
                       help="inspect phase-3 weights only: objective/held-out gradient alignment and temporary BN modes")
    args = parser.parse_args()
    if not args.inspect_start and args.phase4 is None:
        parser.error("--phase4 is required unless using --inspect-start")
    for path in (p for p in (args.phase3, args.phase4) if p is not None):
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
    if args.inspect_start:
        print(f"Device: {device}; phase-3-only inspection on {args.gradient_batches} sampled batch pairs; no training.")
        print(f"Batch size: {args.gradient_batch_size}; CF scale: {args.cf_scale}")
    elif args.localize:
        print(f"Device: {device}; five full validation passes, no gradients or training.")
    else:
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
    if not args.localize:
        for batch in probe_loader:
            batches.append({"image": batch["image"], "au_labels": batch["au_labels"]})
            if len(batches) == args.gradient_batches:
                break
    if not args.localize and len(batches) != args.gradient_batches:
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

    if args.inspect_start:
        load(args.phase3, 3)
        heldout_loader = DataLoader(val_data, batch_size=args.gradient_batch_size, shuffle=True,
                                   num_workers=0, generator=torch.Generator().manual_seed(args.seed + 1))
        heldout_batches = []
        for batch in heldout_loader:
            heldout_batches.append({"image": batch["image"], "au_labels": batch["au_labels"]})
            if len(heldout_batches) == args.gradient_batches:
                break
        if len(heldout_batches) != args.gradient_batches:
            raise ValueError("Not enough held-out batches for requested inspection")
        inspect_finetune_start(model, batches, heldout_batches, device, args.seed)
        return

    if args.localize:
        localize_regression(model, val_loader, device, cfg.au_ids,
                            lambda: load(args.phase3, 3), lambda: load(args.phase4, 4))
        return

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
