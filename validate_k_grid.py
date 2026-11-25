"""Validate a CtrlAU graph checkpoint across refinement iteration counts.

K=0 is always included. Read each validation image and run the visual/graph
network once. Refinement snapshots share the exact Adam trajectory from the
model's existing ten-step loop. Only copied logits are optimized, never model
parameters. No validation labels enter prediction or refinement.

Optionally exports visual diagnostics (F1/flip curves, per-AU flip breakdowns,
a Fixed-vs-Broken face gallery with per-sample K trajectories, transition
trajectories, CSV of changed samples, and JSON summary) to --viz_dir.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import (
    CTRL_EMOTIONS,
    ModelConfig,
    get_dataset_au_expression_prior,
    get_dataset_aus,
)
from dataset import BP4DDataset, DISFADataset
from diagnose_phase4 import au_metrics
from losses import FACSAUViolationLoss
from models.ctrlau import CtrlAUModel
from train import load_training_checkpoint


# Exact FACS anatomical rules mirrored from losses.FACSAUViolationLoss.
FACS_IMPLICATION_RULES = (
    (26, 25),  # Jaw Drop -> Lips Part
    (27, 25),  # Mouth Stretch -> Lips Part
    (27, 26),  # Mouth Stretch -> Jaw Drop
)
FACS_EXCLUSION_RULES = (
    (24, 25),
    (45, 46),
    (41, 42),
    (41, 43),
    (42, 43),
    (51, 52),
    (53, 54),
    (55, 56),
    (61, 62),
    (63, 64),
)

TRANSITION_ORDER = ("FN->TP", "FP->TN", "TN->FP", "TP->FN")
TRANSITION_COLORS = {
    "FN->TP": "#1B8A5A",  # Fixed: missed AU recovered
    "FP->TN": "#5BBF8A",  # Fixed: false alarm removed
    "TN->FP": "#E68A2E",  # Broken: new false alarm
    "TP->FN": "#C93B3B",  # Broken: true AU erased
}
TRANSITION_LABELS = {
    "FN->TP": "Fixed (FN->TP)",
    "FP->TN": "Fixed (FP->TN)",
    "TN->FP": "Broken (TN->FP)",
    "TP->FN": "Broken (TP->FN)",
}


@torch.no_grad()
def evaluate_k_grid(model, loader, device, k_values, return_details=False):
    k_values = tuple(sorted({0, *k_values}))
    if any(type(k) is not int or k < 0 for k in k_values):
        raise ValueError("K values must be non-negative integers")
    if getattr(model.cfg, "enable_test_time_refinement", False):
        raise ValueError("Disable automatic model refinement before running the K grid")
    model.eval()
    predictions = {k: [] for k in k_values}
    emotion_predictions = {k: [] for k in k_values} if return_details else None
    labels = []
    for batch_index, batch in enumerate(loader, 1):
        # Do not pass AU ground truth into the model or the refinement method.
        outputs = model(batch["image"].to(device, non_blocking=True))
        snapshots = model.refine_graph_predictions(
            outputs["graph_au_logits"], outputs["graph_emo_logits"], k_values
        )
        for k in k_values:
            predictions[k].append(snapshots[k][0].cpu())
            if return_details:
                emotion_predictions[k].append(snapshots[k][1].cpu())
        labels.append(batch["au_labels"].cpu())
        if batch_index % 100 == 0:
            print(f"Validation: {batch_index}/{len(loader)} batches; all K values evaluated", flush=True)
    if not labels:
        raise ValueError("Validation split is empty")
    labels = torch.cat(labels)
    cat_predictions = {k: torch.cat(predictions[k]) for k in k_values}
    results = {k: au_metrics(cat_predictions[k], labels) for k in k_values}
    if not return_details:
        return results
    cat_emotions = {k: torch.cat(emotion_predictions[k]) for k in k_values}
    return results, {
        "predictions": cat_predictions,
        "emotion_predictions": cat_emotions,
        "labels": labels,
    }


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


def _active_facs_partners(au_ids):
    """Map each AU id to other active AUs connected to it by a FACS rule."""
    active = set(int(au) for au in au_ids)
    partners = {int(au): [] for au in au_ids}
    for src, dst in FACS_IMPLICATION_RULES + FACS_EXCLUSION_RULES:
        if src in active and dst in active:
            if dst not in partners[src]:
                partners[src].append(dst)
            if src not in partners[dst]:
                partners[dst].append(src)
    return partners


def _hard_facs_violation_stats(probs, au_ids):
    """Count binary (> 0.5) violations across all active FACS rules."""
    pred = probs > 0.5
    index = {int(au): i for i, au in enumerate(au_ids)}
    total_violations = 0
    rule_counts = {}
    for src, dst in FACS_IMPLICATION_RULES:
        if src in index and dst in index:
            count = int((pred[:, index[src]] & (~pred[:, index[dst]])).sum().item())
            rule_counts[f"AU{src}=>AU{dst}"] = count
            total_violations += count
    for a, b in FACS_EXCLUSION_RULES:
        if a in index and b in index:
            count = int((pred[:, index[a]] & pred[:, index[b]]).sum().item())
            rule_counts[f"AU{a}^AU{b}"] = count
            total_violations += count
    return total_violations, rule_counts


def _resolve_sample_info(dataset, idx):
    """Resolve human-readable ID, subject group, frame number, and disk path."""
    if dataset is None or not hasattr(dataset, "samples"):
        return {
            "sample_id": f"sample_{idx:05d}",
            "group_key": "all",
            "frame_num": idx,
            "image_path": None,
        }
    samples = dataset.samples
    if idx < 0 or idx >= len(samples):
        return {
            "sample_id": f"sample_{idx:05d}",
            "group_key": "all",
            "frame_num": idx,
            "image_path": None,
        }
    entry = samples[idx]
    if isinstance(entry, tuple) and len(entry) == 3:
        subj, frame_num, img_path = entry
        return {
            "sample_id": f"{subj} #{frame_num}",
            "group_key": str(subj),
            "frame_num": int(frame_num),
            "image_path": str(img_path),
        }
    if isinstance(entry, tuple) and len(entry) == 2:
        rel_path, full_path = entry
        rel = Path(str(rel_path))
        group = rel.parent.as_posix() if rel.parent.as_posix() not in ("", ".") else rel.stem
        digits = re.findall(r"\d+", rel.stem)
        frame_num = int(digits[-1]) if digits else idx
        return {
            "sample_id": str(rel_path),
            "group_key": group,
            "frame_num": frame_num,
            "image_path": str(full_path),
        }
    return {
        "sample_id": f"sample_{idx:05d}",
        "group_key": "all",
        "frame_num": idx,
        "image_path": None,
    }


def _format_active_aus(binary_row, au_ids):
    return [int(au_ids[i]) for i, flag in enumerate(binary_row) if bool(flag)]


def analyze_k_changes(
    predictions,
    labels,
    au_ids,
    compare_k=None,
    dataset=None,
    emotion_predictions=None,
    emotion_names=None,
):
    """Quantify flips, FACS rule violations, probability drift, and changed samples."""
    k_values = sorted(predictions.keys())
    if 0 not in predictions:
        raise ValueError("K=0 predictions are required for change analysis")
    au_ids = tuple(int(au) for au in au_ids)
    num_samples, num_aus = labels.shape
    truth = labels.bool()
    p0 = predictions[0]
    pred0 = p0 > 0.5
    facs_loss_fn = FACSAUViolationLoss(au_ids)
    partners = _active_facs_partners(au_ids)

    metrics = {k: au_metrics(predictions[k], labels) for k in k_values}
    means = {k: float(metrics[k]["f1"].mean().item()) for k in k_values}
    best_k = max(k_values, key=means.get)

    if compare_k is None:
        if best_k > 0:
            compare_k = best_k
        else:
            nonzero = [k for k in k_values if k > 0]
            compare_k = nonzero[-1] if nonzero else 0
    elif compare_k not in predictions:
        raise ValueError(f"compare_k={compare_k} not present in evaluated K values {k_values}")

    emo0_top = None
    if emotion_predictions is not None and 0 in emotion_predictions:
        emo0_top = emotion_predictions[0].argmax(dim=1)
        if emotion_names is None:
            emotion_names = tuple(CTRL_EMOTIONS[: emotion_predictions[0].shape[1]])

    per_k = {}
    for k in k_values:
        pk = predictions[k]
        predk = pk > 0.5
        delta = pk - p0
        abs_delta = delta.abs()

        fn_to_tp = truth & (~pred0) & predk
        fp_to_tn = (~truth) & pred0 & (~predk)
        tn_to_fp = (~truth) & (~pred0) & predk
        tp_to_fn = truth & pred0 & (~predk)
        flipped = pred0 ^ predk

        per_au_fn_tp = fn_to_tp.sum(dim=0).tolist()
        per_au_fp_tn = fp_to_tn.sum(dim=0).tolist()
        per_au_tn_fp = tn_to_fp.sum(dim=0).tolist()
        per_au_tp_fn = tp_to_fn.sum(dim=0).tolist()

        fixed = int((fn_to_tp | fp_to_tn).sum().item())
        broken = int((tn_to_fp | tp_to_fn).sum().item())
        hard_viol, rule_counts = _hard_facs_violation_stats(pk, au_ids)
        fuzzy_viol = float(facs_loss_fn(pk).item())

        nonzero_abs = abs_delta[abs_delta > 1e-7]
        p95_abs = float(torch.quantile(nonzero_abs, 0.95).item()) if nonzero_abs.numel() else 0.0

        entry = {
            "avg_f1": means[k],
            "delta_avg_f1_pp": 100.0 * (means[k] - means[0]),
            "per_au_f1": [float(v) for v in metrics[k]["f1"].tolist()],
            "per_au_delta_f1_pp": [
                100.0 * float((metrics[k]["f1"][i] - metrics[0]["f1"][i]).item())
                for i in range(num_aus)
            ],
            "fn_to_tp": int(fn_to_tp.sum().item()),
            "fp_to_tn": int(fp_to_tn.sum().item()),
            "tn_to_fp": int(tn_to_fp.sum().item()),
            "tp_to_fn": int(tp_to_fn.sum().item()),
            "fixed": fixed,
            "broken": broken,
            "total_flips": fixed + broken,
            "net_gain": fixed - broken,
            "samples_with_any_flip": int(flipped.any(dim=1).sum().item()),
            "per_au_fn_to_tp": [int(x) for x in per_au_fn_tp],
            "per_au_fp_to_tn": [int(x) for x in per_au_fp_tn],
            "per_au_tn_to_fp": [int(x) for x in per_au_tn_fp],
            "per_au_tp_to_fn": [int(x) for x in per_au_tp_fn],
            "per_au_net_gain": [
                int((per_au_fn_tp[i] + per_au_fp_tn[i]) - (per_au_tn_fp[i] + per_au_tp_fn[i]))
                for i in range(num_aus)
            ],
            "per_au_mean_signed_delta_p": [float(x) for x in delta.mean(dim=0).tolist()],
            "per_au_mean_abs_delta_p": [float(x) for x in abs_delta.mean(dim=0).tolist()],
            "per_au_max_abs_delta_p": [float(x) for x in abs_delta.max(dim=0).values.tolist()],
            "mean_abs_delta_p": float(abs_delta.mean().item()),
            "p95_active_abs_delta_p": p95_abs,
            "max_abs_delta_p": float(abs_delta.max().item()),
            "fuzzy_facs_violation": fuzzy_viol,
            "hard_facs_violations": hard_viol,
            "hard_facs_rule_counts": rule_counts,
        }
        if emo0_top is not None and k in emotion_predictions:
            emok = emotion_predictions[k]
            entry["expr_top1_flips"] = int((emok.argmax(dim=1) != emo0_top).sum().item())
            entry["expr_mean_abs_delta_p"] = float((emok - emotion_predictions[0]).abs().mean().item())
        per_k[k] = entry

    # Build per-event records for all flipped (sample, AU) pairs at compare_k.
    p_cmp = predictions[compare_k]
    pred_cmp = p_cmp > 0.5
    flipped_mask = pred0 ^ pred_cmp
    au_index_map = {au: i for i, au in enumerate(au_ids)}

    def _build_event(s_idx, a_idx):
        info = _resolve_sample_info(dataset, s_idx)
        gt_val = int(truth[s_idx, a_idx].item())
        b0 = bool(pred0[s_idx, a_idx].item())
        bk = bool(pred_cmp[s_idx, a_idx].item())
        if b0 != bk:
            if gt_val == 1 and (not b0) and bk:
                transition, outcome = "FN->TP", "fixed"
            elif gt_val == 0 and b0 and (not bk):
                transition, outcome = "FP->TN", "fixed"
            elif gt_val == 0 and (not b0) and bk:
                transition, outcome = "TN->FP", "broken"
            else:
                transition, outcome = "TP->FN", "broken"
        else:
            transition, outcome = "NO_FLIP", "unchanged"

        first_flip_k = None
        traj = {}
        for kv in k_values:
            val = float(predictions[kv][s_idx, a_idx].item())
            traj[kv] = val
            if first_flip_k is None and kv > 0 and ((val > 0.5) != b0):
                first_flip_k = kv

        au_num = au_ids[a_idx]
        partner_info = {}
        for partner_au in partners[au_num]:
            p_idx = au_index_map[partner_au]
            partner_info[partner_au] = {
                "gt": int(truth[s_idx, p_idx].item()),
                "p0": float(p0[s_idx, p_idx].item()),
                "pk": float(p_cmp[s_idx, p_idx].item()),
                "trajectory": {kv: float(predictions[kv][s_idx, p_idx].item()) for kv in k_values},
            }

        ev = {
            "sample_idx": s_idx,
            "sample_id": info["sample_id"],
            "group_key": info["group_key"],
            "frame_num": info["frame_num"],
            "image_path": info["image_path"],
            "au": au_num,
            "au_idx": a_idx,
            "transition": transition,
            "outcome": outcome,
            "gt": gt_val,
            "p0": float(p0[s_idx, a_idx].item()),
            "pk": float(p_cmp[s_idx, a_idx].item()),
            "delta_p": float((p_cmp[s_idx, a_idx] - p0[s_idx, a_idx]).item()),
            "first_flip_k": first_flip_k,
            "trajectory": traj,
            "gt_aus": _format_active_aus(truth[s_idx].tolist(), au_ids),
            "pred0_aus": _format_active_aus(pred0[s_idx].tolist(), au_ids),
            "predk_aus": _format_active_aus(pred_cmp[s_idx].tolist(), au_ids),
            "partner_aus": partner_info,
        }
        if emo0_top is not None and compare_k in emotion_predictions:
            e0 = int(emo0_top[s_idx].item())
            ek = int(emotion_predictions[compare_k][s_idx].argmax().item())
            ev["expr_k0"] = emotion_names[e0]
            ev["expr_k_cmp"] = emotion_names[ek]
        return ev

    flip_coords = flipped_mask.nonzero(as_tuple=False).tolist()
    compare_events = [_build_event(s_idx, a_idx) for s_idx, a_idx in flip_coords]
    compare_events.sort(
        key=lambda e: (
            0 if e["outcome"] == "fixed" else 1,
            -abs(e["delta_p"]),
            e["sample_idx"],
            e["au"],
        )
    )

    fallback_top_shifts = []
    if not compare_events and compare_k > 0:
        abs_diff = (p_cmp - p0).abs().reshape(-1)
        top_count = min(32, abs_diff.numel())
        top_vals, top_flat = torch.topk(abs_diff, k=top_count)
        for val, flat_idx in zip(top_vals.tolist(), top_flat.tolist()):
            if val <= 0.0:
                break
            s_idx, a_idx = divmod(flat_idx, num_aus)
            fallback_top_shifts.append(_build_event(s_idx, a_idx))

    return {
        "k_values": k_values,
        "au_ids": au_ids,
        "num_samples": num_samples,
        "metrics": metrics,
        "means": means,
        "best_k": best_k,
        "compare_k": compare_k,
        "per_k": per_k,
        "compare_events": compare_events,
        "fallback_top_shifts": fallback_top_shifts,
        "active_facs_partners": partners,
    }


def _select_diverse_events(events, max_count, min_frame_gap=15):
    """Select diverse high-impact examples across subjects, frames, and subtypes."""
    if not events or max_count <= 0:
        return []

    # Group by (subtype, group_key) sorted by descending |delta_p|
    by_subtype = {}
    for ev in events:
        by_subtype.setdefault(ev["transition"], []).append(ev)
    for ev_list in by_subtype.values():
        ev_list.sort(key=lambda e: (-abs(e["delta_p"]), e["sample_idx"]))

    # Interleave subtypes (e.g. FN->TP & FP->TN) so both are visible if present
    subtypes = sorted(by_subtype.keys())
    selected = []
    selected_ids = set()
    used_frames = {}  # group_key -> list[int]

    def _is_far_enough(ev):
        frames = used_frames.get(ev["group_key"], [])
        return all(abs(ev["frame_num"] - f) >= min_frame_gap for f in frames)

    # Pass 1: round-robin across subtypes and subject groups with temporal spacing
    group_queues = {}
    for st in subtypes:
        groups = {}
        for ev in by_subtype[st]:
            groups.setdefault(ev["group_key"], []).append(ev)
        group_queues[st] = [groups[g] for g in sorted(groups.keys(), key=lambda g: -abs(groups[g][0]["delta_p"]))]

    made_progress = True
    while len(selected) < max_count and made_progress:
        made_progress = False
        for st in subtypes:
            for q in group_queues[st]:
                if len(selected) >= max_count:
                    break
                while q:
                    cand = q.pop(0)
                    key = (cand["sample_idx"], cand["au"])
                    if key in selected_ids:
                        continue
                    if _is_far_enough(cand):
                        selected.append(cand)
                        selected_ids.add(key)
                        used_frames.setdefault(cand["group_key"], []).append(cand["frame_num"])
                        made_progress = True
                        break

    # Pass 2: if still short of max_count, relax temporal spacing and take highest |delta_p|
    if len(selected) < max_count:
        remaining = [
            ev for ev in sorted(events, key=lambda e: (-abs(e["delta_p"]), e["sample_idx"]))
            if (ev["sample_idx"], ev["au"]) not in selected_ids
        ]
        for cand in remaining:
            if len(selected) >= max_count:
                break
            selected.append(cand)
            selected_ids.add((cand["sample_idx"], cand["au"]))

    return selected


def print_change_report(analysis, au_ids, max_examples=5):
    """Print flip summary across K, per-AU transitions, and sample examples."""
    k_values = analysis["k_values"]
    per_k = analysis["per_k"]
    cmp_k = analysis["compare_k"]
    events = analysis["compare_events"]

    print("\n=== Refinement Change Summary vs K=0 ===")
    print(
        f"{'K':<5} {'Flips':>7} {'Fixed':>7} {'(FN->TP':>8} {'FP->TN)':>8} "
        f"{'Broken':>7} {'(TN->FP':>8} {'TP->FN)':>8} {'Net':>6} "
        f"{'HardFACS':>9} {'FuzzyFACS':>10} {'Mean|dp|':>9}"
    )
    for k in k_values:
        s = per_k[k]
        print(
            f"{k:<5d} {s['total_flips']:>7d} {s['fixed']:>7d} {s['fn_to_tp']:>8d} {s['fp_to_tn']:>8d} "
            f"{s['broken']:>7d} {s['tn_to_fp']:>8d} {s['tp_to_fn']:>8d} {s['net_gain']:>+6d} "
            f"{s['hard_facs_violations']:>9d} {s['fuzzy_facs_violation']:>10.5f} {s['mean_abs_delta_p']:>9.5f}"
        )

    cmp_stats = per_k[cmp_k]
    print(f"\n=== Per-AU Flip Breakdown at K={cmp_k} (vs K=0) ===")
    print(f"{'AU':<6} {'dF1(pp)':>9} {'FN->TP':>7} {'FP->TN':>7} {'TN->FP':>7} {'TP->FN':>7} {'Net':>6} {'Mean dp':>10} {'Max|dp|':>9}")
    for i, au in enumerate(au_ids):
        print(
            f"AU{au:<4d} {cmp_stats['per_au_delta_f1_pp'][i]:>+9.4f} "
            f"{cmp_stats['per_au_fn_to_tp'][i]:>7d} {cmp_stats['per_au_fp_to_tn'][i]:>7d} "
            f"{cmp_stats['per_au_tn_to_fp'][i]:>7d} {cmp_stats['per_au_tp_to_fn'][i]:>7d} "
            f"{cmp_stats['per_au_net_gain'][i]:>+6d} "
            f"{cmp_stats['per_au_mean_signed_delta_p'][i]:>+10.5f} "
            f"{cmp_stats['per_au_max_abs_delta_p'][i]:>9.5f}"
        )
    if cmp_stats["hard_facs_rule_counts"]:
        rules_0 = per_k[0]["hard_facs_rule_counts"]
        rules_k = cmp_stats["hard_facs_rule_counts"]
        rule_str = ", ".join(f"{name}: {rules_0[name]} -> {rules_k[name]}" for name in rules_k)
        print(f"Active binary FACS violations (K=0 -> K={cmp_k}): {rule_str}")
    if "expr_top1_flips" in cmp_stats:
        print(
            f"Expression head changes at K={cmp_k}: "
            f"{cmp_stats['expr_top1_flips']}/{analysis['num_samples']} top-1 flips, "
            f"mean |dp|={cmp_stats['expr_mean_abs_delta_p']:.5f}"
        )

    fixed_events = [e for e in events if e["outcome"] == "fixed"]
    broken_events = [e for e in events if e["outcome"] == "broken"]
    print(
        f"\nFlipped sample-AU pairs at K={cmp_k}: {len(events)} total "
        f"({len(fixed_events)} fixed, {len(broken_events)} broken, "
        f"across {cmp_stats['samples_with_any_flip']} unique images)"
    )

    def _print_examples(title, subset):
        if not subset:
            print(f"  {title}: none")
            return
        chosen = _select_diverse_events(subset, max_examples)
        print(f"  {title} (showing {len(chosen)} diverse examples of {len(subset)}):")
        for ev in chosen:
            partner_note = ""
            if ev["partner_aus"]:
                parts = [
                    f"AU{pau}:{info['p0']:.2f}->{info['pk']:.2f}(gt={info['gt']})"
                    for pau, info in ev["partner_aus"].items()
                ]
                partner_note = " | " + ", ".join(parts)
            print(
                f"    idx={ev['sample_idx']:<6d} {ev['sample_id']:<20} AU{ev['au']:<2d} "
                f"{ev['transition']:<6} gt={ev['gt']}  p:{ev['p0']:.3f}->{ev['pk']:.3f} "
                f"({ev['delta_p']:+.3f}, flip@K={ev['first_flip_k']}){partner_note}"
            )

    _print_examples(f"FIXED (Wrong at K=0 -> Right at K={cmp_k})", fixed_events)
    _print_examples(f"BROKEN (Right at K=0 -> Wrong at K={cmp_k})", broken_events)


def _load_cropped_face_image(image_path, size=224):
    """Load an image with the same Resize(256) + CenterCrop(224) framing as validation."""
    if not image_path or not Path(image_path).is_file():
        return None
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        short_side = min(w, h)
        if short_side > 0:
            scale = 256.0 / short_side
            new_w, new_h = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
        w, h = img.size
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        return img.crop((left, top, left + size, top + size))


def _save_curves_and_flips_plot(plt, analysis, out_path):
    k_values = analysis["k_values"]
    au_ids = analysis["au_ids"]
    per_k = analysis["per_k"]
    best_k = analysis["best_k"]
    x = np.arange(len(k_values))
    x_labels = [str(k) for k in k_values]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), dpi=160)
    fig.patch.set_facecolor("white")

    # Panel A: Average F1 (%) vs K
    ax = axes[0, 0]
    avg_f1 = [100.0 * per_k[k]["avg_f1"] for k in k_values]
    ax.plot(x, avg_f1, marker="o", linewidth=2.4, color="#2563EB", label="Average F1 (%)")
    ax.axhline(avg_f1[0], linestyle="--", linewidth=1.2, color="#64748B", label=f"K=0 baseline ({avg_f1[0]:.2f}%)")
    best_idx = k_values.index(best_k)
    ax.scatter([best_idx], [avg_f1[best_idx]], s=150, marker="*", color="#F59E0B", edgecolor="#1E293B", zorder=5,
               label=f"Best K={best_k} ({avg_f1[best_idx]:.2f}%, {avg_f1[best_idx] - avg_f1[0]:+.2f} pp)")
    for xi, yi in zip(x, avg_f1):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1E293B")
    f1_span = max(max(avg_f1) - min(avg_f1), 0.05)
    ax.set_ylim(min(avg_f1) - 0.15 * f1_span, max(avg_f1) + 0.30 * f1_span)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Refinement Steps (K)")
    ax.set_ylabel("Average F1 (%)")
    ax.set_title("A. Average Validation F1 vs. K", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.3, loc="lower right")

    # Panel B: Per-AU F1 Change (pp) vs K
    ax = axes[0, 1]
    palette = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0891B2", "#CA8A04", "#DB2777"]
    unchanged_aus = []
    color_idx = 0
    for i, au in enumerate(au_ids):
        deltas = [per_k[k]["per_au_delta_f1_pp"][i] for k in k_values]
        if max(abs(v) for v in deltas) > 1e-6:
            ax.plot(x, deltas, marker="o", linewidth=2.2, color=palette[color_idx % len(palette)],
                    label=f"AU{au} (end: {deltas[-1]:+.2f} pp)")
            color_idx += 1
        else:
            unchanged_aus.append(f"AU{au}")
    if unchanged_aus:
        ax.plot(x, np.zeros_like(x, dtype=float), linestyle="--", linewidth=1.2, color="#94A3B8",
                label=f"Unchanged: {', '.join(unchanged_aus)}")
    ax.axhline(0.0, color="#334155", linewidth=0.9, alpha=0.6)
    ax.margins(y=0.16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Refinement Steps (K)")
    ax.set_ylabel("F1 Change vs. K=0 (percentage points)")
    ax.set_title("B. Per-AU F1 Change Across K", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.2, loc="best")

    # Panel C: Stacked Flips (Fixed UP vs Broken DOWN) + Net Gain
    ax = axes[1, 0]
    fn_tp = np.array([per_k[k]["fn_to_tp"] for k in k_values], dtype=float)
    fp_tn = np.array([per_k[k]["fp_to_tn"] for k in k_values], dtype=float)
    tn_fp = np.array([per_k[k]["tn_to_fp"] for k in k_values], dtype=float)
    tp_fn = np.array([per_k[k]["tp_to_fn"] for k in k_values], dtype=float)
    net = (fn_tp + fp_tn) - (tn_fp + tp_fn)

    width = 0.58
    ax.bar(x, fn_tp, width=width, color=TRANSITION_COLORS["FN->TP"], label="Fixed: FN->TP (gained TP)")
    ax.bar(x, fp_tn, width=width, bottom=fn_tp, color=TRANSITION_COLORS["FP->TN"], label="Fixed: FP->TN (removed FP)")
    ax.bar(x, -tn_fp, width=width, color=TRANSITION_COLORS["TN->FP"], label="Broken: TN->FP (new FP)")
    ax.bar(x, -tp_fn, width=width, bottom=-tn_fp, color=TRANSITION_COLORS["TP->FN"], label="Broken: TP->FN (lost TP)")
    ax.plot(x, net, marker="D", markersize=5, linewidth=2.0, color="#0F172A", label="Net Gain (Fixed - Broken)")
    ax.axhline(0.0, color="#0F172A", linewidth=1.0)
    for xi, n_val in zip(x, net):
        if xi == 0:
            continue
        ax.annotate(f"{int(n_val):+d}", (xi, n_val), textcoords="offset points",
                    xytext=(0, 8 if n_val >= 0 else -14), ha="center", fontsize=8, fontweight="bold", color="#0F172A")
    max_pos = max(float((fn_tp + fp_tn).max(initial=0.0)), float(net.max(initial=0.0)), 1.0)
    max_neg = max(float((tn_fp + tp_fn).max(initial=0.0)), float((-net).max(initial=0.0)), 0.5)
    ax.set_ylim(-max_neg * 1.30, max_pos * 1.48)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Refinement Steps (K)")
    ax.set_ylabel("Flipped Predictions (+Fixed / -Broken)")
    ax.set_title("C. Prediction Flips vs. K=0 (Wrong->Right vs. Right->Wrong)", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.6, loc="upper left", ncol=2, framealpha=0.92)

    # Panel D: FACS Violations & Probability Drift vs K
    ax = axes[1, 1]
    hard_v = [per_k[k]["hard_facs_violations"] for k in k_values]
    fuzzy_v = [per_k[k]["fuzzy_facs_violation"] for k in k_values]
    mean_dp = [per_k[k]["mean_abs_delta_p"] for k in k_values]
    p95_dp = [per_k[k]["p95_active_abs_delta_p"] for k in k_values]

    line1 = ax.plot(x, hard_v, marker="s", linewidth=2.2, color="#7C3AED",
                    label=f"Hard FACS violations (fuzzy: {fuzzy_v[0]:.4f}->{fuzzy_v[best_idx]:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Refinement Steps (K)")
    ax.set_ylabel("Hard Binary FACS Violations (count)", color="#7C3AED")
    ax.tick_params(axis="y", labelcolor="#7C3AED")
    ax.margins(y=0.18)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax.twinx()
    line2 = ax2.plot(x, p95_dp, marker="^", linestyle="--", linewidth=1.9, color="#0D9488",
                     label="95th-pct |dp| (affected entries)")
    line3 = ax2.plot(x, mean_dp, marker="o", linestyle=":", linewidth=1.9, color="#EA580C",
                     label="Mean |p_K - p_0| (all AUs)")
    ax2.set_ylabel("Probability Shift |p_K - p_0|", color="#0F172A")
    ax2.margins(y=0.22)
    ax.set_title("D. FACS Violation Reduction & Probability Drift vs. K", fontweight="bold")
    lines = line1 + line2 + line3
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.8, loc="lower right", framealpha=0.92)

    fig.suptitle(
        f"CtrlAU Test-Time Refinement (RLR) Sweep — Best K={best_k} "
        f"(Avg F1: {100.0 * per_k[0]['avg_f1']:.2f}% -> {100.0 * per_k[best_k]['avg_f1']:.2f}%)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_per_au_breakdown_plot(plt, analysis, predictions, labels, out_path):
    au_ids = analysis["au_ids"]
    cmp_k = analysis["compare_k"]
    stats = analysis["per_k"][cmp_k]
    y_pos = np.arange(len(au_ids))

    fig, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=160, gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor("white")

    # Left: Diverging horizontal stacked bar chart per AU
    fn_tp = np.array(stats["per_au_fn_to_tp"], dtype=float)
    fp_tn = np.array(stats["per_au_fp_to_tn"], dtype=float)
    tn_fp = np.array(stats["per_au_tn_to_fp"], dtype=float)
    tp_fn = np.array(stats["per_au_tp_to_fn"], dtype=float)

    ax_bar.barh(y_pos, fn_tp, color=TRANSITION_COLORS["FN->TP"], label="Fixed: FN->TP")
    ax_bar.barh(y_pos, fp_tn, left=fn_tp, color=TRANSITION_COLORS["FP->TN"], label="Fixed: FP->TN")
    ax_bar.barh(y_pos, -tn_fp, color=TRANSITION_COLORS["TN->FP"], label="Broken: TN->FP")
    ax_bar.barh(y_pos, -tp_fn, left=-tn_fp, color=TRANSITION_COLORS["TP->FN"], label="Broken: TP->FN")
    ax_bar.axvline(0.0, color="#0F172A", linewidth=1.0)

    max_pos = max(float((fn_tp + fp_tn).max(initial=0.0)), 1.0)
    max_neg = max(float((tn_fp + tp_fn).max(initial=0.0)), 1.0)
    ax_bar.set_xlim(-max_neg * 1.35, max_pos * 2.45)
    for i in range(len(au_ids)):
        right_end = fn_tp[i] + fp_tn[i]
        net_i = stats["per_au_net_gain"][i]
        df1_i = stats["per_au_delta_f1_pp"][i]
        dp_i = stats["per_au_mean_signed_delta_p"][i]
        ax_bar.text(
            right_end + max_pos * 0.06,
            y_pos[i],
            f"net {net_i:+d} | dF1 {df1_i:+.2f}pp | dp {dp_i:+.4f}",
            va="center", ha="left", fontsize=8.1, color="#1E293B",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"AU{au}" for au in au_ids], fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Flipped Samples (<- Broken  |  Fixed ->)")
    ax_bar.set_title(f"Per-AU Flip Breakdown at K={cmp_k} vs. K=0", fontweight="bold")
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax_bar.legend(fontsize=8.2, loc="lower left")

    # Right: Decision-boundary zoomed scatter p(K=0) vs p(K=compare_k)
    p0 = predictions[0].numpy()
    pk = predictions[cmp_k].numpy()
    truth = labels.bool().numpy()
    active_cols = [i for i in range(len(au_ids)) if stats["per_au_max_abs_delta_p"][i] > 1e-5]
    if not active_cols:
        active_cols = list(range(len(au_ids)))
    active_label = ", ".join(f"AU{au_ids[i]}" for i in active_cols)

    p0_act = p0[:, active_cols].reshape(-1)
    pk_act = pk[:, active_cols].reshape(-1)
    gt_act = truth[:, active_cols].reshape(-1)
    b0_act = p0_act > 0.5
    bk_act = pk_act > 0.5
    flipped_any = b0_act ^ bk_act

    if flipped_any.any():
        max_dev = max(
            float(np.abs(p0_act[flipped_any] - 0.5).max()),
            float(np.abs(pk_act[flipped_any] - 0.5).max()),
        )
        span = min(0.5, max(0.025, max_dev * 1.85))
    else:
        span = 0.5
    lo, hi = max(0.0, 0.5 - span), min(1.0, 0.5 + span)

    in_window = (p0_act >= lo) & (p0_act <= hi) & (pk_act >= lo) & (pk_act <= hi)
    unch_idx = np.flatnonzero((~flipped_any) & in_window)
    if len(unch_idx) > 4000:
        rng = np.random.default_rng(0)
        unch_idx = rng.choice(unch_idx, size=4000, replace=False)

    ax_scat.fill_between([lo, 0.5], 0.5, hi, color="#ECFDF5", alpha=0.55)
    ax_scat.fill_between([0.5, hi], lo, 0.5, color="#FEF2F2", alpha=0.45)
    ax_scat.plot([lo, hi], [lo, hi], linestyle="--", color="#64748B", linewidth=1.1, label="No change (y = x)")
    ax_scat.axvline(0.5, color="#334155", linestyle=":", linewidth=1.1)
    ax_scat.axhline(0.5, color="#334155", linestyle=":", linewidth=1.1)
    if len(unch_idx):
        ax_scat.scatter(p0_act[unch_idx], pk_act[unch_idx], s=14, alpha=0.28, color="#94A3B8",
                        edgecolors="none", label="Unchanged (in window)")

    markers = {"FN->TP": "^", "FP->TN": "s", "TN->FP": "D", "TP->FN": "X"}
    masks = {
        "FN->TP": gt_act & (~b0_act) & bk_act,
        "FP->TN": (~gt_act) & b0_act & (~bk_act),
        "TN->FP": (~gt_act) & (~b0_act) & bk_act,
        "TP->FN": gt_act & b0_act & (~bk_act),
    }
    for tr in TRANSITION_ORDER:
        m = masks[tr]
        cnt = int(m.sum())
        if cnt > 0:
            ax_scat.scatter(
                p0_act[m], pk_act[m], s=58, marker=markers[tr],
                color=TRANSITION_COLORS[tr], edgecolor="#0F172A", linewidth=0.55, alpha=0.92, zorder=4,
                label=f"{TRANSITION_LABELS[tr]} ({cnt})",
            )
    ax_scat.text(lo + 0.03 * (hi - lo), hi - 0.05 * (hi - lo), "Flipped OFF -> ON",
                 fontsize=8.2, fontweight="bold", color="#166534", va="top", ha="left")
    ax_scat.text(hi - 0.03 * (hi - lo), lo + 0.04 * (hi - lo), "Flipped ON -> OFF",
                 fontsize=8.2, fontweight="bold", color="#991B1B", va="bottom", ha="right")
    ax_scat.set_xlim(lo, hi)
    ax_scat.set_ylim(lo, hi)
    ax_scat.set_xlabel("Direct Graph Probability p(K=0)")
    ax_scat.set_ylabel(f"Refined Probability p(K={cmp_k})")
    zoom_note = f" [zoomed {lo:.3f}..{hi:.3f}]" if span < 0.49 else ""
    ax_scat.set_title(f"Decision Boundary Flips ({active_label}){zoom_note}", fontweight="bold")
    ax_scat.grid(True, linestyle=":", alpha=0.4)
    ax_scat.legend(fontsize=8.0, loc="upper left", bbox_to_anchor=(0.0, 0.90), framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _draw_sample_card(fig, parent_spec, event, k_values, compare_k, accent_color):
    """Draw one face card: cropped RGB image on top + probability vs K sparkline below."""
    sub = parent_spec.subgridspec(2, 1, height_ratios=[2.35, 1.15], hspace=0.22)
    ax_img = fig.add_subplot(sub[0])
    ax_trj = fig.add_subplot(sub[1])

    face = _load_cropped_face_image(event["image_path"])
    if face is not None:
        ax_img.imshow(face)
    else:
        ax_img.set_facecolor("#F1F5F9")
        ax_img.text(0.5, 0.5, f"Image unavailable\n{event['sample_id']}", ha="center", va="center",
                    fontsize=8.5, color="#475569", transform=ax_img.transAxes)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(2.4)

    short_id = event["sample_id"]
    if len(short_id) > 22:
        short_id = "..." + short_id[-19:]
    title_line1 = f"{event['transition']} | AU{event['au']} (GT={event['gt']}) | {short_id}"
    title_line2 = f"p0={event['p0']:.3f} -> p{compare_k}={event['pk']:.3f} ({event['delta_p']:+.3f})"
    if event["partner_aus"]:
        p_au, p_info = next(iter(event["partner_aus"].items()))
        title_line2 += f" | AU{p_au}:{p_info['p0']:.2f}->{p_info['pk']:.2f}"
    ax_img.set_title(f"{title_line1}\n{title_line2}", fontsize=7.6, fontweight="bold", color="#0F172A", pad=4)

    # Bottom sparkline: zoomed probability trajectory across K around the 0.5 threshold
    x = np.arange(len(k_values))
    y_main = [event["trajectory"][k] for k in k_values]
    ax_trj.plot(x, y_main, marker="o", markersize=3.4, linewidth=2.0, color=accent_color, label=f"AU{event['au']}")
    ax_trj.axhline(0.5, linestyle="--", linewidth=1.0, color="#0F172A", alpha=0.8)
    if compare_k in k_values:
        ax_trj.axvline(k_values.index(compare_k), linestyle=":", linewidth=1.0, color=accent_color, alpha=0.55)
    y_min = min(min(y_main), 0.5)
    y_max = max(max(y_main), 0.5)
    y_pad = max(0.008, 0.35 * (y_max - y_min))
    y_lo, y_hi = max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)
    ax_trj.set_ylim(y_lo, y_hi)
    tick_step = max(1, len(k_values) // 5)
    ticks = list(range(0, len(k_values), tick_step))
    if (len(k_values) - 1) not in ticks:
        ticks.append(len(k_values) - 1)
    ax_trj.set_xticks(ticks)
    ax_trj.set_xticklabels([str(k_values[i]) for i in ticks], fontsize=7)
    ax_trj.set_yticks([y_lo, 0.5, y_hi])
    ax_trj.set_yticklabels([f"{y_lo:.3f}", "0.500", f"{y_hi:.3f}"], fontsize=6.6)
    ax_trj.grid(True, linestyle=":", alpha=0.4)
    ax_trj.legend(fontsize=6.6, loc="best", frameon=False, handlelength=1.4)

    gt_s = ",".join(str(a) for a in event["gt_aus"]) or "none"
    p0_s = ",".join(str(a) for a in event["pred0_aus"]) or "none"
    pk_s = ",".join(str(a) for a in event["predk_aus"]) or "none"
    ax_trj.set_xlabel(f"GT:[{gt_s}]  K0:[{p0_s}]  K{compare_k}:[{pk_s}]", fontsize=6.6, color="#334155", labelpad=2)


def _save_sample_gallery_plot(plt, analysis, num_samples, out_path):
    from matplotlib.gridspec import GridSpec

    k_values = analysis["k_values"]
    cmp_k = analysis["compare_k"]
    events = analysis["compare_events"]
    fixed = _select_diverse_events([e for e in events if e["outcome"] == "fixed"], num_samples)
    broken = _select_diverse_events([e for e in events if e["outcome"] == "broken"], num_samples)

    if not fixed and not broken:
        fallback = _select_diverse_events(analysis["fallback_top_shifts"], num_samples)
        if not fallback:
            fig, ax = plt.subplots(figsize=(10, 3.2), dpi=160)
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                f"No prediction flips or probability shifts detected between K=0 and K={cmp_k}.\n"
                f"Active FACS rules on this dataset: {list(analysis['per_k'][0]['hard_facs_rule_counts'].keys()) or 'none'}",
                ha="center", va="center", fontsize=11, color="#334155",
            )
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            return

        cols = min(4, len(fallback))
        rows = int(np.ceil(len(fallback) / cols))
        fig = plt.figure(figsize=(3.7 * cols, 4.1 * rows + 0.8), dpi=160)
        fig.patch.set_facecolor("white")
        gs = GridSpec(rows, cols, figure=fig, wspace=0.22, hspace=0.45, top=0.88, bottom=0.06, left=0.04, right=0.98)
        for idx, ev in enumerate(fallback):
            _draw_sample_card(fig, gs[idx // cols, idx % cols], ev, k_values, cmp_k, "#2563EB")
        fig.suptitle(
            f"No Threshold Flips at K={cmp_k} — Showing Top {len(fallback)} Largest Probability Shifts",
            fontsize=12, fontweight="bold", y=0.97,
        )
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    sections = []
    if fixed:
        noun = "case" if len(fixed) == 1 else "cases"
        sections.append((
            f"FIXED BY REFINEMENT: Wrong at K=0 -> Right at K={cmp_k} "
            f"(showing {len(fixed)} diverse {noun} of {analysis['per_k'][cmp_k]['fixed']})",
            fixed,
            "#1B8A5A",
        ))
    if broken:
        noun = "case" if len(broken) == 1 else "cases"
        sections.append((
            f"BROKEN BY REFINEMENT: Right at K=0 -> Wrong at K={cmp_k} "
            f"(showing {len(broken)} diverse {noun} of {analysis['per_k'][cmp_k]['broken']})",
            broken,
            "#C93B3B",
        ))

    cols = min(4, max(len(sec[1]) for sec in sections))
    row_specs = []
    height_ratios = []
    for title, items, color in sections:
        row_specs.append(("header", title, color))
        height_ratios.append(0.18)
        n_rows = int(np.ceil(len(items) / cols))
        for r in range(n_rows):
            row_specs.append(("cards", items[r * cols : (r + 1) * cols], color))
            height_ratios.append(1.0)

    card_rows = sum(1 for kind, _, _ in row_specs if kind == "cards")
    fig = plt.figure(figsize=(3.85 * cols, 4.15 * card_rows + 0.65 * len(sections) + 0.5), dpi=160)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        len(row_specs), cols, figure=fig,
        height_ratios=height_ratios,
        wspace=0.22, hspace=0.38, top=0.95, bottom=0.04, left=0.035, right=0.985,
    )

    for r_idx, (kind, payload, color) in enumerate(row_specs):
        if kind == "header":
            ax_h = fig.add_subplot(gs[r_idx, :])
            ax_h.axis("off")
            ax_h.text(
                0.0, 0.35, payload, fontsize=11.5, fontweight="bold", color="white",
                va="center", ha="left", transform=ax_h.transAxes,
                bbox={"facecolor": color, "edgecolor": "none", "boxstyle": "round,pad=0.35"},
            )
        else:
            for c_idx, ev in enumerate(payload):
                card_color = TRANSITION_COLORS.get(ev["transition"], color)
                _draw_sample_card(fig, gs[r_idx, c_idx], ev, k_values, cmp_k, card_color)

    fig.suptitle(
        f"Sample-Level Impact of Test-Time Refinement (K=0 vs. K={cmp_k})",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_transition_trajectories_plot(plt, analysis, predictions, labels, out_path):
    k_values = analysis["k_values"]
    au_ids = analysis["au_ids"]
    cmp_k = analysis["compare_k"]
    x = np.arange(len(k_values))
    x_labels = [str(k) for k in k_values]

    p0 = predictions[0].numpy()
    p_cmp = predictions[cmp_k].numpy()
    truth = labels.bool().numpy()
    b0 = p0 > 0.5
    bk = p_cmp > 0.5

    au_flip_counts = [(i, int((b0[:, i] ^ bk[:, i]).sum())) for i in range(len(au_ids))]
    flipped_aus = [i for i, cnt in sorted(au_flip_counts, key=lambda t: (-t[1], t[0])) if cnt > 0][:8]

    if not flipped_aus:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=160)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"No AU threshold flips occurred between K=0 and K={cmp_k}.",
            ha="center", va="center", fontsize=11, color="#334155",
        )
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    cols = min(2, len(flipped_aus))
    rows = int(np.ceil(len(flipped_aus) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 4.6 * rows), dpi=160, squeeze=False)
    fig.patch.set_facecolor("white")

    stacked = np.stack([predictions[k].numpy() for k in k_values], axis=0)  # (K, N, A)
    rng = np.random.default_rng(0)

    for plot_idx, a_idx in enumerate(flipped_aus):
        ax = axes[plot_idx // cols, plot_idx % cols]
        gt_col = truth[:, a_idx]
        b0_col = b0[:, a_idx]
        bk_col = bk[:, a_idx]
        masks = {
            "FN->TP": gt_col & (~b0_col) & bk_col,
            "FP->TN": (~gt_col) & b0_col & (~bk_col),
            "TN->FP": (~gt_col) & (~b0_col) & bk_col,
            "TP->FN": gt_col & b0_col & (~bk_col),
        }
        vals_for_ylim = [0.5]
        for tr in TRANSITION_ORDER:
            idxs = np.flatnonzero(masks[tr])
            if not len(idxs):
                continue
            trajs = stacked[:, idxs, a_idx]  # (K, M)
            vals_for_ylim.extend([float(trajs.min()), float(trajs.max())])
            color = TRANSITION_COLORS[tr]
            sample_idxs = idxs if len(idxs) <= 25 else rng.choice(idxs, size=25, replace=False)
            for s_i in sample_idxs:
                ax.plot(x, stacked[:, s_i, a_idx], color=color, alpha=0.18, linewidth=1.0)
            med = np.median(trajs, axis=1)
            q25 = np.percentile(trajs, 25, axis=1)
            q75 = np.percentile(trajs, 75, axis=1)
            ax.fill_between(x, q25, q75, color=color, alpha=0.20)
            ax.plot(x, med, marker="o", markersize=4.5, linewidth=2.5, color=color,
                    label=f"{TRANSITION_LABELS[tr]} (n={len(idxs)})")

        ax.axhline(0.5, linestyle="--", color="#0F172A", linewidth=1.2, label="Threshold = 0.5")
        if cmp_k in k_values:
            ax.axvline(k_values.index(cmp_k), linestyle=":", color="#475569", linewidth=1.2)
        y_lo_raw, y_hi_raw = min(vals_for_ylim), max(vals_for_ylim)
        pad = max(0.010, 0.30 * (y_hi_raw - y_lo_raw))
        ax.set_ylim(max(0.0, y_lo_raw - pad), min(1.0, y_hi_raw + pad))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Refinement Steps (K)")
        ax.set_ylabel("Predicted Probability (zoomed around 0.5)")
        ax.set_title(
            f"AU{au_ids[a_idx]} — Trajectories of Flipped Samples (K=0 vs. K={cmp_k})",
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(fontsize=8.2, loc="best")

    for empty_idx in range(len(flipped_aus), rows * cols):
        axes[empty_idx // cols, empty_idx % cols].axis("off")

    fig.suptitle(
        "Probability Evolution Across K by Outcome (Median, IQR Band, & Sample Curves)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_changed_samples_csv(analysis, out_path):
    k_values = analysis["k_values"]
    cmp_k = analysis["compare_k"]
    rows = analysis["compare_events"] or analysis["fallback_top_shifts"]

    fieldnames = [
        "sample_idx", "sample_id", "image_path", "au", "transition", "outcome",
        "gt", "p_k0", f"p_k{cmp_k}", "delta_p", "first_flip_k",
        "gt_aus", "pred_k0_aus", f"pred_k{cmp_k}_aus",
    ]
    if rows and "expr_k0" in rows[0]:
        fieldnames += ["expr_k0", f"expr_k{cmp_k}"]
    fieldnames += [f"p_k{k}" for k in k_values]

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ev in rows:
            row = {
                "sample_idx": ev["sample_idx"],
                "sample_id": ev["sample_id"],
                "image_path": ev["image_path"] or "",
                "au": ev["au"],
                "transition": ev["transition"],
                "outcome": ev["outcome"],
                "gt": ev["gt"],
                "p_k0": f"{ev['p0']:.6f}",
                f"p_k{cmp_k}": f"{ev['pk']:.6f}",
                "delta_p": f"{ev['delta_p']:+.6f}",
                "first_flip_k": "" if ev["first_flip_k"] is None else ev["first_flip_k"],
                "gt_aus": " ".join(str(a) for a in ev["gt_aus"]),
                "pred_k0_aus": " ".join(str(a) for a in ev["pred0_aus"]),
                f"pred_k{cmp_k}_aus": " ".join(str(a) for a in ev["predk_aus"]),
            }
            if "expr_k0" in ev:
                row["expr_k0"] = ev["expr_k0"]
                row[f"expr_k{cmp_k}"] = ev["expr_k_cmp"]
            for k in k_values:
                row[f"p_k{k}"] = f"{ev['trajectory'][k]:.6f}"
            writer.writerow(row)


def save_visualizations(analysis, predictions, labels, output_dir, num_samples=8):
    """Save diagnostic plots, changed-samples CSV, and JSON summary to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    csv_path = output_dir / "changed_samples.csv"
    _save_changed_samples_csv(analysis, csv_path)
    saved["changed_samples_csv"] = csv_path

    json_path = output_dir / "k_grid_summary.json"
    summary_payload = {
        "k_values": analysis["k_values"],
        "au_ids": list(analysis["au_ids"]),
        "num_samples": analysis["num_samples"],
        "best_k": analysis["best_k"],
        "compare_k": analysis["compare_k"],
        "per_k": {str(k): v for k, v in analysis["per_k"].items()},
        "num_flipped_events": len(analysis["compare_events"]),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    saved["summary_json"] = json_path

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[Warning] matplotlib not available ({exc}); skipped PNG plots (CSV/JSON saved).")
        return saved

    curves_path = output_dir / "01_k_curves_and_flips.png"
    _save_curves_and_flips_plot(plt, analysis, curves_path)
    saved["curves_and_flips"] = curves_path

    breakdown_path = output_dir / "02_per_au_flip_breakdown.png"
    _save_per_au_breakdown_plot(plt, analysis, predictions, labels, breakdown_path)
    saved["per_au_breakdown"] = breakdown_path

    gallery_path = output_dir / "03_sample_gallery_fixed_vs_broken.png"
    _save_sample_gallery_plot(plt, analysis, num_samples=num_samples, out_path=gallery_path)
    saved["sample_gallery"] = gallery_path

    traj_path = output_dir / "04_trajectories_by_transition.png"
    _save_transition_trajectories_plot(plt, analysis, predictions, labels, traj_path)
    saved["trajectories"] = traj_path

    return saved


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
    parser.add_argument("--viz_dir", "--viz-dir", type=Path, default=Path("k_grid_viz"),
                        help="Directory for visualization PNGs, changed_samples.csv, and summary JSON")
    parser.add_argument("--no_viz", "--no-viz", action="store_true",
                        help="Disable writing visualization PNGs/CSV/JSON (console report still prints)")
    parser.add_argument("--compare_k", "--compare-k", type=int, default=None,
                        help="Specific K to compare against K=0 for flip breakdown and face gallery "
                             "(defaults to best K>0, or largest K>0 if K=0 is best)")
    parser.add_argument("--num_samples", "--num-samples", type=int, default=8,
                        help="Max Fixed and max Broken sample cards to display in the face gallery")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1 or args.num_workers < 0 or any(k < 0 for k in args.ks):
        parser.error("batch_size must be positive; num_workers and K must be non-negative")
    if args.compare_k is not None and args.compare_k < 0:
        parser.error("compare_k must be non-negative")
    if args.num_samples < 1:
        parser.error("num_samples must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = sorted({0, *args.ks, *([args.compare_k] if args.compare_k is not None else [])})
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
    results, details = evaluate_k_grid(model, loader, device, k_values, return_details=True)
    means = print_results(results, cfg.au_ids)
    if recorded_f1 is not None:
        print(f"K=0 vs recorded checkpoint F1: {100 * (means[0] - recorded_f1):+.4f} percentage points")

    analysis = analyze_k_changes(
        details["predictions"],
        details["labels"],
        cfg.au_ids,
        compare_k=args.compare_k,
        dataset=val_data,
        emotion_predictions=details["emotion_predictions"],
        emotion_names=model.emotion_names,
    )
    print_change_report(analysis, cfg.au_ids)

    if not args.no_viz:
        saved = save_visualizations(
            analysis,
            details["predictions"],
            details["labels"],
            output_dir=args.viz_dir,
            num_samples=args.num_samples,
        )
        print(f"\nSaved K-grid visualizations to {args.viz_dir.resolve()}:")
        for key, path in saved.items():
            print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
