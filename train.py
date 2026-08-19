"""
Training script for CtrlAU.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import ModelConfig, DISFA_AUS
from dataset import DISFADataset
from models.ctrlau import CtrlAUModel


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_losses = {}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)
        
        # Forward
        outputs = model(images, au_labels=au_labels)
        losses = outputs["losses"]
        total_loss = losses["total_loss"]
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in running_losses:
                running_losses[k] = 0.0
            running_losses[k] += v.item()
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_losses["total_loss"] / num_batches
            print(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Total Loss: {avg_total:.4f}")
    
    # Print epoch summary
    print(f"\n--- Epoch {epoch} Summary ---")
    for k, v in sorted(running_losses.items()):
        print(f"  {k}: {v / num_batches:.4f}")
    print()
    
    return {k: v / num_batches for k, v in running_losses.items()}


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        images = batch["image"].to(device)
        au_labels = batch["au_labels"].to(device)
        
        outputs = model(images)
        au_probs = outputs["au_probs"]
        
        all_preds.append(au_probs.cpu())
        all_labels.append(au_labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute per-AU F1
    preds_binary = (all_preds > 0.5).float()
    
    f1_scores = []
    for i, au in enumerate(DISFA_AUS):
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


import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CtrlAU Model")
    parser.add_argument(
        "--data_root", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), "DISFA_Data"),
        help="Path to the DISFA dataset root folder containing 'img' and 'ActionUnit_Labels'"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint (.pth file) to resume training from"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Which fold to use for validation (1, 2, or 3). The other two will be used for training."
    )
    args = parser.parse_args()

    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Standard DISFA 3-Fold Cross Validation Subject Splits
    group_1 = ["SN001", "SN002", "SN003", "SN004", "SN005", "SN006", "SN007", "SN008", "SN009"]
    group_2 = ["SN010", "SN011", "SN012", "SN013", "SN016", "SN017", "SN018", "SN021", "SN023"]
    group_3 = ["SN024", "SN025", "SN026", "SN027", "SN028", "SN029", "SN030", "SN031", "SN032"]
    
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
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    
    train_dataset = DISFADataset(data_root=args.data_root, subjects=train_subjects)
    val_dataset = DISFADataset(data_root=args.data_root, subjects=val_subjects)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    
    # Model
    model = CtrlAUModel(cfg=cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print("Checkpoint loaded successfully.")
        else:
            print(f"Warning: Checkpoint file '{args.resume}' not found. Starting from scratch.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs,
    )
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.num_epochs}")
        print(f"{'='*60}")
        
        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        print("Validation:")
        avg_f1 = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        # Save checkpoint for this specific epoch
        epoch_ckpt_path = f"ctrlau_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"  -> Saved epoch checkpoint: {epoch_ckpt_path}")

        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "ctrlau_best.pth")
            print(f"  -> New best F1: {best_f1:.4f}, 'ctrlau_best.pth' updated.")
    
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
