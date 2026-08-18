"""
Quick sanity check: dummy forward + backward pass to verify shapes and gradients.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, NUM_AUS
from models.ctrlau import CtrlAUModel


def test_forward_backward():
    print("=" * 60)
    print("CtrlAU Architecture Verification")
    print("=" * 60)
    
    cfg = ModelConfig()
    device = torch.device("cpu")  # CPU for quick test
    
    print("\n1. Creating model...")
    model = CtrlAUModel(cfg=cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n2. Creating dummy data...")
    B = 4  # small batch
    images = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=device)
    au_labels = torch.randint(0, 2, (B, NUM_AUS), device=device).float()
    print(f"   Images shape: {images.shape}")
    print(f"   AU labels shape: {au_labels.shape}")
    
    print("\n3. Forward pass...")
    model.train()
    outputs = model(images, au_labels=au_labels)
    
    print(f"   au_logits: {outputs['au_logits'].shape}")
    print(f"   au_probs: {outputs['au_probs'].shape}")
    print(f"   emotion_probs: {outputs['emotion_probs'].shape}")
    print(f"   emotion_pseudo: {outputs['emotion_pseudo'].shape}")
    print(f"   au_au_adj: {outputs['au_au_adj'].shape}")
    print(f"   au_exp_adj: {outputs['au_exp_adj'].shape}")
    print(f"   au_au_importance_mask: {outputs['au_au_importance_mask'].shape}")
    print(f"   au_au_polarity_mask: {outputs['au_au_polarity_mask'].shape}")
    print(f"   z_img: {outputs['z_img'].shape}")
    
    print("\n4. Loss values:")
    losses = outputs["losses"]
    for k, v in sorted(losses.items()):
        print(f"   {k}: {v.item():.6f}")
    
    print("\n5. Backward pass...")
    total_loss = losses["total_loss"]
    total_loss.backward()
    
    # Check gradients
    has_nan = False
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if torch.isnan(param.grad).any():
                print(f"   WARNING: NaN gradient in {name}")
                has_nan = True
            grad_norms[name] = grad_norm
    
    # Print top-5 gradient norms
    sorted_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    print("\n   Top-5 gradient norms:")
    for name, norm in sorted_norms[:5]:
        print(f"     {name}: {norm:.6f}")
    
    if not has_nan:
        print("\n   [OK] No NaN gradients detected.")
    
    print("\n6. Checking for NaN in outputs...")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print(f"   WARNING: NaN in {k}")
            else:
                print(f"   [OK] {k}: OK")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    if torch.isnan(vv).any():
                        print(f"   WARNING: NaN in losses.{kk}")
                    else:
                        print(f"   [OK] losses.{kk}: OK")
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                if isinstance(vv, torch.Tensor) and torch.isnan(vv).any():
                    print(f"   WARNING: NaN in {k}[{i}]")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_backward()
