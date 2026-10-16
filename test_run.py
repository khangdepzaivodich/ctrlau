"""
Quick sanity check: dummy forward + backward pass to verify shapes and gradients.
"""
import torch
import sys
import os
import warnings

# Suppress PyTorch JIT deprecation FutureWarning from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ModelConfig, DISFA_AUS, BP4D_AUS, CTRL_NUM_EMOTIONS,
    DISFA_M_AE, BP4D_M_AE, compute_expression_pseudo_labels,
)
from models.ctrlau import CtrlAUModel


def test_forward_backward():
    print("=" * 60)
    print("CtrlAU Architecture Verification")
    print("=" * 60)
    
    cfg = ModelConfig()
    device = torch.device("cpu")  # CPU for quick test
    
    print("\n1. Creating model...")
    model = CtrlAUModel(cfg=cfg).to(device)
    model.set_phase(3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n2. Creating dummy data...")
    B = 4  # small batch
    images = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=device)
    au_labels = torch.randint(0, 2, (B, len(DISFA_AUS)), device=device).float()
    print(f"   Images shape: {images.shape}")
    print(f"   AU labels shape: {au_labels.shape}")

    print("\n3. Verifying Phase 1 excludes FACS loss...")
    model.set_phase(1)
    phase1_outputs = model(images, au_labels=au_labels, phase=1)
    assert not any("facs" in name for name in phase1_outputs["losses"])
    assert phase1_outputs["emotion_pseudo"].shape == (B, CTRL_NUM_EMOTIONS)
    print("   [OK] Phase 1 has no FACS loss.")

    print("\n4b. Verifying strict AU -> Expression graph topology...")
    ae_adj = model.graph_module._build_au_to_expr_adjacency().detach()
    n_au = model.num_aus
    n_expr = model.num_emotions
    assert ae_adj.shape == (n_au + n_expr, n_au + n_expr)
    assert torch.count_nonzero(ae_adj[:n_au, :n_au]) == 0
    assert torch.count_nonzero(ae_adj[n_au:, :]) == 0
    assert torch.equal(ae_adj[:n_au, n_au:], model.graph_module.au_to_expr_adj.detach())
    test_au = torch.randn(1, n_au, 256, requires_grad=True)
    test_expr = torch.randn(1, n_expr, 256, requires_grad=True)
    test_nodes, _ = model.graph_module.forward_au_exp(test_au, test_expr)
    expr_to_au_grad = torch.autograd.grad(
        test_nodes[:, :n_au].sum(), test_expr, allow_unused=True
    )[0]
    assert expr_to_au_grad is None or torch.count_nonzero(expr_to_au_grad) == 0
    print("   [OK] Only AU -> Expression edges are learnable; reverse edges are zero.")

    print("\n4. Verifying published prior dimensions and Neutral rule...")
    assert DISFA_M_AE.shape == (len(DISFA_AUS), CTRL_NUM_EMOTIONS)
    assert BP4D_M_AE.shape == (len(BP4D_AUS), CTRL_NUM_EMOTIONS)
    neutral = compute_expression_pseudo_labels(torch.zeros(1, len(DISFA_AUS)), DISFA_M_AE)
    assert neutral.argmax(dim=1).item() == CTRL_NUM_EMOTIONS - 1
    print("   [OK] DISFA/BP4D priors and all-zero -> Neutral verified.")
    
    print("\n4. Forward pass...")
    model.set_phase(3)
    outputs = model(images, au_labels=au_labels, phase=3)
    
    print(f"   au_logits: {outputs['au_logits'].shape}")
    print(f"   au_probs: {outputs['au_probs'].shape}")
    print(f"   emotion_probs: {outputs['emotion_probs'].shape}")
    print(f"   emotion_pseudo: {outputs['emotion_pseudo'].shape}")
    print(f"   au_au_adj: {outputs['au_au_adj'].shape}")
    print(f"   au_exp_adj: {outputs['au_exp_adj'].shape}")
    print(f"   au_au_importance_mask: {outputs['au_au_importance_mask'].shape}")
    print(f"   au_au_polarity_mask: {outputs['au_au_polarity_mask'].shape}")
    print(f"   z_img: {outputs['z_img'].shape}")
    
    print("\n5. Loss values:")
    losses = outputs["losses"]
    for k, v in sorted(losses.items()):
        print(f"   {k}: {v.item():.6f}")
    
    print("\n6. Backward pass...")
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
    
    print("\n7. Checking for NaN in outputs...")
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
