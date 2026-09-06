"""
Test script to verify Phase 1:
1. CtrlAUModel (combined SymGraphAU backbone + CtrlAU Phase 1 regularizers)
2. SymGraphAUModel (pure MultiviewSymAU baseline)
Verifies forward pass, backward pass, loss computation, and gradient isolation.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, NUM_AUS
from models.ctrlau import CtrlAUModel
from models.symgraphau import SymGraphAUModel

def test_ctrlau_phase1():
    print("=" * 60)
    print("Testing CtrlAUModel Phase 1 (Combined Method)")
    print("=" * 60)
    
    cfg = ModelConfig()
    device = torch.device("cpu")
    
    # 1. Instantiate CtrlAUModel
    model = CtrlAUModel(cfg=cfg).to(device)
    model.set_phase(1)
    
    # 2. Check parameter states
    print("\n[Parameter Trainable Status]")
    bb_trainable = any(p.requires_grad for p in model.backbone.parameters())
    gl_trainable = any(p.requires_grad for p in model.global_linear.parameters())
    au_trainable = any(p.requires_grad for p in model.au_head.parameters())
    em_trainable = any(p.requires_grad for p in model.emotion_head.parameters())
    vp_trainable = any(p.requires_grad for p in model.visual_proj.parameters())
    tp_trainable = any(p.requires_grad for p in model.text_proj.parameters())
    evp_trainable = any(p.requires_grad for p in model.emotion_visual_proj.parameters())
    etp_trainable = any(p.requires_grad for p in model.emotion_text_proj.parameters())
    
    te_trainable = any(p.requires_grad for p in model.text_encoder.parameters())
    gm_trainable = any(p.requires_grad for p in model.graph_module.parameters())
    mm_trainable = any(p.requires_grad for p in model.mask_module.parameters())
    gc_trainable = any(p.requires_grad for p in model.graph_au_classifiers.parameters())
    
    print(f"  Backbone (ResNet-50):         {bb_trainable} (expected: True)")
    print(f"  LinearBlock (2048->512):      {gl_trainable} (expected: True)")
    print(f"  SymAUHead (8 branches):       {au_trainable} (expected: True)")
    print(f"  SymExprHead (7 branches):     {em_trainable} (expected: True)")
    print(f"  AU visual projection:         {vp_trainable} (expected: True)")
    print(f"  AU text projection:           {tp_trainable} (expected: True)")
    print(f"  Emotion visual projection:    {evp_trainable} (expected: True)")
    print(f"  Emotion text projection:      {etp_trainable} (expected: True)")
    print(f"  CLIP Text Encoder:            {te_trainable} (expected: False - frozen)")
    print(f"  GAT Graph Module:             {gm_trainable} (expected: False - frozen in P1)")
    print(f"  Mask Module:                  {mm_trainable} (expected: False - frozen in P1)")
    print(f"  Graph AU Classifiers:         {gc_trainable} (expected: False - frozen in P1)")
    
    assert bb_trainable and gl_trainable and au_trainable and em_trainable
    assert vp_trainable and tp_trainable and evp_trainable and etp_trainable
    assert not te_trainable and not gm_trainable and not mm_trainable and not gc_trainable
    
    # 3. Synthetic forward pass with labels
    B = 4
    images = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (B, NUM_AUS), device=device).float()
    
    print("\n[Forward Pass with Labels]")
    outputs = model(images, au_labels=labels, phase=1)
    
    au_probs = outputs["au_probs"]
    em_probs = outputs["emotion_probs"]
    em_pseudo = outputs["emotion_pseudo"]
    losses = outputs["losses"]
    
    print(f"  au_probs shape:        {au_probs.shape} (expected: [{B}, 8])")
    print(f"  emotion_probs shape:   {em_probs.shape} (expected: [{B}, 7])")
    print(f"  emotion_pseudo shape:  {em_pseudo.shape} (expected: [{B}, 7])")
    
    assert au_probs.shape == (B, 8)
    assert em_probs.shape == (B, 7)
    assert em_pseudo.shape == (B, 7)
    
    print("\n[Loss Breakdown in Phase 1]")
    required_losses = [
        "loss_wa", "loss_we", "loss_phase1",
        "loss_ib", "loss_align", "loss_decorr",
        "loss_contrastive", "loss_emo_contrastive",
        "loss_facs_au", "total_loss"
    ]
    for k in required_losses:
        assert k in losses, f"Missing loss component: {k}"
        print(f"  {k:<22}: {losses[k].item():.4f}")
        
    # 4. Backward pass
    print("\n[Backward Pass & Gradient Verification]")
    total_loss = losses["total_loss"]
    total_loss.backward()
    
    bb_has_grad = any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
    gl_has_grad = any(p.grad is not None for p in model.global_linear.parameters() if p.requires_grad)
    au_has_grad = any(p.grad is not None for p in model.au_head.parameters() if p.requires_grad)
    em_has_grad = any(p.grad is not None for p in model.emotion_head.parameters() if p.requires_grad)
    vp_has_grad = any(p.grad is not None for p in model.visual_proj.parameters() if p.requires_grad)
    tp_has_grad = any(p.grad is not None for p in model.text_proj.parameters() if p.requires_grad)
    evp_has_grad = any(p.grad is not None for p in model.emotion_visual_proj.parameters() if p.requires_grad)
    etp_has_grad = any(p.grad is not None for p in model.emotion_text_proj.parameters() if p.requires_grad)
    
    gm_has_grad = any(p.grad is not None for p in model.graph_module.parameters())
    te_has_grad = any(p.grad is not None for p in model.text_encoder.parameters())
    
    print(f"  Backbone has gradients:         {bb_has_grad} (expected: True)")
    print(f"  LinearBlock has gradients:      {gl_has_grad} (expected: True)")
    print(f"  SymAUHead has gradients:        {au_has_grad} (expected: True)")
    print(f"  SymExprHead has gradients:      {em_has_grad} (expected: True)")
    print(f"  visual_proj has gradients:      {vp_has_grad} (expected: True)")
    print(f"  text_proj has gradients:        {tp_has_grad} (expected: True)")
    print(f"  emotion_visual_proj has grad:   {evp_has_grad} (expected: True)")
    print(f"  emotion_text_proj has grad:     {etp_has_grad} (expected: True)")
    print(f"  GAT Graph Module has grad:      {gm_has_grad} (expected: False)")
    print(f"  CLIP Text Encoder has grad:     {te_has_grad} (expected: False)")
    
    assert bb_has_grad and gl_has_grad and au_has_grad and em_has_grad
    assert vp_has_grad and tp_has_grad and evp_has_grad and etp_has_grad
    assert not gm_has_grad and not te_has_grad
    
    # 5. Inference pass without labels
    print("\n[Inference Pass (No Labels)]")
    model.eval()
    with torch.no_grad():
        eval_outputs = model(images, phase=1)
        assert "au_probs" in eval_outputs
        assert eval_outputs["au_probs"].shape == (B, 8)
        print(f"  Inference output shape: {eval_outputs['au_probs'].shape} - OK!")
    
    print("\n CtrlAUModel Phase 1 test PASSED successfully!")


def test_symgraphau_baseline():
    print("\n" + "=" * 60)
    print("Testing SymGraphAUModel Phase 1 (Pure MultiviewSymAU Baseline)")
    print("=" * 60)
    
    cfg = ModelConfig()
    device = torch.device("cpu")
    
    model = SymGraphAUModel(cfg=cfg).to(device)
    model.set_phase(1)
    
    B = 4
    images = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (B, NUM_AUS), device=device).float()
    
    outputs = model(images, au_labels=labels, phase=1)
    losses = outputs["losses"]
    
    print("\n[Baseline Loss Breakdown in Phase 1]")
    for k, v in losses.items():
        print(f"  {k:<22}: {v.item():.4f}")
        
    assert "loss_wa" in losses
    assert "loss_we" in losses
    assert "loss_phase1" in losses
    assert "total_loss" in losses
    # Baseline should NOT compute CtrlAU regularizers in Phase 1
    assert "loss_ib" not in losses
    assert "loss_contrastive" not in losses
    
    print("\n SymGraphAUModel Pure Baseline test PASSED successfully!")


if __name__ == "__main__":
    test_ctrlau_phase1()
    test_symgraphau_baseline()
