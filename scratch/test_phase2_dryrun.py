"""
Comprehensive verification test for Phase 1 -> Phase 2 transition in CtrlAU:
1. Tests Phase 1 clean JFL training step
2. Saves Phase 1 checkpoint
3. Loads checkpoint into Phase 2 with strict=False and cleaned state_dict
4. Verifies frozen vs trainable parameters in Phase 2
5. Verifies Phase 2 forward pass, loss breakdown (FACS Emotion, DAG, Causal, Counterfactual), and backward pass
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from collections import OrderedDict
from config import ModelConfig, NUM_AUS
from models.ctrlau import CtrlAUModel

def test_phase1_to_phase2():
    print("=" * 65)
    print("1. Testing CtrlAU Phase 1 (Clean Feature Learning)")
    print("=" * 65)
    device = torch.device("cpu")
    cfg = ModelConfig()
    
    # Phase 1: Pure JFL
    cfg.lambda_ib = 0.0
    cfg.lambda_align = 0.0
    cfg.lambda_decorr = 0.0
    cfg.lambda_contrastive = 0.0
    cfg.lambda_emo_contrastive = 0.0
    cfg.lambda_facs_au = 0.0
    
    model_p1 = CtrlAUModel(cfg=cfg).to(device)
    model_p1.set_phase(1)
    
    B = 2
    images = torch.randn(B, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (B, NUM_AUS), device=device).float()
    
    # Forward Phase 1
    out_p1 = model_p1(images, au_labels=labels, phase=1)
    loss_p1 = out_p1["losses"]["total_loss"]
    print(f"  Phase 1 Total Loss: {loss_p1.item():.4f}")
    assert loss_p1.item() > 0
    
    # Backward Phase 1
    loss_p1.backward()
    assert any(p.grad is not None for p in model_p1.backbone.parameters())
    print("  Phase 1 Forward + Backward verified successfully!")
    
    # Save dummy Phase 1 checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "dummy_p1_ckpt.pth")
    torch.save(model_p1.state_dict(), ckpt_path)
    print(f"  Saved dummy Phase 1 checkpoint to: {ckpt_path}")
    
    print("\n" + "=" * 65)
    print("2. Testing Phase 2 Transition & Checkpoint Loading")
    print("=" * 65)
    
    # Instantiate Phase 2 model
    cfg_p2 = ModelConfig()
    model_p2 = CtrlAUModel(cfg=cfg_p2).to(device)
    model_p2.set_phase(2)
    
    # Load checkpoint using train.py logic
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cleaned_state = OrderedDict()
    for k, v in ckpt.items():
        new_k = k.replace("module.", "")
        if new_k.startswith("stage1."):
            new_k = new_k.replace("stage1.", "")
        cleaned_state[new_k] = v
        
    missing, unexpected = model_p2.load_state_dict(cleaned_state, strict=False)
    print(f"  Checkpoint loaded: matched={len(cleaned_state) - len(unexpected)}, missing={len(missing)} (Graph heads)")
    
    # Clean up dummy file
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
        
    # Verify parameter frozen states in Phase 2
    print("\n" + "=" * 65)
    print("3. Verifying Frozen vs Trainable Parameters in Phase 2")
    print("=" * 65)
    
    # Must be frozen:
    bb_frozen = all(not p.requires_grad for p in model_p2.backbone.parameters())
    gl_frozen = all(not p.requires_grad for p in model_p2.global_linear.parameters())
    au_frozen = all(not p.requires_grad for p in model_p2.au_head.parameters())
    em_frozen = all(not p.requires_grad for p in model_p2.emotion_head.parameters())
    te_frozen = all(not p.requires_grad for p in model_p2.text_encoder.parameters())
    
    print(f"  Backbone (ResNet-50) frozen:    {bb_frozen} (expected: True)")
    print(f"  LinearBlock (2048->512) frozen: {gl_frozen} (expected: True)")
    print(f"  AU Head (CNN) frozen:           {au_frozen} (expected: True)")
    print(f"  Emotion Head (CNN) frozen:      {em_frozen} (expected: True)")
    print(f"  CLIP Text Encoder frozen:       {te_frozen} (expected: True)")
    
    assert bb_frozen and gl_frozen and au_frozen and em_frozen and te_frozen
    
    # Must be trainable:
    gm_trainable = any(p.requires_grad for p in model_p2.graph_module.parameters())
    mm_trainable = any(p.requires_grad for p in model_p2.mask_module.parameters())
    ga_trainable = any(p.requires_grad for p in model_p2.graph_au_classifiers.parameters())
    ge_trainable = any(p.requires_grad for p in model_p2.graph_emo_classifiers.parameters())
    
    print(f"  Graph Module (GAT) trainable:   {gm_trainable} (expected: True)")
    print(f"  Mask Module trainable:          {mm_trainable} (expected: True)")
    print(f"  Graph AU Classifiers trainable: {ga_trainable} (expected: True)")
    print(f"  Graph Emo Classifiers trainable:{ge_trainable} (expected: True)")
    
    assert gm_trainable and mm_trainable and ga_trainable and ge_trainable
    
    print("\n" + "=" * 65)
    print("4. Testing Phase 2 Forward & Backward Pass")
    print("=" * 65)
    
    # Configure Phase 2 loss weights
    model_p2.cfg.lambda_au = 0.0
    model_p2.cfg.lambda_emotion = 0.0
    model_p2.cfg.lambda_ib = 0.0
    model_p2.cfg.lambda_align = 0.0
    model_p2.cfg.lambda_decorr = 0.0
    model_p2.cfg.lambda_contrastive = 0.0
    model_p2.cfg.lambda_emo_contrastive = 0.0
    model_p2.cfg.lambda_facs_au = 0.0
    
    model_p2.cfg.lambda_au_au = 1.0
    model_p2.cfg.lambda_graph_au = 1.0
    model_p2.cfg.lambda_graph_emo = 1.0
    model_p2.cfg.lambda_dag = 0.1
    model_p2.cfg.lambda_causal_au = 0.1
    model_p2.cfg.lambda_causal_exp = 0.1
    model_p2.cfg.lambda_facs_exp = 0.1
    model_p2.cfg.lambda_cf_important = 0.1
    model_p2.cfg.lambda_cf_unimportant = 0.1
    
    out_p2 = model_p2(images, au_labels=labels, phase=2)
    losses_p2 = out_p2["losses"]
    
    print("  Phase 2 Loss Breakdown:")
    for k in sorted(losses_p2.keys()):
        print(f"    {k:<24}: {losses_p2[k].item():.4f}")
        
    assert "loss_facs_exp" in losses_p2
    assert "loss_dag" in losses_p2
    assert "loss_causal_au" in losses_p2
    assert "loss_causal_exp" in losses_p2
    assert "loss_cf_important" in losses_p2
    assert "loss_cf_unimportant" in losses_p2
    assert "graph_au_probs" in out_p2
    
    # Backward pass
    total_loss_p2 = losses_p2["total_loss"]
    total_loss_p2.backward()
    
    # Check that frozen parts have NO gradients
    bb_has_grad = any(p.grad is not None for p in model_p2.backbone.parameters())
    gl_has_grad = any(p.grad is not None for p in model_p2.global_linear.parameters())
    print(f"  Backbone received gradients:    {bb_has_grad} (expected: False - FROZEN)")
    print(f"  LinearBlock received gradients: {gl_has_grad} (expected: False - FROZEN)")
    assert not bb_has_grad and not gl_has_grad
    
    # Check that graph modules DO have gradients
    gm_has_grad = any(p.grad is not None for p in model_p2.graph_module.parameters())
    print(f"  Graph Module has gradients:     {gm_has_grad} (expected: True)")
    assert gm_has_grad
    
    print("\n>>> ALL PHASE 1 -> PHASE 2 VERIFICATION CHECKS PASSED 100%! <<<")

if __name__ == "__main__":
    test_phase1_to_phase2()
