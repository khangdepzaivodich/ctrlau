import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from models.symgraphau import SymGraphAUModel
from config import ModelConfig, DISFA_AUS

def test_dryrun():
    print("Testing full Phase 1 pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ModelConfig()
    model = SymGraphAUModel(cfg=cfg).to(device)
    model.set_phase(1)
    
    # Test on-the-fly class weights
    dummy_weights = torch.tensor([1.2, 1.5, 0.8, 1.1, 1.6, 0.9, 0.4, 0.5], device=device)
    model.update_class_weights(dummy_weights)
    assert torch.isclose(model.wal_weights.sum(), torch.tensor(8.0, device=device))
    print("Class weights update verified!")
    
    # Forward pass
    images = torch.randn(4, 3, 224, 224, device=device)
    au_labels = torch.randint(0, 2, (4, 8), device=device)
    
    outputs = model(images, au_labels=au_labels, phase=1)
    loss = outputs["losses"]["total_loss"]
    assert loss.item() > 0
    print(f"Forward step passed! Total Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Optimizer step passed!")
    
    # Eval pass
    model.eval()
    with torch.no_grad():
        eval_out = model(images, phase=1)
        assert eval_out["au_probs"].shape == (4, 8)
    print("Eval step passed!")
    
    print("\n>>> ALL PHASE 1 CHECKS PASSED 100%! <<<")

if __name__ == "__main__":
    test_dryrun()
