import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.symgraphau import SymGraphAUModel
from config import ModelConfig

def test_symgraphau():
    print("Testing SymGraphAUModel...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    cfg = ModelConfig()
    model = SymGraphAUModel(cfg=cfg).to(device)
    
    # 1. Test Phase 1 (MultiviewSymAU Stage 1 Backbone Testing)
    print("\n--- Test Phase 1 ---")
    model.set_phase(1)
    images = torch.randn(4, 3, 224, 224, device=device)
    au_labels = torch.randint(0, 2, (4, 8), device=device)
    
    outputs = model(images, au_labels=au_labels, phase=1)
    
    assert "losses" in outputs
    assert "total_loss" in outputs["losses"]
    assert "loss_wa" in outputs["losses"]
    assert "loss_we" in outputs["losses"]
    assert "loss_phase1" in outputs["losses"]
    assert "au_probs" in outputs
    assert outputs["au_probs"].shape == (4, 8)
    assert outputs["emotion_probs"].shape == (4, 7)
    
    loss = outputs["losses"]["total_loss"]
    print(f"Phase 1 Loss: {loss.item():.4f}")
    print(f"  L_wa: {outputs['losses']['loss_wa'].item():.4f}")
    print(f"  L_we: {outputs['losses']['loss_we'].item():.4f}")
    
    loss.backward()
    print("Backward pass successful!")
    
    # Verify only backbone, au_head, emotion_head got gradients
    assert model.backbone.features[0].weight.grad is not None
    assert model.au_head.au_embed_heads[0].net[0].weight.grad is not None
    assert model.emotion_head.extractors[0].net[0].weight.grad is not None
    # Verify graph module did NOT get gradients
    for p in model.graph_module.parameters():
        assert p.grad is None
    print("Gradients verified: Backbone and CNN heads updated, Graph module untouched!")
    
    # 2. Test Phase 1 Eval (Inference mode)
    print("\n--- Test Phase 1 Inference ---")
    model.eval()
    with torch.no_grad():
        eval_out = model(images, phase=1)
        assert eval_out["au_probs"].shape == (4, 8)
        assert eval_out["emotion_probs"].shape == (4, 7)
        print("Inference successful!")
        
    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_symgraphau()
