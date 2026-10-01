"""
Unit test for FACS AU Violation Loss on AU-AU Graph
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from losses import FACSAUViolationLoss

def test_facs_au_graph_loss():
    print("=" * 60)
    print("Testing FACS AU Violation Loss on AU-AU Graph Outputs")
    print("=" * 60)
    
    loss_fn = FACSAUViolationLoss()
    
    # DISFA AU indices:
    # 0: AU1, 1: AU2, 2: AU4, 3: AU6, 4: AU9, 5: AU12, 6: AU25, 7: AU26
    
    # Case 1: Violation - Jaw Drop (AU26=0.95) but Lips Part is 0.05
    p_violation = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.95]], requires_grad=True)
    loss_v = loss_fn(p_violation)
    print(f"Case 1 (Severe Violation: AU26=0.95, AU25=0.05) -> Loss: {loss_v.item():.4f}")
    assert loss_v.item() > 0.8
    
    # Test gradient flow
    loss_v.backward()
    assert p_violation.grad is not None
    # AU26 grad should be positive (wants p26 to decrease)
    # AU25 grad should be negative (wants p25 to increase)
    grad_25 = p_violation.grad[0, 6].item()
    grad_26 = p_violation.grad[0, 7].item()
    print(f"Gradients: grad(AU25)={grad_25:.4f} (negative), grad(AU26)={grad_26:.4f} (positive)")
    assert grad_25 < 0, "Loss must push AU25 probability to increase!"
    assert grad_26 > 0, "Loss must push AU26 probability to decrease!"
    
    # Case 2: Anatomically Valid - Both active (AU26=0.9, AU25=0.9)
    p_valid = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]])
    loss_valid = loss_fn(p_valid)
    print(f"Case 2 (Valid: AU26=0.9, AU25=0.9) -> Loss: {loss_valid.item():.4f}")
    assert loss_valid.item() < 0.1
    
    # Case 3: Anatomically Valid - Lips Part active without jaw drop (AU26=0.0, AU25=0.85)
    p_valid_smile = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.85, 0.0]])
    loss_smile = loss_fn(p_valid_smile)
    print(f"Case 3 (Valid: AU26=0.0, AU25=0.85) -> Loss: {loss_smile.item():.4f}")
    assert loss_smile.item() == 0.0
    
    print("\nAll FACS AU-AU Graph Loss checks passed successfully!")

if __name__ == "__main__":
    test_facs_au_graph_loss()
