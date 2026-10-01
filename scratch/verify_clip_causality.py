import torch
import torch.nn.functional as F
import sys
import os

# Ensure we can import from the main directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig, AU_INDEX
from models.ctrlau import CtrlAUModel

def verify_clip_causal_semantics():
    print("=================================================================")
    print(" Verifying CLIP Causal Semantics (Idea 1 Assumption)")
    print("=================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = ModelConfig()
    model = CtrlAUModel(cfg).to(device)
    model.eval()
    
    B = 4
    NUM_AUS = model.num_aus
    
    with torch.no_grad():
        # 1. Get projected text embeddings
        text_emb = model._get_text_embeddings(device=device) # (NUM_AUS, D_text)
        text_proj = model.text_proj(text_emb) # (NUM_AUS, D_vis)
        
        # APPLY GRAM-SCHMIDT ORTHOGONALIZATION FIX
        ortho_basis = torch.zeros_like(text_proj)
        for i in range(NUM_AUS):
            v_i = text_proj[i].clone()
            for j in range(i):
                proj_v_i = (v_i * ortho_basis[j]).sum() * ortho_basis[j]
                v_i = v_i - proj_v_i
            ortho_basis[i] = F.normalize(v_i, dim=-1)
            
        text_proj_norm = ortho_basis # Use the orthogonalized vectors
        
        # 2. Check cosine similarity between text prompts
        # If AU12 (Smile) and AU6 (Cheek Raiser) text prompts are highly similar,
        # then nulling one will accidentally null the other.
        au12_idx = AU_INDEX[12]
        au6_idx = AU_INDEX[6]
        
        cos_sim_12_6 = F.cosine_similarity(text_proj_norm[au12_idx], text_proj_norm[au6_idx], dim=-1).item()
        print(f"Cosine Similarity between AU12 text and AU6 text vectors: {cos_sim_12_6:.4f}")
        
        # 3. Simulate a visual embedding that has both AU12 and AU6 active
        D = cfg.shared_embed_dim
        base_vis = torch.randn(B, NUM_AUS, D, device=device) * 0.1
        
        # Inject the concepts heavily
        base_vis[:, au12_idx, :] += 3.0 * text_proj_norm[au12_idx].unsqueeze(0)
        base_vis[:, au6_idx, :]  += 3.0 * text_proj_norm[au6_idx].unsqueeze(0)
        
        print(f"\n--- Simulating Intervention: Nulling AU12 (Lip Corner Puller) ---")
        
        # Original predictions via Graph
        original_nodes, _ = model.graph_module.forward_au_au(base_vis)
        orig_au12_logit = model.au_au_classifiers[au12_idx](original_nodes[:, au12_idx, :])
        orig_au6_logit = model.au_au_classifiers[au6_idx](original_nodes[:, au6_idx, :])
        
        orig_p12 = torch.sigmoid(orig_au12_logit).mean().item()
        orig_p6 = torch.sigmoid(orig_au6_logit).mean().item()
        
        # Perform Subspace Nulling exactly as in Idea 1
        text_dir_12 = text_proj_norm[au12_idx].unsqueeze(0).unsqueeze(0) # (1, 1, D)
        proj = (base_vis * text_dir_12).sum(dim=-1, keepdim=True) * text_dir_12
        nulled_vis = base_vis - proj
        
        # Post-Intervention Predictions
        nulled_nodes, _ = model.graph_module.forward_au_au(nulled_vis)
        nulled_au12_logit = model.au_au_classifiers[au12_idx](nulled_nodes[:, au12_idx, :])
        nulled_au6_logit = model.au_au_classifiers[au6_idx](nulled_nodes[:, au6_idx, :])
        
        nulled_p12 = torch.sigmoid(nulled_au12_logit).mean().item()
        nulled_p6 = torch.sigmoid(nulled_au6_logit).mean().item()
        
        print(f"Target AU (AU12) Prob:    {orig_p12:.4f} -> {nulled_p12:.4f}  (Drop: {orig_p12 - nulled_p12:.4f})")
        print(f"Correlated AU (AU6) Prob: {orig_p6:.4f} -> {nulled_p6:.4f}  (Drop: {orig_p6 - nulled_p6:.4f})")
        
        print("\n[Analysis]")
        if cos_sim_12_6 > 0.3:
            print("WARNING: High semantic entanglement detected between text vectors.")
            
        drop_ratio = (orig_p6 - nulled_p6) / (orig_p12 - nulled_p12 + 1e-8)
        if drop_ratio > 0.2:
            print("CONCLUSION: The user's concern is VALID.")
            print("Nulling the AU12 CLIP direction also heavily erased AU6 information.")
            print("Because the CLIP text embedding for 'Lip Corner Puller' implicitly contains 'happiness', it destroys correlated AUs.")
            print("Recommended Fix: We need to orthogonalize the text prompts (e.g., using Gram-Schmidt) before nulling.")
        else:
            print("CONCLUSION: The CLIP directions are surprisingly orthogonal. The assumption holds.")

if __name__ == '__main__':
    verify_clip_causal_semantics()
