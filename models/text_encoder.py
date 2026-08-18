"""
Text encoder: CLIP text encoder for AU descriptions.
Produces frozen text embeddings for each AU description.
"""
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder(nn.Module):
    """
    Uses CLIP's text encoder to produce embeddings for AU descriptions.
    The text encoder is frozen (no gradient).
    """
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(
            clip_model_name,
            use_safetensors=True,
        )
        
        # Freeze text encoder
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.text_model.config.hidden_size  # 512 for base
    
    @torch.no_grad()
    def forward(self, descriptions):
        """
        Args:
            descriptions: list of strings, one per AU
        Returns:
            text_embeds: (N_AU, embed_dim) text embeddings
        """
        device = next(self.text_model.parameters()).device
        
        tokens = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        outputs = self.text_model(**tokens)
        
        # Use the pooled output (CLS token)
        text_embeds = outputs.pooler_output  # (N_AU, embed_dim)
        
        return text_embeds
