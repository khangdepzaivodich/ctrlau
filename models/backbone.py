"""
Visual backbone: ResNet50 feature extractor.
Outputs a global feature vector z_img.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class VisualBackbone(nn.Module):
    """
    ResNet50 backbone that outputs a global image feature z_img.
    The final classification layer is removed.
    """
    
    def __init__(self, feat_dim=2048, pretrained=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.feat_dim = feat_dim
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            z_img: (B, feat_dim) global image features
        """
        z = self.features(x)       # (B, 2048, 1, 1)
        z = z.flatten(start_dim=1) # (B, 2048)
        return z
