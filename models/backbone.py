"""
Visual backbone: ResNet50 feature extractor.
Outputs a global feature vector z_img.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class VisualBackbone(nn.Module):
    """
    ResNet18 backbone that outputs a global image feature z_img.
    The final classification layer is removed.
    """
    
    def __init__(self, name="resnet50", feat_dim=2048, pretrained=True):
        super().__init__()
        if name == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        else:
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
        # Remove the final FC layer AND the AdaptiveAvgPool2d layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_dim = feat_dim
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            z_img: (B, feat_dim, D_patches) sequence of spatial patches
        """
        z = self.features(x)       # (B, 512, 7, 7) for ResNet18
        z = z.flatten(start_dim=2) # (B, 512, 49)
        return z
