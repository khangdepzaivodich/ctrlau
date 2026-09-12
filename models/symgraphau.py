"""
Exact copy-paste of MultiviewSymAU Phase 1 source files.
No modifications. Directly from:
  - scratch/MultiviewSymAU/model/resnet.py
  - scratch/MultiviewSymAU/model/basic_block.py
  - scratch/MultiviewSymAU/model/SymStage1.py
  - scratch/MultiviewSymAU/utils.py (WeightedAsymmetricLoss, ExpressionBCELoss)
  - scratch/MultiviewSymAU/train_Sym_Stage_1.py (M_AE_DISFA, au_to_expr_pseudo)
"""

# ================================================================
# FROM: scratch/MultiviewSymAU/model/resnet.py
# ================================================================

# ResNet
# Deep Residual Learning for Image Recognition
# Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

models_dir = os.path.expanduser('checkpoints')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b,c,h,w = x.shape
        x = x.view(b,c,-1).permute(0,2,1)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckpt_path = os.path.join(models_dir, model_name['resnet50'])
        try:
            # PyTorch >=2.6: weights_only=True by default
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            # older torch (no weights_only param)
            state = torch.load(ckpt_path, map_location='cpu')
        # nếu file chứa 'state_dict'
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # xoá prefix 'module.' nếu có
        from collections import OrderedDict
        new_state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f'[resnet50] loaded; missing={len(missing)} unexpected={len(unexpected)}')
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model


# ================================================================
# FROM: scratch/MultiviewSymAU/model/basic_block.py
# ================================================================

def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


# ================================================================
# FROM: scratch/MultiviewSymAU/model/SymStage1.py
# ================================================================

EMB_DIM = 256  # kích thước embedding cố định cho cả AU và Expression


# ------------------------------------------------------------
#  Conv1D extractor cho 1 AU / 1 Emotion (phương án A)
#  Input : x (B, D, C_in)
#  Output: emb (B, C_emb=EMB_DIM)
# ------------------------------------------------------------
class Conv1DExtractor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 emb_channels: int = EMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hid_channels)
        self.conv2 = nn.Conv1d(hid_channels, emb_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(emb_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, D, C_in)
        Return:
            emb: (B, C_emb)
        """
        # (B, D, C_in) -> (B, C_in, D) cho Conv1d
        x = x.transpose(1, 2)
        # Conv1
        x = self.conv1(x)          # (B, C_hid, D)
        x = self.bn1(x)
        x = self.relu(x)
        # Conv2
        x = self.conv2(x)          # (B, C_emb, D)
        x = self.bn2(x)
        x = self.relu(x)
        # GAP theo chiều D
        emb = x.mean(dim=-1)       # (B, C_emb)
        return emb


# ------------------------------------------------------------
#  AU Head: N_a nhánh, mỗi nhánh = extractor riêng cho 1 AU
# ------------------------------------------------------------
class AUHead(nn.Module):
    """
    Joint AU head cho Stage-1 (JFL).

    Input:
        x: (B, D, C_in)  # feature map từ backbone (đã qua global_linear)
    Output:
        V_a: (B, N_a, EMB_DIM)   # AU embeddings V^a_i
        p_a: (B, N_a)            # AU probabilities (dùng cho L_wa)
    """
    def __init__(self,
                 in_channels: int,
                 num_aus: int,
                 hid_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.num_aus      = num_aus
        self.hid_channels = hid_channels
        self.emb_channels = EMB_DIM

        # Mỗi AU có 1 Conv1DExtractor riêng
        self.extractors = nn.ModuleList([
            Conv1DExtractor(
                in_channels=self.in_channels,
                hid_channels=self.hid_channels,
                emb_channels=self.emb_channels,
            )
            for _ in range(self.num_aus)
        ])

        # Classifier cho từng AU: EMB_DIM -> 1
        self.classifiers = nn.ModuleList([
            nn.Linear(self.emb_channels, 1)
            for _ in range(self.num_aus)
        ])

    def forward(self, x):
        """
        x: (B, D, C_in)
        """
        emb_list = []
        logit_list = []

        # Lặp từng AU extractor
        for i in range(self.num_aus):
            emb_i = self.extractors[i](x)          # (B, EMB_DIM)
            emb_list.append(emb_i.unsqueeze(1))    # (B, 1, EMB_DIM)
            logit_i = self.classifiers[i](emb_i)   # (B, 1)
            logit_list.append(logit_i)

        # Ghép các AU lại
        V_a    = torch.cat(emb_list, dim=1)        # (B, N_a, EMB_DIM)
        logits = torch.cat(logit_list, dim=1)      # (B, N_a)

        # Chuyển logits -> probabilities p^a (dùng đúng trong (3))
        p_a = torch.sigmoid(logits)

        return V_a, p_a


# ------------------------------------------------------------
#  Expression Head: N_e nhánh, giống AUHead
# ------------------------------------------------------------
class ExprHead(nn.Module):
    """
    Joint Expression head cho Stage-1 (JFL).

    Input:
        x: (B, D, C_in)
    Output:
        V_e: (B, N_e, EMB_DIM)   # Expression embeddings V^e_j
        p_e: (B, N_e)            # Expression probabilities (dùng cho L_we)
    """
    def __init__(self,
                 in_channels: int,
                 num_expr: int,
                 hid_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.num_expr     = num_expr
        self.hid_channels = hid_channels
        self.emb_channels = EMB_DIM

        self.extractors = nn.ModuleList([
            Conv1DExtractor(
                in_channels=self.in_channels,
                hid_channels=self.hid_channels,
                emb_channels=self.emb_channels,
            )
            for _ in range(self.num_expr)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(self.emb_channels, 1)
            for _ in range(self.num_expr)
        ])

    def forward(self, x):
        """
        x: (B, D, C_in)
        """
        emb_list = []
        logit_list = []

        for i in range(self.num_expr):
            emb_i = self.extractors[i](x)          # (B, EMB_DIM)
            emb_list.append(emb_i.unsqueeze(1))    # (B, 1, EMB_DIM)
            logit_i = self.classifiers[i](emb_i)   # (B, 1)
            logit_list.append(logit_i)

        V_e    = torch.cat(emb_list, dim=1)        # (B, N_e, EMB_DIM)
        logits = torch.cat(logit_list, dim=1)      # (B, N_e)

        # probabilities p^e (dùng đúng trong (4))
        p_e = torch.sigmoid(logits)

        return V_e, p_e


# ------------------------------------------------------------
#  Stage-1 model: Backbone + AUHead + ExprHead
# ------------------------------------------------------------
class MEFARGStage1(nn.Module):
    """
    Stage-1: Joint Tasks for Node Feature Learning (JFL)
    - Backbone (Swin/ResNet) -> feature map (B, D, C_in)
    - global_linear: C_in -> C_mid
    - AUHead:    tạo V^a và AU probabilities p^a
    - ExprHead:  tạo V^e và Expression probabilities p^e

    num_aus  : N_a  (ví dụ DISFA: 8)
    num_expr : N_e  (ví dụ: 7 emotion: Angry, Fear, Happy, Sad, Surp, Disg, Neutral)
    """
    def __init__(self,
                 num_aus: int = 8,
                 num_expr: int = 7,
                 backbone: str = 'swin_transformer_base'):
        super().__init__()

        # ---------------- Backbone ----------------
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                from .swin_transformer import swin_transformer_tiny
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                from .swin_transformer import swin_transformer_small
                self.backbone = swin_transformer_small()
            else:
                from .swin_transformer import swin_transformer_base
                self.backbone = swin_transformer_base()

            # Swin trả về (B, D, C_in)
            self.in_channels = self.backbone.num_features
            # giảm kênh một chút
            self.mid_channels = self.in_channels // 2
            # bỏ head classification mặc định
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()

            # Giả định resnet đã sửa để trả (B, D, C_in)
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.mid_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Map C_in -> C_mid, giữ nguyên D
        self.global_linear = LinearBlock(self.in_channels, self.mid_channels)

        # Hai head: dùng mid_channels làm C_in cho Conv1d,
        # hid_channels có thể để = mid_channels
        self.au_head = AUHead(
            in_channels=self.mid_channels,
            num_aus=num_aus,
            hid_channels=self.mid_channels
        )
        self.expr_head = ExprHead(
            in_channels=self.mid_channels,
            num_expr=num_expr,
            hid_channels=self.mid_channels
        )

    def forward(self, x):
        """
        x: input images, shape tùy backbone (ví dụ (B, 3, 224, 224))

        Returns:
            V_a: (B, N_a, EMB_DIM)   # AU node features
            V_e: (B, N_e, EMB_DIM)   # Expression node features
            p_a: (B, N_a)            # AU probabilities  (dùng trong L_wa)
            p_e: (B, N_e)            # Expr probabilities (dùng trong L_we)
        """
        # Backbone output: (B, D, C_in) – ví dụ (64, 49, 2048)
        feat = self.backbone(x)

        # LinearBlock trên kênh, giữ D → (B, D, mid_channels)
        feat = self.global_linear(feat)

        # AU branch
        V_a, p_a = self.au_head(feat)

        # Expression branch
        V_e, p_e = self.expr_head(feat)

        return V_a, V_e, p_a, p_e


# ================================================================
# FROM: scratch/MultiviewSymAU/utils.py (losses only)
# ================================================================

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

class ExpressionBCELoss(nn.Module):
    """
    Loss (4) trong paper SymGraphAU.
    Multi-label BCE nhưng không asymmetric, không weight.
    """
    def __init__(self, eps=1e-8):
        super(ExpressionBCELoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        """
        x: p^e  (B, N_e)  - probabilities (sigmoid output)
        y: y^e  (B, N_e)  - one-hot pseudo labels
        """
        xs_pos = x
        xs_neg = 1 - x

        # BCE cơ bản
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Không asymmetric focusing
        loss = los_pos + los_neg   # (B, N_e)

        # Mean theo N_e
        loss = loss.mean(dim=-1)   # (B,)

        # Mean theo batch
        return -loss.mean()


# ================================================================
# FROM: scratch/MultiviewSymAU/train_Sym_Stage_1.py
# M_AE_DISFA matrix and au_to_expr_pseudo function
# ================================================================

# M_AE_DISFA.csv:
#       Angry, Fear, Happy, Sad, Surprise, Disgust, Neutral
# AU1:  0.50,  0.90, 0.10,  0.90, 0.90,   0.10,   0.10
# AU2:  0.50,  0.90, 0.10,  0.10, 0.90,   0.10,   0.10
# AU4:  0.90,  0.50, 0.10,  0.90, 0.10,   0.50,   0.10
# AU6:  0.10,  0.50, 0.90,  0.50, 0.50,   0.10,   0.10
# AU9:  0.50,  0.10, 0.10,  0.10, 0.10,   0.90,   0.10
# AU12: 0.10,  0.10, 0.90,  0.10, 0.50,   0.50,   0.10
# AU25: 0.50,  0.90, 0.90,  0.50, 0.90,   0.50,   0.10
# AU26: 0.10,  0.50, 0.50,  0.10, 0.90,   0.10,   0.10

M_AE_DISFA = torch.tensor([
    [0.50, 0.90, 0.10, 0.90, 0.90, 0.10, 0.10],  # AU1
    [0.50, 0.90, 0.10, 0.10, 0.90, 0.10, 0.10],  # AU2
    [0.90, 0.50, 0.10, 0.90, 0.10, 0.50, 0.10],  # AU4
    [0.10, 0.50, 0.90, 0.50, 0.50, 0.10, 0.10],  # AU6
    [0.50, 0.10, 0.10, 0.10, 0.10, 0.90, 0.10],  # AU9
    [0.10, 0.10, 0.90, 0.10, 0.50, 0.50, 0.10],  # AU12
    [0.50, 0.90, 0.90, 0.50, 0.90, 0.50, 0.10],  # AU25
    [0.10, 0.50, 0.50, 0.10, 0.90, 0.10, 0.10],  # AU26
], dtype=torch.float32)

SYM_EMOTIONS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]
NUM_SYM_EMOTIONS = len(SYM_EMOTIONS)


def au_to_expr_pseudo(Y_a: torch.Tensor,
                      M_AE: torch.Tensor,
                      neutral_index: int) -> torch.Tensor:
    """
    Y_a: (B, N_a)  - nhãn AU (0/1) hoặc xác suất sau sigmoid
    M_AE: (N_a, N_e) - ma trận AU-Expression
    neutral_index: int - chỉ số class 'Neutral' trong trục expression

    return: Y_e (B, N_e), one-hot pseudo-label expression
    """
    B, N_a = Y_a.shape
    N_e = M_AE.shape[1]

    # Đảm bảo dùng float cho nhân ma trận
    Y_a_float = Y_a.float()

    # (1) Tính score theo Eq.(1): s = Y^a * M_{A-E}
    scores = Y_a_float @ M_AE    # (B, N_e)

    # (2) Chọn ke = argmax(score) cho mỗi sample
    ke = scores.argmax(dim=1)    # (B,)

    # (3) Xử lý các mẫu không có AU nào kích hoạt → Neutral
    neutral_mask = (Y_a_float.sum(dim=1) == 0)   # (B,)
    ke[neutral_mask] = neutral_index

    # (4) Tạo one-hot Y^e theo Eq.(2)
    Y_e = torch.zeros(B, N_e, device=Y_a.device, dtype=Y_a_float.dtype)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)

    return Y_e  # shape (B, N_e)


# ================================================================
# Backward-compatible aliases for ctrlau.py imports
# ================================================================
SymAUHead = AUHead
SymExprHead = ExprHead


# ================================================================
# SymGraphAUModel: Wrapping MEFARGStage1 with train.py interface
# ================================================================

class SymGraphAUModel(nn.Module):
    """
    Pure MultiviewSymAU Stage 1 model matching MEFARGStage1 100%.
    """
    def __init__(self, cfg=None):
        super().__init__()
        from config import ModelConfig, NUM_AUS
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        self.phase = getattr(cfg, "phase", 1)
        self.num_aus = NUM_AUS
        self.num_emotions = NUM_SYM_EMOTIONS

        # The exact Stage 1 architecture
        self.stage1 = MEFARGStage1(num_aus=self.num_aus, num_expr=self.num_emotions, backbone='resnet50')

        # Expose submodules for compatibility
        self.backbone = self.stage1.backbone
        self.global_linear = self.stage1.global_linear
        self.au_head = self.stage1.au_head
        self.emotion_head = self.stage1.expr_head

        self.register_buffer("M_AE", M_AE_DISFA)

        # Default weights
        raw_au_pos_weights = torch.tensor([
            19.11, 22.18, 5.56, 11.67, 22.90, 6.76, 2.61, 10.34
        ])
        wal_weights = raw_au_pos_weights / raw_au_pos_weights.sum() * self.num_aus
        self.register_buffer("wal_weights", wal_weights)
        self.wal_loss = WeightedAsymmetricLoss(weight=self.wal_weights)
        self.expression_bce_loss = ExpressionBCELoss()

    def update_class_weights(self, weights: torch.Tensor):
        w = weights / weights.sum() * self.num_aus
        self.wal_weights.copy_(w)
        self.wal_loss.weight = self.wal_weights

    def set_phase(self, phase: int):
        self.phase = phase

    def forward(self, images, au_labels=None, phase=None):
        V_a, V_e, outputs_AU, outputs_Emo = self.stage1(images)

        losses = {}
        emotion_pseudo = None
        if au_labels is not None:
            targets_Emo = au_to_expr_pseudo(au_labels, self.M_AE, neutral_index=6)
            loss_wa = self.wal_loss(outputs_AU, au_labels.float())
            loss_we = self.expression_bce_loss(outputs_Emo, targets_Emo)
            gamma = getattr(self.cfg, "lam", 0.05)
            loss_phase1 = loss_wa + gamma * loss_we

            losses["loss_wa"] = loss_wa
            losses["loss_au"] = loss_wa
            losses["loss_we"] = loss_we
            losses["loss_emotion"] = loss_we
            losses["loss_phase1"] = loss_phase1
            losses["total_loss"] = loss_phase1
            emotion_pseudo = targets_Emo
        else:
            emotion_pseudo = au_to_expr_pseudo((outputs_AU > 0.5).float(), self.M_AE, neutral_index=6)

        return {
            "au_probs": outputs_AU,
            "au_logits": outputs_AU,
            "emotion_probs": outputs_Emo,
            "emotion_pseudo": emotion_pseudo,
            "V_a": V_a,
            "V_e": V_e,
            "losses": losses,
        }
