"""
Module: src/models.py
Mục đích:
1. Định nghĩa Baseline CNN (mốc đối chứng benchmark 48x48 ảnh xám).
2. Định nghĩa Custom Improved CNN (tích hợp BatchNorm, Dropout, GAP).
3. ĐỊNH NGHĨA PRE-TRAINED BACKBONES & TRANSFER LEARNING:
   - mobilenet_v3_large, resnet18 : pre-train ImageNet (torchvision)
   - efficientnet_b0              : pre-train ImageNet (timm)
   - hsemotion_b0, hsemotion_b2   : EfficientNet pre-train trên KHUÔN MẶT (VGGFace2) rồi
                                    fine-tune nhận diện cảm xúc trên AffectNet (HSEmotion).
     -> Backbone đã "hiểu" cơ mặt nên fine-tune trên RAF-DB tốt hơn hẳn backbone ImageNet.
4. Giao diện thống nhất forward_features() / forward_head() phục vụ Grad-CAM,
   flip-consistency loss và chia learning rate backbone/head.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights
)

# Đảm bảo in tiếng Việt trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ====================================================================
# 1. BASELINE CNN (Mô hình cơ sở làm mốc đối chứng)
# ====================================================================
class BaselineCNN(nn.Module):
    """Vanilla CNN 3 khối truyền thống"""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

    def get_last_conv_layer(self) -> nn.Module:
        return self.conv3


# ====================================================================
# 2. CUSTOM IMPROVED CNN (BatchNorm, Dropout, GAP)
# ====================================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

    def get_last_conv_layer(self) -> nn.Module:
        return self.block3[3]


# ====================================================================
# 3. PRE-TRAINED BACKBONE & TRANSFER LEARNING
# ====================================================================
TIMM_BACKBONES = {
    # tên trong project : (kiến trúc timm, file HSEmotion | None, kích thước ảnh vào)
    "efficientnet_b0": ("efficientnet_b0", None, 224),
    "hsemotion_b0": ("tf_efficientnet_b0", "enet_b0_8_best_afew", 224),
    "hsemotion_b2": ("tf_efficientnet_b2", "enet_b2_7", 260),
}
TORCHVISION_BACKBONES = ("mobilenet_v3_large", "resnet18")
PRETRAINED_BACKBONES = TORCHVISION_BACKBONES + tuple(TIMM_BACKBONES)

# Thứ tự lớp của HSEmotion. Thứ tự 7 lớp của project: angry, disgust, fear, happy, neutral, sad, surprise
# -> mô hình 8 lớp bỏ "Contempt" (index 1); mô hình 7 lớp trùng thứ tự với project.
HSE_TO_PROJECT_ROWS = {8: [0, 2, 3, 4, 5, 6, 7], 7: [0, 1, 2, 3, 4, 5, 6]}
HSE_URL = "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/{}.pt?raw=true"


def get_input_size(model_name: str) -> int:
    name = model_name.lower()
    if name in ("baseline", "improved"):
        return 48
    if name in TIMM_BACKBONES:
        return TIMM_BACKBONES[name][2]
    return 224


def _load_hsemotion_weights(hse_name: str):
    """Tải file .pt của HSEmotion (cache ở ~/.hsemotion) và trả về (state_dict backbone, W, b classifier)."""
    import urllib.request
    cache_dir = os.path.join(os.path.expanduser("~"), ".hsemotion")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, hse_name + ".pt")
    if not os.path.isfile(path):
        print(f"📥 Đang tải trọng số HSEmotion {hse_name} ...")
        urllib.request.urlretrieve(HSE_URL.format(hse_name), path)
    # File HSEmotion là cả mô hình timm được pickle (nguồn: repo GitHub chính thức của HSEmotion).
    # Chỉ lấy state_dict; mô hình được dựng lại bằng timm hiện tại.
    full = torch.load(path, map_location="cpu", weights_only=False)
    sd = full.state_dict()
    fc = full.classifier[0] if isinstance(full.classifier, nn.Sequential) else full.classifier
    backbone_sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    return backbone_sd, fc.weight.detach().clone(), fc.bias.detach().clone()


class PretrainedFER(nn.Module):
    """
    Mô hình nhận diện cảm xúc khai thác Pre-trained Vision Backbones.
    - head='mlp'   : Linear(D,256) -> BN -> Act -> Dropout -> Linear(256,C)
    - head='linear': Dropout -> Linear(D,C). Với hsemotion_* lớp Linear được khởi tạo
                     từ classifier AffectNet nên mô hình đã dự đoán được cảm xúc ngay từ epoch 0.
    Mặc định: torchvision backbones dùng 'mlp' (tương thích checkpoint cũ), timm dùng 'linear'.
    """
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head: str | None = None,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.input_size = get_input_size(self.backbone_name)
        self.is_timm = self.backbone_name in TIMM_BACKBONES
        if head is None:
            head = "linear" if self.is_timm else "mlp"
        self.head_type = head

        if self.backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = tv_models.mobilenet_v3_large(weights=weights)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = self._make_head(in_features, nn.Hardswish, drop_rate=0.4)

        elif self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = tv_models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = self._make_head(in_features, nn.ReLU, drop_rate=0.4)

        elif self.is_timm:
            import timm
            arch, hse_name, _ = TIMM_BACKBONES[self.backbone_name]
            use_timm_pretrained = pretrained and hse_name is None
            self.model = timm.create_model(arch, pretrained=use_timm_pretrained, num_classes=0,
                                           drop_path_rate=drop_path_rate)
            in_features = self.model.num_features
            self.classifier = self._make_head(in_features, nn.SiLU, drop_rate=drop_rate)

            if pretrained and hse_name is not None:
                backbone_sd, W, b = _load_hsemotion_weights(hse_name)
                self.model.load_state_dict(backbone_sd, strict=True)
                rows = HSE_TO_PROJECT_ROWS.get(W.shape[0])
                if self.head_type == "linear" and rows is not None and num_classes == 7:
                    with torch.no_grad():
                        self.classifier[-1].weight.copy_(W[rows])
                        self.classifier[-1].bias.copy_(b[rows])
        else:
            raise ValueError(f"Backbone '{backbone_name}' không được hỗ trợ. Chọn: {PRETRAINED_BACKBONES}")

        if freeze_backbone:
            self.freeze_backbone()

    def _make_head(self, in_features: int, act, drop_rate: float) -> nn.Sequential:
        if self.head_type == "linear":
            return nn.Sequential(nn.Dropout(p=drop_rate), nn.Linear(in_features, self.num_classes))
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            act(),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, self.num_classes)
        )

    # ---------------- forward ----------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về feature map cuối cùng [B, C, H, W] (trước global pooling)."""
        if self.backbone_name == "mobilenet_v3_large":
            return self.model.features(x)
        if self.backbone_name == "resnet18":
            m = self.model
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return self.model.forward_features(x)

    def forward_head(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(nn.functional.adaptive_avg_pool2d(fmap, 1), 1)
        return self.head_module()(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    # ---------------- param groups / freeze ----------------
    def head_module(self) -> nn.Module:
        if self.backbone_name == "mobilenet_v3_large":
            return self.model.classifier
        if self.backbone_name == "resnet18":
            return self.model.fc
        return self.classifier

    def head_parameters(self):
        return list(self.head_module().parameters())

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def freeze_backbone(self):
        """Đóng băng Backbone, chỉ giữ lại Classifier Head (Giai đoạn 1)."""
        for p in self.backbone_parameters():
            p.requires_grad = False
        for p in self.head_parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        """Mở khóa toàn bộ mô hình để tiến hành Fine-Tuning sâu (Giai đoạn 2)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self) -> nn.Module:
        """Module có output chính là feature map cuối (sau activation) để hook Grad-CAM."""
        if self.backbone_name == "mobilenet_v3_large":
            return self.model.features[-1]
        if self.backbone_name == "resnet18":
            return self.model.layer4
        return self.model.bn2


def build_model(model_name: str, num_classes: int = 7, pretrained: bool = True, **kwargs) -> nn.Module:
    name = model_name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes=num_classes)
    return PretrainedFER(backbone_name=name, num_classes=num_classes, pretrained=pretrained, **kwargs)


def load_checkpoint_model(ckpt_path, device: torch.device | str = "cpu"):
    """Dựng lại mô hình từ checkpoint (đọc tên backbone + kiểu head từ metadata). Trả về (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    kwargs = {}
    if model_name in PRETRAINED_BACKBONES and "head" in ckpt:
        kwargs["head"] = ckpt["head"]
    model = build_model(model_name, num_classes=len(ckpt.get("classes", [0] * 7)), pretrained=False, **kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Đếm tổng số tham số và số tham số có thể huấn luyện."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("=" * 65)
    print("🔬 KIỂM TRA CÁC BACKBONE PRE-TRAINED")
    print("=" * 65)
    for name in PRETRAINED_BACKBONES:
        m = PretrainedFER(backbone_name=name, num_classes=7, pretrained=False).eval()
        s = m.input_size
        x = torch.randn(2, 3, s, s)
        total, _ = count_parameters(m)
        with torch.no_grad():
            fmap = m.forward_features(x)
            out = m.forward_head(fmap)
        print(f"• {name:<20} params={total/1e6:5.2f}M  input={s}  fmap={tuple(fmap.shape)}  out={tuple(out.shape)}")
    print("=" * 65)
