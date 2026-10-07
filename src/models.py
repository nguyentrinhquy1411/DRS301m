"""
Module: src/models.py
Mục đích:
1. Định nghĩa Baseline CNN (mốc đối chứng benchmark 48x48).
2. Định nghĩa Custom Improved CNN (tích hợp BatchNorm, Dropout, GAP).
3. ĐỊNH NGHĨA PRE-TRAINED BACKBONES & TRANSFER LEARNING (MobileNetV3-Large, ResNet-18)
   phục vụ nhận diện cảm xúc đa khuôn mặt thời gian thực trên dữ liệu thật RAF-DB.
4. Hỗ trợ trích xuất lớp Conv cuối cùng phục vụ thuật toán Grad-CAM cho mọi kiến trúc.
"""

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
# 3. PRE-TRAINED BACKBONE & TRANSFER LEARNING (SOTA REAL-TIME)
# ====================================================================
class PretrainedFER(nn.Module):
    """
    Mô hình nhận diện cảm xúc khai thác Pre-trained Vision Backbones:
    - Mặc định: MobileNetV3-Large (nhẹ ~21 MB, tốc độ >50 FPS, tối ưu quét nhiều khuôn mặt)
    - Tùy chọn: ResNet-18 (~44 MB)
    - Tích hợp sẵn cơ chế Freeze / Unfreeze cho Two-Stage Transfer Learning.
    - Cung cấp hàm get_last_conv_layer() phục vụ thuật toán Grad-CAM.
    """
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = tv_models.mobilenet_v3_large(weights=weights)
            
            # Thay thế classifier head
            # MobileNetV3-Large pooling cho ra vector 960 chiều
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.Hardswish(),
                nn.Dropout(p=0.4),
                nn.Linear(256, num_classes)
            )

        elif self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = tv_models.resnet18(weights=weights)
            
            # ResNet-18 pooling cho ra vector 512 chiều
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Backbone '{backbone_name}' không được hỗ trợ. Chọn: 'mobilenet_v3_large' hoặc 'resnet18'.")

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self):
        """Đóng băng toàn bộ trọng số của Backbone, chỉ giữ lại Classifier Head (Giai đoạn 1)."""
        if self.backbone_name == "mobilenet_v3_large":
            for param in self.model.features.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif self.backbone_name == "resnet18":
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def unfreeze_backbone(self):
        """Mở khóa toàn bộ mô hình để tiến hành Fine-Tuning sâu (Giai đoạn 2)."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self) -> nn.Module:
        """Trả về lớp Conv cuối cùng của Backbone để hook Grad-CAM."""
        if self.backbone_name == "mobilenet_v3_large":
            # Lớp Conv cuối cùng nằm trong khối features[-1]
            return self.model.features[-1][0]
        elif self.backbone_name == "resnet18":
            return self.model.layer4[-1].conv2


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Đếm tổng số tham số và số tham số có thể huấn luyện."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("=" * 65)
    print("🔬 KIỂM TRA MÔ HÌNH PRE-TRAINED MOBILENET-V3 & RESNET-18")
    print("=" * 65)

    dummy_rgb = torch.randn(2, 3, 224, 224)  # 2 ảnh RGB 224x224

    # 1. MobileNetV3-Large
    mobilenet = PretrainedFER(backbone_name="mobilenet_v3_large", num_classes=7, pretrained=True)
    m_total, m_trainable = count_parameters(mobilenet)
    m_out = mobilenet(dummy_rgb)
    print(f"1. MOBILENET-V3 LARGE:")
    print(f"   • Tổng tham số          : {m_total:,}")
    print(f"   • Tham số có thể train   : {m_trainable:,}")
    print(f"   • Output shape           : {m_out.shape} -> [Batch=2, Classes=7]")
    print(f"   • Target Grad-CAM Conv   : {mobilenet.get_last_conv_layer()}")

    # Thử đóng băng backbone
    mobilenet.freeze_backbone()
    _, m_frozen_trainable = count_parameters(mobilenet)
    print(f"   • Khi đóng băng (Freeze) : {m_frozen_trainable:,} params (Chỉ train Classifier Head!)")

    # 2. ResNet-18
    resnet = PretrainedFER(backbone_name="resnet18", num_classes=7, pretrained=True)
    r_total, r_trainable = count_parameters(resnet)
    r_out = resnet(dummy_rgb)
    print(f"\n2. RESNET-18:")
    print(f"   • Tổng tham số          : {r_total:,}")
    print(f"   • Tham số có thể train   : {r_trainable:,}")
    print(f"   • Output shape           : {r_out.shape}")
    print(f"   • Target Grad-CAM Conv   : {resnet.get_last_conv_layer()}")
    print("=" * 65)
    print("✅ CẢ 2 MÔ HÌNH PRE-TRAINED ĐÃ SẴN SÀNG CHO FINE-TUNING & GRAD-CAM!")
