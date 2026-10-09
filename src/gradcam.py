"""
Module: src/gradcam.py
Mục đích:
1. Hiện thực hóa thuật toán Explainable AI: Grad-CAM (Gradient-weighted Class Activation Mapping).
2. Trích xuất Feature Maps và Gradients từ lớp Convolutional cuối cùng.
3. Tạo bản đồ nhiệt (Heatmap) và phủ màu (Overlay) lên ảnh khuôn mặt gốc để giải thích quyết định của AI.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Đảm bảo in tiếng Việt trên Terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class GradCAM:
    """
    Lớp xử lý giải thích mô hình bằng Grad-CAM.

    BẢN CHẤT TOÁN HỌC:
    1. Forward: Bắt lấy Feature Map (A^k) của lớp Conv cuối cùng.
    2. Backward: Tính đạo hàm của điểm số cảm xúc (y^c) đối với Feature Map: d(y^c) / d(A^k).
    3. Trọng số kênh alpha_k^c: Lấy trung bình cộng (GAP) của Gradient theo không gian (H x W).
    4. Heatmap: L_Grad-CAM = ReLU( sum( alpha_k^c * A^k ) ).
       (ReLU loại bỏ các điểm ảnh làm giảm điểm số, chỉ giữ lại các điểm ảnh đóng góp tích cực vào dự đoán).

    Gradient được bắt bằng tensor hook trên output của lớp mục tiêu (thay vì
    register_full_backward_hook) để tương thích với các activation in-place của timm/torchvision.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        self.gradients = None
        self.activations = None
        self._handle = self.target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        self._handle.remove()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Tạo Heatmap cho một ảnh đơn lẻ (input_tensor shape: [1, C, H, W]).

        Trả về:
        - heatmap: Ma trận 2D kích thước H x W trong dải [0.0, 1.0]
        - pred_class: Chỉ số cảm xúc dự đoán
        - probabilities: Mảng xác suất của các lớp cảm xúc
        """
        self.model.zero_grad()

        # 1. Forward pass (bật grad kể cả khi caller đang ở trong torch.no_grad())
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
            pred_class = int(logits.argmax(dim=1).item())

            # Nếu không chỉ định target_class, ta giải thích cho chính lớp mô hình tự đoán
            if target_class is None:
                target_class = pred_class

            # 2. Backward pass từ điểm số của lớp target
            logits[0, target_class].backward()

        # 3. Tính trọng số alpha_k = GAP của gradients: [1, Channels, H, W] -> [1, Channels, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Tổng hợp có trọng số + ReLU (chỉ giữ lại tác động tích cực)
        cam = F.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))

        # 5. Chuẩn hóa về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # 6. Phóng to (Upsample) Heatmap lên kích thước của ảnh đầu vào
        heatmap = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        self.model.zero_grad(set_to_none=True)
        return heatmap, pred_class, probs


def overlay_heatmap_on_image(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trộn Heatmap màu lên ảnh khuôn mặt gốc.

    Tham số:
    - original_image: Ảnh mặt RGB [H, W, 3] hoặc ảnh xám [H, W], giá trị [0, 255]
    - heatmap: Ma trận 2D [h, w] giá trị [0.0, 1.0] (tự resize theo ảnh gốc)
    - alpha: Tỷ lệ hòa trộn (0.5 = 50% ảnh gốc + 50% heatmap)

    Trả về:
    - colored_heatmap: Ảnh heatmap màu RGB
    - overlay_image: Ảnh khuôn mặt đã phủ heatmap RGB
    """
    if len(original_image.shape) == 2:
        orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = original_image
    if heatmap.shape[:2] != orig_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1.0 - alpha) * orig_rgb + alpha * colored_heatmap)
    return colored_heatmap, overlay


if __name__ == "__main__":
    print("=" * 65)
    print("🔬 KIỂM THỬ THUẬT TOÁN GRAD-CAM")
    print("=" * 65)

    from models import PretrainedFER

    model = PretrainedFER("mobilenet_v3_large", pretrained=False)
    grad_cam = GradCAM(model=model, target_layer=model.get_last_conv_layer())
    dummy_face = torch.randn(1, 3, 224, 224)
    heatmap, pred_cls, probs = grad_cam.generate_heatmap(dummy_face)

    print(f"• Dự đoán: Lớp {pred_cls} ({probs[pred_cls]*100:.2f}%)")
    print(f"• Kích thước Heatmap: {heatmap.shape}")
    print(f"• Dải giá trị Heatmap: [{heatmap.min():.2f}, {heatmap.max():.2f}]")
    print("=" * 65)
    print("✅ GRAD-CAM HOẠT ĐỘNG!")
