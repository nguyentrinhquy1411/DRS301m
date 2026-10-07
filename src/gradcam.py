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
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Đăng ký Hook để tự động lưu activations và gradients
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] chứa d(Loss hoặc Score) / d(Feature Map)
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Tạo Heatmap cho một ảnh đơn lẻ (input_tensor shape: [1, 1, 48, 48]).
        
        Trả về:
        - heatmap: Ma trận 2D kích thước 48x48 trong dải [0.0, 1.0]
        - pred_class: Chỉ số cảm xúc dự đoán (0 đến 6)
        - probabilities: Mảng xác suất của cả 7 lớp cảm xúc
        """
        self.model.zero_grad()
        
        # 1. Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
        pred_class = int(logits.argmax(dim=1).item())

        # Nếu không chỉ định target_class, ta giải thích cho chính lớp mô hình tự đoán
        if target_class is None:
            target_class = pred_class

        # 2. Backward pass từ điểm số của lớp target
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # 3. Tính trọng số alpha_k = GAP của gradients: [1, Channels, H, W] -> [1, Channels, 1, 1]
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 4. Tổng hợp có trọng số: sum(alpha_k * A^k)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # 5. Áp dụng hàm kích hoạt ReLU (chỉ giữ lại tác động tích cực)
        cam = F.relu(cam)

        # 6. Chuẩn hóa về [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # 7. Phóng to (Upsample) Heatmap lên kích thước của ảnh gốc (48x48)
        heatmap = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return heatmap, pred_class, probs


def overlay_heatmap_on_image(
    original_grayscale: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trộn Heatmap màu lên ảnh khuôn mặt xám gốc.
    
    Tham số:
    - original_grayscale: Ảnh mặt xám 2D (48x48) giá trị [0, 255]
    - heatmap: Ma trận 2D (48x48) giá trị [0.0, 1.0]
    - alpha: Tỷ lệ hòa trộn (0.5 = 50% ảnh gốc + 50% heatmap)
    
    Trả về:
    - colored_heatmap: Ảnh heatmap màu RGB (48x48x3)
    - overlay_image: Ảnh khuôn mặt đã phủ heatmap RGB (48x48x3)
    """
    # 1. Đưa heatmap về dạng số nguyên [0, 255] và phủ màu Jet
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # 2. Đưa ảnh gốc sang RGB nếu đang là 1 kênh
    if len(original_grayscale.shape) == 2:
        orig_rgb = cv2.cvtColor(original_grayscale, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = original_grayscale

    # 3. Phép hòa trộn Alpha Blending: (1 - alpha)*Ảnh_gốc + alpha*Heatmap
    overlay = np.uint8((1.0 - alpha) * orig_rgb + alpha * colored_heatmap)

    return colored_heatmap, overlay


if __name__ == "__main__":
    print("=" * 65)
    print("🔬 KIỂM THỬ THUẬT TOÁN GRAD-CAM")
    print("=" * 65)

    from models import ImprovedCNN
    
    # Khởi tạo mô hình và lấy lớp Conv cuối
    model = ImprovedCNN(num_classes=7)
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    # Giả lập 1 ảnh khuôn mặt đầu vào
    dummy_face = torch.randn(1, 1, 48, 48)
    
    # Tạo heatmap
    heatmap, pred_cls, probs = grad_cam.generate_heatmap(dummy_face)
    
    print(f"• Dự đoán cảm xúc ban đầu: Lớp {pred_cls} ({probs[pred_cls]*100:.2f}%)")
    print(f"• Kích thước Heatmap sinh ra: {heatmap.shape} (Đúng chuẩn 48x48)")
    print(f"• Dải giá trị Heatmap       : [{heatmap.min():.2f}, {heatmap.max():.2f}]")

    # Kiểm tra hàm phủ màu
    dummy_orig = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    colored_hm, overlay = overlay_heatmap_on_image(dummy_orig, heatmap)
    print(f"• Kích thước ảnh Overlay     : {overlay.shape} (RGB)")
    print("=" * 65)
    print("✅ GRAD-CAM HOẠT ĐỘNG HOÀN HẢO!")
