"""
Module: src/inference.py
Mục đích: Phần suy luận dùng chung cho train.py / evaluate.py / app.py / webcam_app.py
1. predict_loader(): suy luận cả DataLoader, hỗ trợ Test-Time Augmentation (lật ngang).
2. FERPredictor: nạp checkpoint bất kỳ backbone, tiền xử lý đúng kích thước/chuẩn hoá
   lúc train, dự đoán theo batch nhiều khuôn mặt + Grad-CAM.
3. FaceTracker: gán ID ổn định cho từng khuôn mặt qua các frame (IoU) và làm mượt
   xác suất theo thời gian (EMA) -> nhãn không nhảy loạn giữa các frame video/webcam.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from models import load_checkpoint_model, get_input_size
from gradcam import GradCAM
from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_loader(model, loader, device, tta: bool = False, amp: bool = False):
    """Trả về (logits [N, C] float32 CPU, labels [N]). TTA = trung bình logits ảnh gốc + ảnh lật ngang."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images).float()
            if tta:
                logits = 0.5 * (logits + model(torch.flip(images, dims=[3])).float())
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


class FERPredictor:
    """Bọc một checkpoint để dùng trong app: tiền xử lý + dự đoán batch + Grad-CAM."""
    def __init__(self, ckpt_path, device: torch.device | None = None):
        self.device = device or get_device()
        self.model, self.ckpt = load_checkpoint_model(ckpt_path, self.device)
        self.model_name = self.ckpt.get("model_name", "mobilenet_v3_large")
        self.input_size = int(self.ckpt.get("img_size", get_input_size(self.model_name)))
        self.grayscale = self.model_name in ("baseline", "improved")
        self.val_f1 = float(self.ckpt.get("best_val_f1", 0.0))
        self.grad_cam = GradCAM(self.model, self.model.get_last_conv_layer())
        if self.grayscale:
            mean, std = [0.5], [0.5]
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.mean = torch.tensor(mean, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, -1, 1, 1)

    def preprocess(self, crops_rgb: list[np.ndarray]) -> tuple[torch.Tensor, list[np.ndarray]]:
        """List ảnh RGB uint8 -> tensor [N, C, S, S] đã chuẩn hoá + list ảnh RGB đã resize (để vẽ Grad-CAM)."""
        s = self.input_size
        resized = [cv2.resize(np.asarray(c), (s, s), interpolation=cv2.INTER_AREA if c.shape[0] > s else cv2.INTER_LINEAR)
                   for c in crops_rgb]
        arr = np.stack(resized).astype(np.float32) / 255.0
        if self.grayscale:
            arr = (arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return (x - self.mean) / self.std, resized

    @torch.no_grad()
    def predict(self, crops_rgb: list[np.ndarray], tta: bool = True) -> np.ndarray:
        """Trả về ma trận xác suất [N, 7]."""
        if not crops_rgb:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        x, _ = self.preprocess(crops_rgb)
        logits = self.model(x)
        if tta:
            logits = 0.5 * (logits + self.model(torch.flip(x, dims=[3])))
        return F.softmax(logits.float(), dim=1).cpu().numpy()

    def explain(self, crop_rgb: np.ndarray, tta: bool = True):
        """Grad-CAM cho 1 khuôn mặt. Trả về (heatmap [S,S], pred_idx, probs [7], ảnh RGB đã resize)."""
        x, resized = self.preprocess([crop_rgb])
        probs = self.predict([crop_rgb], tta=tta)[0]
        pred_idx = int(np.argmax(probs))
        heatmap, _, _ = self.grad_cam.generate_heatmap(x, target_class=pred_idx)
        return heatmap, pred_idx, probs, resized[0]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """
    Tracker IoU tham lam + EMA xác suất cho từng khuôn mặt.
    alpha: trọng số của frame mới (0.4 -> ~ trung bình 2-3 frame gần nhất).
    """
    def __init__(self, alpha: float = 0.4, iou_threshold: float = 0.3, max_missed: int = 10):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}   # id -> {"bbox", "probs", "missed"}
        self._next_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def update(self, bboxes: list, probs: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Nhận bbox + xác suất frame hiện tại, trả về [(track_id, xác suất đã làm mượt)] theo thứ tự đầu vào."""
        pairs = sorted(
            ((_iou(b, t["bbox"]), i, tid) for i, b in enumerate(bboxes) for tid, t in self.tracks.items()),
            reverse=True,
        )
        assigned, used = {}, set()
        for iou, i, tid in pairs:
            if iou < self.iou_threshold:
                break
            if i in assigned or tid in used:
                continue
            assigned[i] = tid
            used.add(tid)

        results = []
        for i, b in enumerate(bboxes):
            p = np.asarray(probs[i], dtype=np.float32)
            if i in assigned:
                t = self.tracks[assigned[i]]
                t["probs"] = (1 - self.alpha) * t["probs"] + self.alpha * p
                t["bbox"], t["missed"] = b, 0
                tid = assigned[i]
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {"bbox": b, "probs": p, "missed": 0}
                used.add(tid)
            results.append((tid, self.tracks[tid]["probs"].copy()))

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return results
