"""
Module: webcam_app.py
Standalone Real-Time Multi-Face Explainable FER Desktop Application
───────────────────────────────────────────────────────────────────
Chạy trực tiếp trên macOS / Windows / Linux với hiệu năng cao nhất (30-60 FPS).
Không phụ thuộc vào trình duyệt web.

Tính năng:
• Pre-load sẵn CẢ 2 MÔ HÌNH (RAF-DB Standard & Mask-Aware) trên GPU MPS.
• Đổi mô hình tức thì 0ms bằng phím [M] hoặc phím số [1], [2].
• Nhận diện đồng thời nhiều khuôn mặt qua Webcam (Facetime Mac / USB Camera).
• Dự đoán 7 cảm xúc (Happy, Surprise, Neutral, Sad, Fear, Angry, Disgust) thời gian thực.
• Bảng điều khiển HUD bên phải với thanh phần trăm xác suất mượt mà (EMA).
• Explainable AI (Grad-CAM): Hiển thị vùng kích hoạt nơ-ron (mắt, mũi, miệng).
• Phím tắt tương tác:
    [1]: Chọn mô hình Baseline RAF-DB
    [2]: Chọn mô hình Mask-Aware (Khẩu trang)
    [M]: Đổi mô hình qua lại
    [G]: Bật / Tắt Grad-CAM
    [S]: Chụp ảnh màn hình lưu vào output/samples
    [Q] hoặc [ESC]: Thoát
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Đảm bảo UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from models import PretrainedFER
from face_detector import MultiFaceDetector
from gradcam import GradCAM

# ── Hằng số & Bảng màu ──
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# BGR colors for OpenCV
COLORS_BGR = {
    "happy": (113, 204, 46),       # Xanh lá tươi #2ecc71
    "surprise": (15, 196, 241),    # Vàng chanh #f1c40f
    "neutral": (219, 152, 52),     # Xanh dương pastel #3498db
    "sad": (182, 89, 155),         # Tím lavender #9b59b6
    "fear": (34, 126, 230),        # Cam #e67e22
    "angry": (60, 76, 231),        # Đỏ tươi #e74c3c
    "disgust": (166, 165, 149),    # Xám bạc #95a5a6
}

EXPLANATIONS = {
    "happy": "Tap trung khoe mieng cuoi & duoi mat",
    "surprise": "Quet vao mi mat mo to & vom chan may",
    "fear": "Phan ung voi co tran cang & anh mat mo rong",
    "angry": "Tap trung long may nhiu lai & nep nhan tran",
    "disgust": "Phan ung voi co nhan song mui",
    "sad": "Chu y vao dau long may xe & mi mat ru xuong",
    "neutral": "Nhiet phan bo deu - co tha long tu nhien",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_optimal_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WebcamFERApp:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.device = get_optimal_device()
        print(f"🚀 Khởi tạo Real-Time FER Desktop App trên thiết bị: {self.device}")

        # Khởi tạo danh sách mô hình
        self.model_configs = [
            {
                "name": "Baseline RAF-DB (Chuan)",
                "short": "RAF-DB",
                "path": ROOT_DIR / "output" / "best_model_rafdb.pth",
                "color_bgr": (219, 152, 52),  # Xanh dương
                "desc": "Khuon mat binh thuong (khong che)"
            },
            {
                "name": "Mask-Aware FER (Khau trang)",
                "short": "Mask-Aware",
                "path": ROOT_DIR / "output" / "best_model_mask_aware.pth",
                "color_bgr": (113, 204, 46),  # Xanh lá
                "desc": "Toi uu khi deo khau trang / che mieng"
            }
        ]

        self.models = []
        self.gradcams = []
        self._preload_all_models()

        self.active_idx = 0
        self.detector = MultiFaceDetector()

        # Trạng thái điều khiển & debounce
        self.enable_gradcam = True
        self.smoothed_probs = np.zeros(7, dtype=np.float32)
        self.last_key_time = 0.0

        # Thông báo flash trên màn hình
        self.flash_timer = 0
        self.flash_msg = ""
        self.flash_color = (113, 204, 46)

        # Kích thước hiển thị
        self.feed_w, self.feed_h = 960, 620
        self.sidebar_w = 340
        self.total_w = self.feed_w + self.sidebar_w
        self.total_h = self.feed_h

    def _preload_all_models(self):
        """Preload cả 2 mô hình vào VRAM/RAM để chuyển đổi tức thì 0ms."""
        for cfg in self.model_configs:
            name = cfg["name"]
            ckpt_path = cfg["path"]
            print(f"📦 Đang nạp sẵn: [{name}]...")

            m = PretrainedFER("mobilenet_v3_large", num_classes=7, pretrained=False).to(self.device)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                m.load_state_dict(state_dict)
                f1_val = ckpt.get("best_val_f1", 0.0)
                print(f"   ✅ Nạp thành công {ckpt_path.name} (Val F1: {f1_val:.4f})")
            else:
                print(f"   ⚠️ Không tìm thấy {ckpt_path.name}, dùng fallback")

            m.eval()
            gc = GradCAM(m, m.get_last_conv_layer())
            self.models.append(m)
            self.gradcams.append(gc)

        print("⚡ CẢ 2 MÔ HÌNH ĐÃ SẴN SÀNG TRONG BỘ NHỚ! Đổi mô hình tức thì không giật lag.\n")

    def switch_model(self, target_idx: int):
        if target_idx != self.active_idx:
            self.active_idx = target_idx % len(self.model_configs)
            cfg = self.model_configs[self.active_idx]
            self.flash_timer = 35
            self.flash_msg = f"DA DOI SANG: {cfg['name'].upper()}"
            self.flash_color = cfg["color_bgr"]
            print(f"🔄 ĐÃ CHUYỂN SANG MÔ HÌNH: [{cfg['name']}]")

    def preprocess_tensor(self, crop_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (224, 224))
        np_norm = resized_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        return tensor, resized_rgb

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"❌ Không thể mở Camera ID {self.camera_id}!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "XAI-FER Real-Time (FaceTime Mac) - Press [1] [2] or [M] to Switch Model"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.total_w, self.total_h)

        print("=" * 65)
        print("🎥 CAMERA REAL-TIME ĐANG CHẠY:")
        print("  • Phím [1]: Chuyển sang mô hình [1] Baseline RAF-DB")
        print("  • Phím [2]: Chuyển sang mô hình [2] Mask-Aware (Khẩu trang)")
        print("  • Phím [M]: Đổi mô hình qua lại")
        print("  • Phím [G]: Bật / Tắt Grad-CAM")
        print("  • Phím [S]: Chụp ảnh màn hình (Snapshot)")
        print("  • Phím [Q] hoặc [ESC]: Thoát")
        print("=" * 65 + "\n")

        fps_tracker = []
        last_time = time.time()

        primary_face_crop = None
        primary_gradcam_overlay = None
        primary_label = "Chua ro"
        primary_conf = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Lật gương
                frame = cv2.flip(frame, 1)
                orig_h, orig_w = frame.shape[:2]

                # FPS
                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps_tracker.append(1.0 / dt)
                    if len(fps_tracker) > 15:
                        fps_tracker.pop(0)
                current_fps = sum(fps_tracker) / max(1, len(fps_tracker))

                # Thu nhỏ ảnh để detect khuôn mặt siêu tốc (~8-12ms)
                det_w = 480
                det_scale = det_w / float(orig_w)
                det_h = int(orig_h * det_scale)
                small_frame = cv2.resize(frame, (det_w, det_h))

                detected_faces = self.detector.detect_faces(small_frame, is_bgr=True)

                display_feed = cv2.resize(frame, (self.feed_w, self.feed_h))
                scale_x = self.feed_w / float(det_w)
                scale_y = self.feed_h / float(det_h)

                # Mô hình đang chọn
                active_model = self.models[self.active_idx]
                active_gradcam = self.gradcams[self.active_idx]

                faces_info = []

                for idx, face in enumerate(detected_faces):
                    sx, sy, sw, sh = face.bbox
                    fx = int(sx * scale_x)
                    fy = int(sy * scale_y)
                    fw = int(sw * scale_x)
                    fh = int(sh * scale_y)

                    fx = max(0, min(self.feed_w - 1, fx))
                    fy = max(0, min(self.feed_h - 1, fy))
                    fw = max(10, min(self.feed_w - fx, fw))
                    fh = max(10, min(self.feed_h - fy, fh))

                    face_crop_bgr = display_feed[fy:fy + fh, fx:fx + fw]
                    if face_crop_bgr.shape[0] < 10 or face_crop_bgr.shape[1] < 10:
                        continue

                    tensor, rgb_224 = self.preprocess_tensor(face_crop_bgr)

                    if idx == 0 and self.enable_gradcam:
                        try:
                            heatmap, pred_idx, probs = active_gradcam.generate_heatmap(tensor)
                            label = CLASSES[pred_idx]
                            conf = float(probs[pred_idx] * 100)

                            heatmap_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                            bgr_224 = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR)
                            overlay = cv2.addWeighted(bgr_224, 0.6, heatmap_bgr, 0.4, 0)

                            primary_gradcam_overlay = overlay
                            primary_face_crop = cv2.resize(face_crop_bgr, (120, 120))
                            primary_label = label
                            primary_conf = conf
                            self.smoothed_probs = 0.65 * self.smoothed_probs + 0.35 * probs
                        except Exception:
                            with torch.no_grad():
                                logits = active_model(tensor)
                                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                                pred_idx = int(np.argmax(probs))
                                label = CLASSES[pred_idx]
                                conf = float(probs[pred_idx] * 100)
                    else:
                        with torch.no_grad():
                            logits = active_model(tensor)
                            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                            pred_idx = int(np.argmax(probs))
                            label = CLASSES[pred_idx]
                            conf = float(probs[pred_idx] * 100)

                        if idx == 0:
                            primary_face_crop = cv2.resize(face_crop_bgr, (120, 120))
                            primary_label = label
                            primary_conf = conf
                            self.smoothed_probs = 0.65 * self.smoothed_probs + 0.35 * probs

                    faces_info.append({
                        "bbox": (fx, fy, fw, fh),
                        "label": label,
                        "conf": conf,
                        "color": COLORS_BGR.get(label, (0, 255, 0))
                    })

                # Vẽ bounding box
                for info in faces_info:
                    fx, fy, fw, fh = info["bbox"]
                    color = info["color"]
                    label_str = f"{info['label'].upper()} {info['conf']:.0f}%"

                    self._draw_modern_box(display_feed, fx, fy, fw, fh, color)

                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    tag_y = max(0, fy - th - 10)
                    cv2.rectangle(display_feed, (fx, tag_y), (fx + tw + 12, fy), color, -1)
                    cv2.putText(display_feed, label_str, (fx + 6, fy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                # Vẽ nhãn FPS & Model đang chọn góc trái trên feed
                cur_cfg = self.model_configs[self.active_idx]
                badge_txt = f"FPS: {current_fps:.1f} | Model: {cur_cfg['short']}"
                cv2.rectangle(display_feed, (12, 12), (280, 42), (20, 20, 20), -1)
                cv2.rectangle(display_feed, (12, 12), (280, 42), cur_cfg['color_bgr'], 1)
                cv2.putText(display_feed, badge_txt, (20, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                # Banner thông báo khi đổi model hoặc chụp snapshot
                if self.flash_timer > 0:
                    self.flash_timer -= 1
                    cv2.rectangle(display_feed, (self.feed_w // 2 - 250, 15), (self.feed_w // 2 + 250, 58), (20, 24, 32), -1)
                    cv2.rectangle(display_feed, (self.feed_w // 2 - 250, 15), (self.feed_w // 2 + 250, 58), self.flash_color, 2)
                    cv2.putText(display_feed, self.flash_msg, (self.feed_w // 2 - 230, 44),
                                cv2.FONT_HERSHEY_DUPLEX, 0.58, self.flash_color, 1, cv2.LINE_AA)

                # Tạo Sidebar HUD
                sidebar = self._render_sidebar(
                    fps=current_fps,
                    face_count=len(detected_faces),
                    face_crop=primary_face_crop,
                    gradcam_overlay=primary_gradcam_overlay,
                    label=primary_label,
                    conf=primary_conf,
                    probs=self.smoothed_probs
                )

                canvas = np.hstack([display_feed, sidebar])
                cv2.imshow(window_name, canvas)

                # Bắt phím điều khiển (có Debounce chống bấm lặp)
                key = cv2.waitKey(1) & 0xFF
                press_now = time.time()
                if key in (ord('q'), ord('Q'), 27):  # ESC or Q
                    break
                elif key in (ord('m'), ord('M')) and (press_now - self.last_key_time > 0.35):
                    self.last_key_time = press_now
                    self.switch_model((self.active_idx + 1) % len(self.model_configs))
                elif key == ord('1') and (press_now - self.last_key_time > 0.35):
                    self.last_key_time = press_now
                    self.switch_model(0)
                elif key == ord('2') and (press_now - self.last_key_time > 0.35):
                    self.last_key_time = press_now
                    self.switch_model(1)
                elif key in (ord('g'), ord('G')) and (press_now - self.last_key_time > 0.35):
                    self.last_key_time = press_now
                    self.enable_gradcam = not self.enable_gradcam
                    self.flash_timer = 25
                    self.flash_msg = f"GRAD-CAM: {'BAT' if self.enable_gradcam else 'TAT'}"
                    self.flash_color = (15, 196, 241)
                    print(f"🔄 Grad-CAM: {'BẬT' if self.enable_gradcam else 'TẮT'}")
                elif key in (ord('s'), ord('S')) and (press_now - self.last_key_time > 0.35):
                    self.last_key_time = press_now
                    self._save_snapshot(canvas)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🛑 Đã tắt camera và đóng ứng dụng.")

    def _draw_modern_box(self, img, x, y, w, h, color, corner_len=18, thickness=2):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    def _render_sidebar(self, fps, face_count, face_crop, gradcam_overlay, label, conf, probs):
        sb = np.full((self.total_h, self.sidebar_w, 3), (22, 24, 30), dtype=np.uint8)
        cv2.line(sb, (0, 0), (0, self.total_h), (50, 55, 70), 2)

        # Header Title
        cv2.rectangle(sb, (12, 12), (self.sidebar_w - 12, 54), (32, 36, 48), -1)
        cv2.putText(sb, "XAI-FER MONITOR", (24, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        dev_text = "MPS GPU" if "mps" in str(self.device) else ("CUDA" if "cuda" in str(self.device) else "CPU")
        cv2.putText(sb, f"[{dev_text}]", (self.sidebar_w - 95, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 150), 1, cv2.LINE_AA)

        # Highlight Active Model Box
        cur_cfg = self.model_configs[self.active_idx]
        cv2.rectangle(sb, (14, 64), (self.sidebar_w - 14, 98), (30, 34, 46), -1)
        cv2.rectangle(sb, (14, 64), (self.sidebar_w - 14, 98), cur_cfg["color_bgr"], 1)

        model_tag = f"[{self.active_idx + 1}] {cur_cfg['short'].upper()}"
        cv2.putText(sb, model_tag, (24, 86), cv2.FONT_HERSHEY_DUPLEX, 0.52, cur_cfg["color_bgr"], 1, cv2.LINE_AA)
        cv2.putText(sb, f"Faces: {face_count}", (self.sidebar_w - 100, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 180), 1, cv2.LINE_AA)

        # Vùng XAI: Grad-CAM Overlay & Face Crop
        box_y = 110
        cv2.rectangle(sb, (16, box_y), (self.sidebar_w - 16, box_y + 140), (28, 31, 40), -1)
        cv2.rectangle(sb, (16, box_y), (self.sidebar_w - 16, box_y + 140), (45, 50, 65), 1)

        if self.enable_gradcam and gradcam_overlay is not None:
            g_disp = cv2.resize(gradcam_overlay, (120, 120))
            sb[box_y + 10:box_y + 130, 24:144] = g_disp
            cv2.putText(sb, "Grad-CAM XAI", (30, box_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        elif face_crop is not None:
            f_disp = cv2.resize(face_crop, (120, 120))
            sb[box_y + 10:box_y + 130, 24:144] = f_disp
            cv2.putText(sb, "Face Crop", (45, box_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(sb, "No Face", (45, box_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Thông tin cảm xúc nổi bật
        pred_color = COLORS_BGR.get(label, (180, 180, 180))
        cv2.putText(sb, "DOMINANT:", (156, box_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1, cv2.LINE_AA)
        cv2.putText(sb, label.upper(), (156, box_y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.75, pred_color, 2, cv2.LINE_AA)
        cv2.putText(sb, f"{conf:.1f}% Confidence", (156, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Giải thích trực quan
        expl_text = EXPLANATIONS.get(label, "")
        if expl_text:
            cv2.putText(sb, expl_text[:28], (20, box_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 200), 1, cv2.LINE_AA)

        # Bảng thanh xác suất 7 cảm xúc
        bar_start_y = box_y + 175
        cv2.putText(sb, "EMOTION PROBABILITIES", (20, bar_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 140, 160), 1, cv2.LINE_AA)

        max_bar_w = 175
        for i, emo in enumerate(CLASSES):
            y = bar_start_y + 18 + (i * 26)
            prob_val = float(probs[i]) if len(probs) == 7 else 0.0
            bar_w = int(prob_val * max_bar_w)
            c = COLORS_BGR.get(emo, (100, 100, 100))

            is_winner = (emo == label)
            text_color = (255, 255, 255) if is_winner else (160, 160, 170)
            cv2.putText(sb, f"{emo[:7].capitalize()}:", (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1 if not is_winner else 2, cv2.LINE_AA)

            bar_x = 90
            cv2.rectangle(sb, (bar_x, y), (bar_x + max_bar_w, y + 12), (38, 42, 54), -1)
            if bar_w > 0:
                cv2.rectangle(sb, (bar_x, y), (bar_x + bar_w, y + 12), c, -1)
            cv2.putText(sb, f"{prob_val * 100:.0f}%", (bar_x + max_bar_w + 6, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

        # Bảng phím tắt phía dưới
        key_y = self.total_h - 70
        cv2.rectangle(sb, (12, key_y), (self.sidebar_w - 12, self.total_h - 10), (28, 31, 40), -1)
        cv2.putText(sb, "[1] RAF-DB   [2] Mask-Aware", (20, key_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(sb, "[M] Toggle   [G] Grad-CAM  [Q] Exit", (20, key_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 165, 175), 1, cv2.LINE_AA)

        return sb

    def _save_snapshot(self, full_canvas):
        samples_dir = ROOT_DIR / "output" / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = samples_dir / f"snapshot_{ts}.png"
        cv2.imwrite(str(save_path), full_canvas)
        self.flash_timer = 25
        self.flash_msg = f"DA LUU SNAPSHOT_{ts}.PNG!"
        self.flash_color = (46, 204, 113)
        print(f"📸 Đã lưu ảnh chụp: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time FER Webcam App")
    parser.add_argument("--camera", type=int, default=0, help="ID Webcam (0: FaceTime HD)")
    args = parser.parse_args()

    app = WebcamFERApp(camera_id=args.camera)
    app.run()
