"""
Module: src/face_detector.py
Mục đích:
1. Phát hiện đồng thời NHIỀU KHUÔN MẶT trong khung hình (Real-Time Multi-Face).
2. Detector chính: OpenCV YuNet (CNN nhẹ, trả về 5 landmark: 2 mắt, mũi, 2 khóe miệng).
   Dự phòng: Haar Cascade nếu không tải được model YuNet.
3. CĂN CHỈNH KHUÔN MẶT (Face Alignment) giống hệt cách RAF-DB "aligned" được tạo:
   dùng phép biến đổi tương tự (similarity transform) đưa 5 landmark về template chuẩn.
   -> Ảnh đưa vào mô hình lúc chạy app có cùng phân phối với ảnh lúc train,
      loại bỏ độ lệch train/inference do crop Haar + padding gây ra.
"""

import sys
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image

# Đảm bảo UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Template 5 điểm của RAF-DB aligned (toạ độ chuẩn hoá [0, 1] theo cạnh ảnh).
# Đo thực nghiệm: chạy YuNet trên ~1,000 ảnh RAF-DB test 100x100 rồi lấy trung bình
# (độ lệch chuẩn ~0.03 -> RAF-DB đúng là đã được căn chỉnh theo template cố định).
# Thứ tự điểm giữ nguyên theo YuNet: mắt (trái ảnh), mắt (phải ảnh), mũi, khóe miệng trái, khóe miệng phải.
RAF_TEMPLATE = np.array([
    [0.2696, 0.3213],
    [0.7026, 0.3157],
    [0.4809, 0.5654],
    [0.2996, 0.7345],
    [0.6892, 0.7288],
], dtype=np.float32)


@dataclass
class DetectedFace:
    face_id: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h) theo toạ độ frame gốc
    crop_rgb: np.ndarray              # Ảnh mặt đã căn chỉnh [S, S, 3] RGB
    crop_pil: Image.Image
    score: float = 1.0
    landmarks: np.ndarray | None = field(default=None, repr=False)  # [5, 2] toạ độ frame gốc
    aligned: bool = False


def _download(url: str, dst: Path) -> bool:
    try:
        print(f"📥 Đang tải {dst.name} ...")
        urllib.request.urlretrieve(url, str(dst))
        return True
    except Exception as e:
        print(f"⚠️ Không tải được {dst.name}: {e}")
        return False


def align_face(frame_rgb: np.ndarray, landmarks: np.ndarray, out_size: int = 224) -> np.ndarray:
    """Căn chỉnh khuôn mặt về template RAF-DB bằng similarity transform (xoay + scale + dịch)."""
    dst = RAF_TEMPLATE * out_size
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("Không ước lượng được phép biến đổi căn chỉnh")
    return cv2.warpAffine(frame_rgb, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def square_crop(frame_rgb: np.ndarray, bbox, scale: float = 1.0, out_size: int = 224) -> np.ndarray:
    """Crop vuông quanh bbox (dùng khi không có landmark), pad biên nếu tràn khung."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * scale
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    H, W = frame_rgb.shape[:2]
    pad = max(0, -x1, -y1, x2 - W, y2 - H)
    if pad > 0:
        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size))


class MultiFaceDetector:
    """
    Bộ phát hiện đa khuôn mặt thời gian thực:
    - Nhận khung hình RGB hoặc BGR (độ phân giải bất kỳ).
    - Tự thu nhỏ frame về `det_width` để detect nhanh, nhưng crop/căn chỉnh trên frame gốc.
    - Trả về danh sách DetectedFace với bbox theo toạ độ frame gốc và ảnh mặt đã căn chỉnh.
    """
    def __init__(
        self,
        backend: str = "auto",          # 'auto' | 'yunet' | 'haar'
        score_threshold: float = 0.7,
        min_face: int = 32,
        det_width: int = 640,
        out_size: int = 224,
    ):
        self.score_threshold = score_threshold
        self.min_face = min_face
        self.det_width = det_width
        self.out_size = out_size
        self.backend = None
        self.yunet = None
        self.haar = None

        if backend in ("auto", "yunet"):
            if YUNET_PATH.exists() or _download(YUNET_URL, YUNET_PATH):
                try:
                    self.yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold, 0.3, 5000)
                    self.backend = "yunet"
                except Exception as e:
                    print(f"⚠️ Không khởi tạo được YuNet ({e}), chuyển sang Haar Cascade.")
            if self.backend is None and backend == "yunet":
                raise RuntimeError("Không khởi tạo được YuNet")

        if self.backend is None:
            if not CASCADE_PATH.exists():
                _download(CASCADE_URL, CASCADE_PATH)
            self.haar = cv2.CascadeClassifier(str(CASCADE_PATH))
            self.backend = "haar"

    def _detect_raw(self, small_bgr: np.ndarray):
        """Trả về list (bbox[x,y,w,h], landmarks[5,2] | None, score) theo toạ độ ảnh nhỏ."""
        if self.backend == "yunet":
            h, w = small_bgr.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(small_bgr)
            out = []
            if faces is not None:
                for f in faces:
                    out.append((f[0:4], f[4:14].reshape(5, 2), float(f[14])))
            return out

        gray = cv2.equalizeHist(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY))
        boxes = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face, self.min_face))
        return [(np.array(b, dtype=np.float32), None, 1.0) for b in boxes]

    def detect_faces(self, frame: np.ndarray, is_bgr: bool = True) -> list[DetectedFace]:
        """
        Phát hiện đồng thời tất cả khuôn mặt trong frame.

        Parameters:
        - frame: Ảnh numpy [H, W, 3] (BGR từ OpenCV hoặc RGB từ Streamlit/PIL)
        - is_bgr: True nếu là định dạng OpenCV mặc định

        Returns:
        - Danh sách DetectedFace (sắp xếp theo diện tích giảm dần, mặt lớn nhất là #1)
        """
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if is_bgr else cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR if is_bgr else cv2.COLOR_RGBA2RGB)

        frame_bgr = frame if is_bgr else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if is_bgr else frame

        H, W = frame_bgr.shape[:2]
        scale = min(1.0, self.det_width / float(W))
        small = cv2.resize(frame_bgr, (int(W * scale), int(H * scale))) if scale < 1.0 else frame_bgr

        raw = []
        for box, lm, score in self._detect_raw(small):
            box = box / scale
            if min(box[2], box[3]) < self.min_face:
                continue
            raw.append((box, None if lm is None else lm / scale, score))
        raw.sort(key=lambda r: -(r[0][2] * r[0][3]))

        detected = []
        for idx, (box, lm, score) in enumerate(raw):
            x, y, w, h = [int(round(v)) for v in box]
            x, y = max(0, x), max(0, y)
            w, h = min(W - x, w), min(H - y, h)
            if w <= 0 or h <= 0:
                continue
            aligned = False
            crop = None
            if lm is not None:
                try:
                    crop = align_face(frame_rgb, lm, self.out_size)
                    aligned = True
                except ValueError:
                    crop = None
            if crop is None:
                # Haar box chỉ bao vùng mắt-miệng; nới nhẹ để gần với khung RAF-DB
                crop = square_crop(frame_rgb, (x, y, w, h), scale=1.1, out_size=self.out_size)

            detected.append(DetectedFace(
                face_id=idx + 1,
                bbox=(x, y, w, h),
                crop_rgb=crop,
                crop_pil=Image.fromarray(crop),
                score=score,
                landmarks=lm,
                aligned=aligned,
            ))
        return detected


if __name__ == "__main__":
    print("=" * 65)
    print("👀 KIỂM THỬ MODULE PHÁT HIỆN ĐA KHUÔN MẶT (MULTI-FACE DETECTOR)...")
    print("=" * 65)
    detector = MultiFaceDetector()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces(dummy_img, is_bgr=True)
    print(f"• Backend      : {detector.backend.upper()}")
    print(f"• Số mặt (ảnh đen): {len(faces)}")
    print("=" * 65)
