"""
Script: scripts/benchmark_fps.py
Đo tốc độ pipeline real-time (giống webcam_app.py): detect + căn chỉnh -> dự đoán batch (TTA) -> tracker
-> Grad-CAM cho khuôn mặt lớn nhất. Tách thời gian từng bước, đo theo số khuôn mặt trong khung hình.

Hai chế độ:
- synthetic (mặc định): ghép k ảnh khuôn mặt (k = 1..5) vào frame 1280x720 -> đo có kiểm soát số mặt.
- --camera ID: đọc frame thật từ webcam (gồm cả thời gian chụp ảnh của camera).

Chạy:  python scripts/benchmark_fps.py --faces-dir data/raf_db/test
       python scripts/benchmark_fps.py --camera 0 --frames 200
"""

import sys
import time
import json
import random
import argparse
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from face_detector import MultiFaceDetector  # noqa: E402
from inference import FERPredictor, FaceTracker, get_device  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_frame(faces: list[np.ndarray], k: int, rng: random.Random) -> np.ndarray:
    frame = np.full((720, 1280, 3), 90, np.uint8)
    size = 220
    slots = [(40 + i * 245, 250) for i in range(5)]
    for (x, y), f in zip(slots[:k], rng.sample(faces, k)):
        frame[y:y + size, x:x + size] = cv2.resize(f, (size, size))
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def run_pipeline(frame, detector, predictor, tracker, gradcam: bool, device, t, tta: bool = True,
                 cam_every: int = 1, frame_idx: int = 0):
    t0 = time.perf_counter()
    faces = detector.detect_faces(frame, is_bgr=True)
    t1 = time.perf_counter()
    if faces:
        probs = predictor.predict([f.crop_rgb for f in faces], tta=tta)
        sync(device)
        t2 = time.perf_counter()
        tracker.update([f.bbox for f in faces], probs)
        if gradcam and frame_idx % cam_every == 0:
            predictor.explain(faces[0].crop_rgb, tta=tta)
            sync(device)
    else:
        t2 = time.perf_counter()
    t3 = time.perf_counter()
    t["detect"].append(t1 - t0)
    t["predict"].append(t2 - t1)
    t["cam"].append(t3 - t2)
    t["total"].append(t3 - t0)
    return len(faces)


def summarize(t: dict, warmup: int) -> dict:
    ms = {k: 1000 * float(np.mean(v[warmup:])) for k, v in t.items()}
    ms["fps"] = 1000.0 / ms["total"]
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT_DIR / "output" / "best_model_mask_aware.pth"))
    ap.add_argument("--faces-dir", default=str(ROOT_DIR / "data" / "raf_db" / "test"))
    ap.add_argument("--camera", type=int, default=None, help="Đo trên webcam thật thay vì frame tổng hợp")
    ap.add_argument("--frames", type=int, default=60, help="Số frame đo cho mỗi cấu hình")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-faces", type=int, default=5)
    ap.add_argument("--no-tta", action="store_true", help="Tắt TTA lật ngang (nhanh gấp ~2 lần ở bước dự đoán)")
    ap.add_argument("--cam-every", type=int, default=1, help="Chỉ tính Grad-CAM mỗi N frame (app dùng lại heatmap cũ)")
    ap.add_argument("--json", default="", help="Ghi kết quả ra file JSON")
    args = ap.parse_args()

    device = get_device()
    detector = MultiFaceDetector()
    predictor = FERPredictor(args.checkpoint, device)
    hw = {"device": device.type, "torch": torch.__version__, "platform": platform.platform(),
          "cpu": platform.processor() or platform.machine()}
    if device.type == "cuda":
        hw["gpu"] = torch.cuda.get_device_name(0)
    print(f"Thiết bị: {hw} | detector={detector.backend} | model={predictor.model_name} ({predictor.input_size}px)")

    results = []
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise SystemExit(f"❌ Không mở được camera {args.camera} (kiểm tra quyền Camera cho Terminal).")
        for gradcam in (False, True):
            t = {k: [] for k in ("detect", "predict", "cam", "total")}
            cap_times, n_faces = [], []
            tracker = FaceTracker()
            for i in range(args.frames + args.warmup):
                c0 = time.perf_counter()
                ok, frame = cap.read()
                cap_times.append(time.perf_counter() - c0)
                if not ok:
                    continue
                n_faces.append(run_pipeline(cv2.flip(frame, 1), detector, predictor, tracker, gradcam, device, t,
                                            not args.no_tta, args.cam_every, i))
            ms = summarize(t, args.warmup)
            ms["capture"] = 1000 * float(np.mean(cap_times[args.warmup:]))
            ms["fps_end_to_end"] = 1000.0 / (ms["total"] + ms["capture"])
            ms.update({"mode": "camera", "gradcam": gradcam, "faces_mean": float(np.mean(n_faces[args.warmup:])),
                       "frame": f"{frame.shape[1]}x{frame.shape[0]}"})
            results.append(ms)
        cap.release()
    else:
        files = sorted(p for p in Path(args.faces_dir).rglob("*.jpg"))[:500]
        if len(files) < args.max_faces:
            raise SystemExit(f"❌ Cần ảnh khuôn mặt trong {args.faces_dir}")
        faces = [np.array(Image.open(p).convert("RGB")) for p in files]
        rng = random.Random(0)
        for gradcam in (False, True):
            for k in range(1, args.max_faces + 1):
                t = {key: [] for key in ("detect", "predict", "cam", "total")}
                tracker = FaceTracker()
                detected = []
                for i in range(args.frames + args.warmup):
                    detected.append(run_pipeline(make_frame(faces, k, rng), detector, predictor, tracker, gradcam, device,
                                                 t, not args.no_tta, args.cam_every, i))
                ms = summarize(t, args.warmup)
                ms.update({"mode": "synthetic", "gradcam": gradcam, "faces": k,
                           "detected_mean": float(np.mean(detected[args.warmup:]))})
                results.append(ms)

    print(f"\nTTA={'tắt' if args.no_tta else 'bật'} | Grad-CAM mỗi {args.cam_every} frame")
    print("| Chế độ | Grad-CAM | Số mặt | Detect (ms) | Dự đoán (ms) | Grad-CAM (ms) | Tổng (ms) | FPS |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        faces = r.get("faces", f"{r.get('faces_mean', 0):.1f}")
        fps = f"{r['fps']:.1f}" + (f" ({r['fps_end_to_end']:.1f} gồm camera)" if "fps_end_to_end" in r else "")
        print(f"| {r['mode']} | {'bật' if r['gradcam'] else 'tắt'} | {faces} | {r['detect']:.1f} | {r['predict']:.1f} | "
              f"{r['cam']:.1f} | {r['total']:.1f} | {fps} |")

    if args.json:
        Path(args.json).write_text(json.dumps({"hardware": hw, "checkpoint": Path(args.checkpoint).name,
                                               "tta": not args.no_tta, "cam_every": args.cam_every,
                                               "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"💾 Đã ghi {args.json}")


if __name__ == "__main__":
    main()
