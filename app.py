"""
Module: app.py
Real-Time Multi-Face Explainable Facial Expression Recognition (XAI-FER)
────────────────────────────────────────────────────────────────────────
• Upload video → frame-by-frame emotion detection + annotated output
• Webcam snapshot → multi-face detection + Grad-CAM heatmap
• Crowd photo analysis
• Grad-CAM XAI inspector
• Head-to-head: Baseline vs Mask-Aware model comparison
"""

import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# ── UTF-8 for Windows ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from models import PretrainedFER, ImprovedCNN
from gradcam import GradCAM, overlay_heatmap_on_image
from face_detector import MultiFaceDetector

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS & CONFIG                                              ║
# ╚═══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="XAI-FER · Real-Time Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOJI_MAP = {
    "angry": "😡 Angry",
    "disgust": "🤢 Disgust",
    "fear": "😨 Fear",
    "happy": "😊 Happy",
    "neutral": "😐 Neutral",
    "sad": "😢 Sad",
    "surprise": "😲 Surprise",
}

EMOTION_COLORS = {
    "happy": (46, 204, 113),
    "surprise": (241, 196, 15),
    "neutral": (52, 152, 219),
    "sad": (155, 89, 182),
    "fear": (230, 126, 34),
    "angry": (231, 76, 60),
    "disgust": (149, 165, 166),
}

EMOTION_HEX = {
    "happy": "#2ecc71",
    "surprise": "#f1c40f",
    "neutral": "#3498db",
    "sad": "#9b59b6",
    "fear": "#e67e22",
    "angry": "#e74c3c",
    "disgust": "#95a5a6",
}

EXPLANATION_INSIGHTS = {
    "happy": "AI tập trung vào **khóe miệng cười** và **vết nhăn đuôi mắt** (Duchenne smile).",
    "surprise": "AI quét vào **mí mắt mở to** và **vòm chân mày giãn**.",
    "fear": "AI phản ứng với **cơ trán căng** và **ánh mắt mở rộng căng thẳng**.",
    "angry": "AI phân tích **lông mày nhíu lại** và **nếp nhăn trán**.",
    "disgust": "AI phản ứng với **cơ nhăn sống mũi co lại** hoặc **mí mắt nheo**.",
    "sad": "AI chú ý vào **đầu lông mày kéo xếch** và **mí mắt trên rũ xuống**.",
    "neutral": "Bản đồ nhiệt phân bổ đều — cơ biểu cảm thả lỏng tự nhiên.",
}

CHECKPOINT_MASK_AWARE = ROOT_DIR / "output" / "best_model_mask_aware.pth"
CHECKPOINT_RAFDB = ROOT_DIR / "output" / "best_model_rafdb.pth"
CHECKPOINT_FALLBACK = ROOT_DIR / "output" / "best_model.pth"
REAL_MASKED_DIR = ROOT_DIR / "data" / "real_masked_faces"
MULTI_SAMPLE_DIR = ROOT_DIR / "data" / "multi_face_samples"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS — Premium Dark Theme                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ── Hero header ── */
.hero-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1.05rem;
    margin-top: -4px;
    margin-bottom: 24px;
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 4px;
}

/* ── Emotion badge ── */
.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 2px 4px;
}

/* ── Section divider ── */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
    margin: 24px 0;
    border: none;
}

/* ── Smoother tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  DEVICE & MODEL HELPERS                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model(ckpt_path_str: str):
    device = get_device()
    model = PretrainedFER(backbone_name="mobilenet_v3_large", num_classes=7, pretrained=True).to(device)
    val_f1, is_trained = 0.0, False
    ckpt_path = Path(ckpt_path_str)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                val_f1 = ckpt.get("best_val_f1", 0.0)
                is_trained = True
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
    model.eval()
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    return model, grad_cam, device, is_trained, val_f1


@st.cache_resource
def get_detector():
    return MultiFaceDetector()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  INFERENCE HELPERS                                               ║
# ╚═══════════════════════════════════════════════════════════════════╝
def preprocess_face(pil_crop: Image.Image, device: torch.device):
    """Resize + ImageNet normalize → (rgb_np, tensor)."""
    rgb = pil_crop.convert("RGB").resize((224, 224))
    np_rgb = np.array(rgb, dtype=np.float32) / 255.0
    tensor = torch.tensor(np_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return np.array(rgb, dtype=np.uint8), tensor


def predict_with_gradcam(model, grad_cam, crop: Image.Image, device):
    rgb_np, tensor = preprocess_face(crop, device)
    heatmap, pred_idx, probs = grad_cam.generate_heatmap(tensor)
    label = CLASSES[pred_idx]
    conf = float(probs[pred_idx] * 100)
    return label, conf, probs, heatmap, rgb_np


def predict_fast(model, crop: Image.Image, device):
    """Fast inference — no Grad-CAM, pure speed."""
    rgb_np, tensor = preprocess_face(crop, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
    return CLASSES[pred_idx], float(probs[pred_idx] * 100), probs


def annotate_frame(frame_rgb, face_results):
    """Draw bounding boxes + labels on frame."""
    out = frame_rgb.copy()
    for r in face_results:
        x, y, w, h = r["bbox"]
        label, conf, fid = r["label"], r["conf"], r["face_id"]
        color = EMOTION_COLORS.get(label, (46, 204, 113))

        # Box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label background
        txt = f"#{fid} {label.upper()} {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(0, y - th - 8)
        cv2.rectangle(out, (x, label_y), (x + tw + 8, y), color, -1)
        cv2.putText(out, txt, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  UI COMPONENTS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝
def render_stat_card(value: str, label: str):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_emotion_bar(probs, pred_label):
    """Render a horizontal bar chart of emotion probabilities."""
    fig, ax = plt.subplots(figsize=(7, 2.5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    sorted_idx = np.argsort(probs)
    labels = [CLASSES[i].capitalize() for i in sorted_idx]
    vals = [probs[i] * 100 for i in sorted_idx]
    colors = [EMOTION_HEX.get(CLASSES[i], "#667eea") if CLASSES[i] == pred_label else "#334155" for i in sorted_idx]

    bars = ax.barh(labels, vals, color=colors, height=0.65, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.tick_params(colors="#888", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.15, color="#555")

    for bar in bars:
        w = bar.get_width()
        if w > 3:
            ax.text(w - 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                    ha="right", va="center", fontsize=7, fontweight="bold", color="white")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — REAL-TIME VIDEO & WEBCAM                               ║
# ╚═══════════════════════════════════════════════════════════════════╝
def tab_realtime(detector, model, grad_cam, device, alpha):
    st.markdown('<p class="hero-sub">🔴 Bật Webcam Mac nhận diện tức thì · 🎬 Tải video lên phân tích · 📸 Chụp ảnh khuôn mặt</p>', unsafe_allow_html=True)

    mode = st.segmented_control(
        "Chọn chế độ:",
        ["🔴 Live Webcam Mac (Real-Time)", "🎬 Upload Video", "📸 Chụp Ảnh Snapshot"],
        default="🔴 Live Webcam Mac (Real-Time)",
    )

    # ── LIVE WEBCAM MAC MODE ──
    if mode == "🔴 Live Webcam Mac (Real-Time)":
        _process_webcam_live(detector, model, grad_cam, device, alpha)

    # ── VIDEO UPLOAD MODE ──
    elif mode == "🎬 Upload Video":
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_up, col_cfg = st.columns([3, 2])
        with col_up:
            video_file = st.file_uploader(
                "Kéo thả hoặc chọn video",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="vid_upload",
                help="Hỗ trợ MP4, MOV, AVI, MKV, WebM"
            )
        with col_cfg:
            st.markdown("##### ⚙️ Cấu hình")
            skip = st.select_slider("Bước nhảy frame", options=[1, 2, 3, 5], value=2,
                                    format_func=lambda x: f"Mỗi {x} frame")
            save_vid = st.checkbox("📥 Xuất video đã gán nhãn", value=True)

        if video_file is None:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; border:2px dashed rgba(102,126,234,0.3); border-radius:16px; margin-top:12px;">
                <div style="font-size:3rem;">🎬</div>
                <div style="color:#888; margin-top:8px;">Tải lên video để bắt đầu phân tích cảm xúc thời gian thực</div>
                <div style="color:#555; font-size:0.85rem; margin-top:4px;">Hỗ trợ MP4 · MOV · AVI · MKV · WebM</div>
            </div>
            """, unsafe_allow_html=True)
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        video_path = tmp.name

        if st.button("▶️  Bắt Đầu Phân Tích", type="primary", use_container_width=True):
            _process_video(video_path, detector, model, grad_cam, device, alpha, skip, save_vid)

    # ── WEBCAM SNAPSHOT ──
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#888; margin-bottom:12px;">
            📸 Chụp ảnh nhanh qua trình duyệt → AI phân tích khuôn mặt + Grad-CAM
        </div>
        """, unsafe_allow_html=True)

        cam = st.camera_input("Bấm để chụp ảnh")
        if cam is not None:
            _process_snapshot(cam, detector, model, grad_cam, device, alpha)


def _process_webcam_live(detector, model, grad_cam, device, alpha):
    """Real-time live streaming directly from Mac's camera inside Streamlit."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12)); border: 1px solid rgba(102, 126, 234, 0.35); border-radius: 12px; padding: 14px 20px; margin-top: 10px; margin-bottom: 20px;">
        <span style="font-size: 1.05rem; font-weight: 700; color: #a5b4fc;">⚡ Muốn chạy mượt mà 30-60 FPS Native App (Không gián đoạn)?</span>
        <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
            Đã tích hợp sẵn ứng dụng macOS Desktop độc lập! Bạn chỉ cần mở Terminal gõ:
            <code style="background: #111827; color: #38bdf8; padding: 3px 8px; border-radius: 6px; font-weight: 600; margin-left: 6px;">uv run python webcam_app.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_toggle, c_cam, c_xai = st.columns([2, 1, 1])
    with c_toggle:
        run_cam = st.toggle("🔴 Bật Live Stream Webcam Mac", value=False, key="toggle_mac_webcam")
    with c_cam:
        cam_idx = st.number_input("Camera ID", min_value=0, max_value=3, value=0, step=1, help="0: FaceTime HD Camera mặc định của Mac")
    with c_xai:
        enable_xai = st.checkbox("Bật Grad-CAM XAI", value=True)

    if not run_cam:
        st.markdown("""
        <div style="text-align:center; padding:50px 20px; border:2px dashed rgba(102,126,234,0.3); border-radius:16px; margin-top:8px;">
            <div style="font-size:3.5rem;">📷</div>
            <div style="font-size: 1.15rem; font-weight: 600; color:#e2e8f0; margin-top:10px;">Camera Mac đang tắt</div>
            <div style="color:#94a3b8; margin-top:6px;">Gạt công tắc <b>"Bật Live Stream Webcam Mac"</b> ở trên để tải luồng video trực tiếp!</div>
        </div>
        """, unsafe_allow_html=True)
        return

    cap = cv2.VideoCapture(int(cam_idx))
    if not cap.isOpened():
        st.error(f"❌ Không thể truy cập Camera ID {cam_idx}. Vui lòng cấp quyền Camera cho Terminal/IDE và thử lại!")
        return

    col_vid, col_info = st.columns([3, 2])
    with col_vid:
        frame_ph = st.empty()
    with col_info:
        stats_ph = st.empty()
        chart_ph = st.empty()
        xai_ph = st.empty()

    t_start = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Không nhận được tín hiệu hình ảnh từ Webcam.")
                break

            # Lật gương để trực quan
            frame = cv2.flip(frame, 1)
            frame_count += 1
            now = time.time()
            if frame_count % 5 == 0:
                dt = now - t_start
                fps = frame_count / dt if dt > 0 else 0

            orig_h, orig_w = frame.shape[:2]
            det_w = 480
            det_scale = det_w / float(orig_w)
            det_h = int(orig_h * det_scale)
            small = cv2.resize(frame, (det_w, det_h))

            detected = detector.detect_faces(small, is_bgr=True)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            scale_x = orig_w / float(det_w)
            scale_y = orig_h / float(det_h)

            results = []
            primary_crop = None
            primary_probs = None
            primary_heatmap = None
            primary_label = ""
            primary_conf = 0.0

            for idx, face in enumerate(detected):
                sx, sy, sw, sh = face.bbox
                fx = max(0, int(sx * scale_x))
                fy = max(0, int(sy * scale_y))
                fw = min(orig_w - fx, int(sw * scale_x))
                fh = min(orig_h - fy, int(sh * scale_y))

                crop_bgr = frame[fy:fy+fh, fx:fx+fw]
                if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
                    continue

                pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

                if idx == 0 and enable_xai:
                    label, conf, probs, heatmap, rgb_np = predict_with_gradcam(model, grad_cam, pil_crop, device)
                    primary_crop = rgb_np
                    primary_heatmap = heatmap
                    primary_probs = probs
                    primary_label = label
                    primary_conf = conf
                else:
                    label, conf, probs = predict_fast(model, pil_crop, device)
                    if idx == 0:
                        primary_probs = probs
                        primary_label = label
                        primary_conf = conf

                results.append({
                    "face_id": idx + 1,
                    "label": label,
                    "conf": conf,
                    "bbox": (fx, fy, fw, fh)
                })

            annotated = annotate_frame(rgb_frame, results)
            frame_ph.image(annotated, use_container_width=True)

            with stats_ph.container():
                s1, s2 = st.columns(2)
                s1.metric("👥 Khuôn mặt", len(results))
                s2.metric("⚡ FPS", f"{fps:.1f}")

            if primary_probs is not None:
                with chart_ph.container():
                    st.markdown(f"**Cảm xúc chính:** {EMOJI_MAP.get(primary_label, primary_label)} ({primary_conf:.1f}%)")
                    render_emotion_bar(primary_probs, primary_label)

            if enable_xai and primary_heatmap is not None and primary_crop is not None:
                with xai_ph.container():
                    st.caption("🔬 **Bản đồ nhiệt Grad-CAM XAI**:")
                    ov = overlay_heatmap_on_image(primary_crop, primary_heatmap, alpha=alpha)
                    st.image(ov, width=170)
                    insight = EXPLANATION_INSIGHTS.get(primary_label, "")
                    if insight:
                        st.caption(f"💡 {insight}")

    finally:
        cap.release()


def _process_video(path, detector, model, grad_cam, device, alpha, skip, save_vid):
    """Process uploaded video frame-by-frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("❌ Không thể đọc video. Hãy kiểm tra định dạng file!")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Info cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat_card(f"{w}×{h}", "Độ phân giải")
    with c2:
        render_stat_card(f"{total}", "Tổng số frame")
    with c3:
        render_stat_card(f"{fps:.0f}", "FPS gốc")
    with c4:
        render_stat_card(f"{total/fps:.1f}s", "Thời lượng")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Setup output writer
    writer = None
    out_path = None
    if save_vid:
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = tmp_out.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, max(1.0, fps / skip), (w, h))

    # UI placeholders
    progress = st.progress(0)
    col_vid, col_info = st.columns([5, 3])
    with col_vid:
        frame_ph = st.empty()
    with col_info:
        stats_ph = st.empty()
        faces_ph = st.empty()

    # Processing loop
    idx = 0
    processed = 0
    t0 = time.time()
    emotion_log = {e: 0 for e in CLASSES}
    timeline = []

    try:
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            idx += 1
            if idx % skip != 0:
                continue

            processed += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb, is_bgr=False)
            results = []

            for face in faces:
                label, conf, probs = predict_fast(model, face.crop_pil, device)
                results.append({"face_id": face.face_id, "label": label, "conf": conf, "bbox": face.bbox})
                emotion_log[label] += 1
                timeline.append({"time": round(idx / fps, 2), "face": face.face_id, "emotion": label, "conf": conf})

            annotated = annotate_frame(rgb, results)
            if writer:
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            # Update UI
            elapsed = time.time() - t0
            run_fps = processed / elapsed if elapsed > 0 else 0
            pct = min(1.0, idx / max(1, total))
            progress.progress(pct, text=f"Frame {idx}/{total}  ·  {run_fps:.1f} FPS  ·  {pct*100:.0f}%")

            frame_ph.image(annotated, use_container_width=True)

            with stats_ph.container():
                s1, s2 = st.columns(2)
                s1.metric("👥 Faces", len(results))
                s2.metric("⚡ FPS", f"{run_fps:.1f}")

            with faces_ph.container():
                if results:
                    for r in results:
                        hex_c = EMOTION_HEX.get(r["label"], "#667eea")
                        st.markdown(
                            f'<span class="emotion-badge" style="background:{hex_c}20; color:{hex_c}; border:1px solid {hex_c}50;">'
                            f'#{r["face_id"]} {EMOJI_MAP[r["label"]]} {r["conf"]:.0f}%</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Chưa phát hiện khuôn mặt...")
    finally:
        cap.release()
        if writer:
            writer.release()

    # ── Results ──
    progress.progress(1.0, text=f"✅ Hoàn tất trong {time.time() - t0:.1f}s!")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Kết Quả Phân Tích")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("#### Tỷ lệ cảm xúc xuất hiện")
        total_count = sum(emotion_log.values())
        if total_count > 0:
            for em, cnt in sorted(emotion_log.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    pct_em = cnt / total_count * 100
                    hex_c = EMOTION_HEX.get(em, "#667eea")
                    st.markdown(
                        f'<div style="display:flex; align-items:center; margin:4px 0;">'
                        f'<div style="width:120px; font-size:0.9rem;">{EMOJI_MAP[em]}</div>'
                        f'<div style="flex:1; background:#1a1a2e; border-radius:8px; height:24px; overflow:hidden;">'
                        f'<div style="width:{pct_em}%; background:{hex_c}; height:100%; border-radius:8px; '
                        f'display:flex; align-items:center; justify-content:flex-end; padding-right:8px; '
                        f'font-size:0.75rem; font-weight:600; color:white;">{cnt} ({pct_em:.0f}%)</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Không phát hiện khuôn mặt nào trong video.")

    with rc2:
        if timeline:
            df = pd.DataFrame(timeline)
            st.markdown("#### Phân bố cảm xúc")
            st.bar_chart(df["emotion"].value_counts(), color="#667eea")

    if out_path and Path(out_path).exists():
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with open(out_path, "rb") as vf:
            st.download_button(
                "📥 Tải xuống Video đã gán nhãn (.mp4)",
                data=vf.read(),
                file_name="fer_annotated.mp4",
                mime="video/mp4",
                use_container_width=True,
            )


def _process_snapshot(cam_data, detector, model, grad_cam, device, alpha):
    """Process a single webcam snapshot with full Grad-CAM."""
    raw = Image.open(cam_data)
    frame = np.array(raw)
    faces = detector.detect_faces(frame, is_bgr=False)

    if not faces:
        st.warning("⚠️ Không phát hiện khuôn mặt nào. Hãy đối diện camera và thử lại!")
        st.image(raw, use_container_width=True)
        return

    st.success(f"🎉 Phát hiện **{len(faces)}** khuôn mặt!")

    # Predict all faces
    results = []
    for face in faces:
        label, conf, probs, hm, rgb_np = predict_with_gradcam(model, grad_cam, face.crop_pil, device)
        results.append({
            "face_id": face.face_id, "label": label, "conf": conf,
            "probs": probs, "heatmap": hm, "crop_rgb": rgb_np, "bbox": face.bbox,
        })

    # Annotated full image
    annotated = annotate_frame(frame, results)
    col_img, col_list = st.columns([5, 3])
    with col_img:
        st.image(annotated, use_container_width=True)
    with col_list:
        for r in results:
            hex_c = EMOTION_HEX.get(r["label"], "#667eea")
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:12px; padding:8px 12px; '
                f'background:linear-gradient(135deg, {hex_c}15, {hex_c}05); border:1px solid {hex_c}40; '
                f'border-radius:12px; margin-bottom:8px;">'
                f'<img src="" width="0" height="0"/>'
                f'<div><strong>Người #{r["face_id"]}</strong><br/>'
                f'<span style="color:{hex_c}; font-weight:600;">{EMOJI_MAP[r["label"]]}</span> '
                f'<span style="color:#888;">({r["conf"]:.1f}%)</span></div></div>',
                unsafe_allow_html=True,
            )
            st.image(r["crop_rgb"], width=80)

    # Grad-CAM inspector
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔬 Grad-CAM Inspector")

    sel = st.selectbox(
        "Chọn khuôn mặt để soi bản đồ nhiệt:",
        [f"Người #{r['face_id']} — {EMOJI_MAP[r['label']]}" for r in results],
    )
    sel_idx = int(sel.split("#")[1].split(" ")[0]) - 1
    chosen = results[sel_idx]

    c1, c2, c3 = st.columns(3)
    _, overlay = overlay_heatmap_on_image(chosen["crop_rgb"], chosen["heatmap"], alpha=alpha)
    hm_colored, _ = overlay_heatmap_on_image(chosen["crop_rgb"], chosen["heatmap"], alpha=1.0)
    with c1:
        st.image(chosen["crop_rgb"], caption="Ảnh gốc", use_container_width=True)
    with c2:
        st.image(hm_colored, caption="Grad-CAM Heatmap", use_container_width=True)
    with c3:
        st.image(overlay, caption=f"Overlay (α={alpha})", use_container_width=True)

    st.info(f"💡 {EXPLANATION_INSIGHTS.get(chosen['label'], '')}")
    render_emotion_bar(chosen["probs"], chosen["label"])


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — CROWD / GROUP PHOTO                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝
def tab_crowd(detector, model, grad_cam, device):
    st.markdown("#### 👥 Phân Tích Cảm Xúc Nhóm / Đám Đông")
    upload = st.file_uploader("Tải ảnh chụp nhóm:", type=["jpg", "jpeg", "png"], key="group_up")

    img = None
    if upload:
        img = Image.open(upload)
    else:
        sample = MULTI_SAMPLE_DIR / "group_people_1.jpg"
        if sample.exists() and st.checkbox("Dùng ảnh mẫu", value=True):
            img = Image.open(sample)

    if img is None:
        return

    frame = np.array(img.convert("RGB"))
    faces = detector.detect_faces(frame, is_bgr=False)

    if not faces:
        st.warning("Không tìm thấy khuôn mặt nào!")
        st.image(img, use_container_width=True)
        return

    st.success(f"Phát hiện **{len(faces)} người**!")
    annotated = frame.copy()
    counts = {}

    for f in faces:
        label, conf, _, _, _ = predict_with_gradcam(model, grad_cam, f.crop_pil, device)
        counts[label] = counts.get(label, 0) + 1
        x, y, w, h = f.bbox
        c = EMOTION_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), c, 3)
        cv2.putText(annotated, f"#{f.face_id} {label.capitalize()}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(annotated, use_container_width=True)
    with c2:
        st.markdown("#### 📊 Thống kê:")
        for em, cnt in counts.items():
            st.metric(EMOJI_MAP[em], f"{cnt} ({cnt / len(faces) * 100:.0f}%)")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — GRAD-CAM XAI INSPECTOR                                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
def tab_xai(model, grad_cam, device, alpha):
    st.markdown("#### 🔍 Kính Lúp XAI — Soi Bản Đồ Nhiệt Grad-CAM")
    source = st.radio("Nguồn ảnh:", ["Upload ảnh khuôn mặt", "Ảnh người đeo khẩu trang (RMFD)"], horizontal=True)

    face_img = None
    if source == "Upload ảnh khuôn mặt":
        up = st.file_uploader("Tải ảnh:", type=["jpg", "png", "jpeg"], key="xai_up")
        if up:
            face_img = Image.open(up)
    else:
        files = sorted(list(REAL_MASKED_DIR.glob("*.jpg")))
        if files:
            sel = st.selectbox(f"Chọn mẫu ({len(files)} ảnh):", [f.name for f in files])
            face_img = Image.open(REAL_MASKED_DIR / sel)
        else:
            st.info("Chưa có ảnh trong `data/real_masked_faces`.")

    if face_img is None:
        return

    label, conf, probs, hm, rgb_np = predict_with_gradcam(model, grad_cam, face_img, device)
    _, overlay = overlay_heatmap_on_image(rgb_np, hm, alpha=alpha)
    hm_colored, _ = overlay_heatmap_on_image(rgb_np, hm, alpha=1.0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        st.metric("Cảm xúc", EMOJI_MAP[label], delta=f"{conf:.1f}%")
    with mc2:
        st.info(f"💡 {EXPLANATION_INSIGHTS.get(label, '')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rgb_np, caption="Ảnh gốc", use_container_width=True)
    with c2:
        st.image(hm_colored, caption="Grad-CAM", use_container_width=True)
    with c3:
        st.image(overlay, caption=f"Overlay (α={alpha})", use_container_width=True)

    render_emotion_bar(probs, label)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — HEAD-TO-HEAD COMPARISON                                ║
# ╚═══════════════════════════════════════════════════════════════════╝
def tab_compare(device, alpha):
    st.markdown("#### ⚔️ So Sánh: Baseline vs Mask-Aware")
    st.caption("Đặt hai mô hình lên bàn cân: khi khuôn mặt bị che khẩu trang, mô hình Mask-Aware chuyển dịch vùng chú ý lên mắt.")

    has_both = CHECKPOINT_RAFDB.exists() and (CHECKPOINT_MASK_AWARE.exists() or CHECKPOINT_FALLBACK.exists())
    if not has_both:
        st.warning("Cần cả 2 checkpoint `best_model_rafdb.pth` và `best_model_mask_aware.pth`.")
        return

    base_model, base_cam, _, _, _ = load_model(str(CHECKPOINT_RAFDB))
    mask_path = str(CHECKPOINT_MASK_AWARE if CHECKPOINT_MASK_AWARE.exists() else CHECKPOINT_FALLBACK)
    mask_model, mask_cam, _, _, _ = load_model(mask_path)

    src = st.radio("Ảnh thử nghiệm:", ["Mẫu có sẵn", "Upload"], horizontal=True, key="cmp_r")
    cmp_img = None
    if src == "Mẫu có sẵn":
        files = sorted(list(REAL_MASKED_DIR.glob("*.jpg")))
        if files:
            sel = st.selectbox("Chọn:", [f.name for f in files], key="cmp_sel")
            cmp_img = Image.open(REAL_MASKED_DIR / sel)
    else:
        up = st.file_uploader("Upload:", type=["jpg", "png"], key="cmp_up")
        if up:
            cmp_img = Image.open(up)

    if cmp_img is None:
        return

    bl, bc, bp, bh, br = predict_with_gradcam(base_model, base_cam, cmp_img, device)
    ml, mc, mp, mh, mr = predict_with_gradcam(mask_model, mask_cam, cmp_img, device)
    _, b_ov = overlay_heatmap_on_image(br, bh, alpha=alpha)
    _, m_ov = overlay_heatmap_on_image(mr, mh, alpha=alpha)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    cb, cm = st.columns(2)
    with cb:
        st.markdown("##### 👤 Baseline (mặt thường)")
        st.metric("Dự đoán", EMOJI_MAP[bl], delta=f"{bc:.1f}%")
        st.image(b_ov, use_container_width=True)
        st.caption("⚠️ Thường bị 'mù' khi vùng miệng bị che khuất.")
    with cm:
        st.markdown("##### 🛡️ Mask-Aware")
        st.metric("Dự đoán", EMOJI_MAP[ml], delta=f"{mc:.1f}%")
        st.image(m_ov, use_container_width=True)
        st.caption("✅ Chuyển dịch chú ý lên vùng mắt và lông mày.")

    st.success(f"🏆 Baseline → `{bl.upper()}` | Mask-Aware → `{ml.upper()}`")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝
def main():
    detector = get_detector()
    device = get_device()

    # ── Available models ──
    models = {}
    if CHECKPOINT_MASK_AWARE.exists():
        models["🛡️ Mask-Aware FER"] = str(CHECKPOINT_MASK_AWARE)
    if CHECKPOINT_RAFDB.exists():
        models["👤 Baseline RAF-DB"] = str(CHECKPOINT_RAFDB)
    if not models:
        models["⚡ MobileNetV3"] = str(CHECKPOINT_FALLBACK)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🎭 XAI-FER")
        st.caption(f"Device: `{device.type.upper()}`{'  ·  GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

        sel_name = st.selectbox("Model:", list(models.keys()))
        model, grad_cam, _, trained, f1 = load_model(models[sel_name])

        if trained:
            st.success(f"✅ Loaded · F1: {f1:.4f}")
        else:
            st.info("⚡ Ready")

        st.markdown("---")
        alpha = st.slider("Grad-CAM α", 0.1, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.markdown(
            "<div style='color:#555; font-size:0.75rem;'>"
            "MobileNetV3-Large · RAF-DB<br/>"
            "MediaPipe + Haar Cascade<br/>"
            "Grad-CAM XAI"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Header ──
    st.markdown('<h1 class="hero-title">🎭 Real-Time Facial Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">MobileNetV3-Large · Mask-Aware · Explainable AI (Grad-CAM)</p>', unsafe_allow_html=True)

    # ── Tabs ──
    t1, t2, t3, t4 = st.tabs([
        "🎥  Real-Time Detection",
        "👥  Crowd Analysis",
        "🔍  XAI Inspector",
        "⚔️  Model Comparison",
    ])

    with t1:
        tab_realtime(detector, model, grad_cam, device, alpha)
    with t2:
        tab_crowd(detector, model, grad_cam, device)
    with t3:
        tab_xai(model, grad_cam, device, alpha)
    with t4:
        tab_compare(device, alpha)


if __name__ == "__main__":
    main()
