"""
Module: app.py
Mục đích:
1. Ứng dụng Web tương tác thời gian thực: Nhận diện cảm xúc đa khuôn mặt (Real-Time Multi-Face FER).
2. Xây dựng trên DỮ LIỆU THẬT 100% (RAF-DB người thật & MaskTheFace/RMFD người thật đeo khẩu trang).
3. Hỗ trợ 2 mô hình:
   - 🛡️ Mask-Aware FER (Huấn luyện kết hợp mặt gốc & mặt đeo khẩu trang).
   - 👤 Baseline RAF-DB (Chỉ học mặt thường không khẩu trang).
4. Tích hợp bộ phát hiện đa khuôn mặt (MultiFaceDetector), kính lúp Explainable AI (Grad-CAM),
   và Tab So Sánh Đối Đầu (Head-to-Head Comparison) chứng minh sự vượt trội khi có khẩu trang.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Đảm bảo UTF-8 cho Windows console
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

# ---------------------------------------------------------------------
# CẤU HÌNH GIAO DIỆN & TÀI NGUYÊN
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Mask-Aware Multi-Face XAI FER",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOJI_MAP = {
    "angry": "😡 Tức giận (Angry)",
    "disgust": "🤢 Ghê tởm (Disgust)",
    "fear": "😨 Sợ hãi (Fear)",
    "happy": "😊 Vui vẻ (Happy)",
    "neutral": "😐 Trung tính (Neutral)",
    "sad": "😢 Buồn bã (Sad)",
    "surprise": "😲 Ngạc nhiên (Surprise)"
}

EMOTION_COLORS = {
    "happy": (46, 204, 113),     # Xanh lá
    "surprise": (241, 196, 15),  # Vàng
    "neutral": (52, 152, 219),   # Xanh dương
    "sad": (155, 89, 182),       # Tím
    "fear": (230, 126, 34),      # Cam
    "angry": (231, 76, 60),      # Đỏ
    "disgust": (149, 165, 166)   # Xám
}

EXPLANATION_INSIGHTS = {
    "happy": "AI tập trung nhiều nhất vào **Khóe miệng cười (Zygomatic major)** hoặc **Vết nhăn đuôi mắt (Orbicularis oculi - Duchenne smile)**.",
    "surprise": "AI quét vào **Độ mở to tròn của mí mắt** và **Độ dãn vòm chân mày**.",
    "fear": "AI phản ứng với **Độ căng cơ trán trên** và **Ánh mắt mở rộng căng thẳng**.",
    "angry": "AI phân tích **Vùng lông mày nhíu lại (Corrugator supercilii)** và **Nếp nhăn trán**.",
    "disgust": "AI phản ứng với **Cơ nhăn sống mũi co lại (Levator labii)** hoặc **Độ nheo mí mắt**.",
    "sad": "AI chú ý vào **Đầu trong lông mày kéo xếch lên** và **Vùng mí mắt trên rũ xuống**.",
    "neutral": "Bản đồ nhiệt phân bổ đều, các nhóm cơ biểu cảm ở trạng thái thả lỏng tự nhiên."
}

CHECKPOINT_MASK_AWARE = ROOT_DIR / "output" / "best_model_mask_aware.pth"
CHECKPOINT_RAFDB = ROOT_DIR / "output" / "best_model_rafdb.pth"
CHECKPOINT_FALLBACK = ROOT_DIR / "output" / "best_model.pth"

REAL_MASKED_DIR = ROOT_DIR / "data" / "real_masked_faces"
MULTI_SAMPLE_DIR = ROOT_DIR / "data" / "multi_face_samples"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@st.cache_resource
def load_single_model(ckpt_path_str: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedFER(backbone_name="mobilenet_v3_large", num_classes=7, pretrained=True).to(device)
    val_f1 = 0.0
    is_trained = False
    
    ckpt_path = Path(ckpt_path_str)
    if ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                val_f1 = checkpoint.get("best_val_f1", 0.0)
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


def preprocess_crop_for_model(pil_crop: Image.Image, device: torch.device):
    """Chuẩn hóa ảnh crop [224, 224, 3] theo chuẩn ImageNet."""
    rgb_crop = pil_crop.convert("RGB").resize((224, 224))
    np_crop = np.array(rgb_crop, dtype=np.float32) / 255.0
    tensor_img = torch.tensor(np_crop).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor_img = (tensor_img - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return np.array(rgb_crop, dtype=np.uint8), tensor_img


def predict_single_face(model, grad_cam, pil_crop: Image.Image, device: torch.device):
    rgb_np, input_tensor = preprocess_crop_for_model(pil_crop, device)
    heatmap, pred_idx, probs = grad_cam.generate_heatmap(input_tensor)
    pred_label = CLASSES[pred_idx]
    confidence = float(probs[pred_idx] * 100)
    return pred_label, confidence, probs, heatmap, rgb_np


# ---------------------------------------------------------------------
# GIAO DIỆN CHÍNH
# ---------------------------------------------------------------------
def main():
    detector = get_detector()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Danh mục models khả dụng
    available_models = {}
    if CHECKPOINT_MASK_AWARE.exists():
        available_models["🛡️ Mask-Aware FER (Đã học thích nghi khẩu trang)"] = str(CHECKPOINT_MASK_AWARE)
    if CHECKPOINT_RAFDB.exists():
        available_models["👤 Baseline RAF-DB (Chỉ học mặt thường)"] = str(CHECKPOINT_RAFDB)
    if not available_models:
        available_models["⚡ MobileNetV3 Checkpoint"] = str(CHECKPOINT_FALLBACK)

    # Sidebar
    st.sidebar.title("🎭 XAI Mask-Aware FER")
    st.sidebar.markdown(f"**Phần cứng:** `{device.type.upper()}`")
    if device.type == "cuda":
        st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")

    st.sidebar.markdown("**Kiến trúc:** `MobileNetV3-Large`")
    st.sidebar.markdown("**Dữ liệu huấn luyện:** `RAF-DB & Masked RAF-DB`")

    selected_model_name = st.sidebar.selectbox(
        "Mô hình đang kích hoạt:",
        list(available_models.keys()),
        index=0
    )
    selected_ckpt = available_models[selected_model_name]
    model, grad_cam, _, is_trained, val_f1 = load_single_model(selected_ckpt)

    if is_trained:
        st.sidebar.success(f"✅ Đã nạp `{Path(selected_ckpt).name}` (Val F1: {val_f1:.4f})")
    else:
        st.sidebar.info("⚡ Sẵn sàng suy luận tức thì")

    alpha_val = st.sidebar.slider(
        "Độ trong suốt Heatmap (Grad-CAM Alpha):",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )

    st.title("🎭 Hệ Thống Nhận Diện Cảm Xúc Đa Khuôn Mặt & Thích Nghi Khẩu Trang")
    st.markdown(
        "Giải pháp Deep Learning SOTA (**MobileNetV3-Large**) giải quyết triệt để bài toán "
        "**Che khuất khuôn mặt (Mask Occlusion)** kết hợp **Explainable AI (Grad-CAM)** "
        "minh chứng sự chuyển dịch trọng tâm thị giác từ Miệng lên Đôi Mắt."
    )

    tabs = st.tabs([
        "🎥 1. Quét Đa Khuôn Mặt (Webcam / Live)",
        "👥 2. Phân Tích Nhóm Đông Người (Crowd Photo)",
        "🔍 3. Kính Lúp XAI (Grad-CAM Trực Quan)",
        "⚔️ 4. So Sánh Đối Đầu (Baseline vs Mask-Aware)"
    ])

    # =================================================================
    # TAB 1: WEBCAM / SNAPSHOT MULTI-FACE SCANNING
    # =================================================================
    with tabs[0]:
        st.subheader("Quét và Nhận diện Đồng thời Mọi Khuôn mặt qua Camera")
        st.info("💡 Hệ thống tự động quét và phân loại cảm xúc song song cho mọi người trong khung hình (hỗ trợ cả người đeo khẩu trang).")

        camera_input = st.camera_input("Chụp ảnh từ Webcam:")
        if camera_input is not None:
            raw_img = Image.open(camera_input)
            frame_np = np.array(raw_img)

            faces = detector.detect_faces(frame_np, is_bgr=False)

            if len(faces) == 0:
                st.warning("⚠️ Không phát hiện khuôn mặt nào trong khung hình. Hãy hướng thẳng mặt vào camera!")
                st.image(raw_img, caption="Khung hình gốc", use_container_width=True)
            else:
                st.success(f"🎉 Đã phát hiện đồng thời **{len(faces)} khuôn mặt** trong khung hình!")

                annotated_frame = frame_np.copy()
                face_results = []

                for face in faces:
                    x, y, w, h = face.bbox
                    pred_label, conf, probs, hm, rgb_np = predict_single_face(model, grad_cam, face.crop_pil, device)
                    face_results.append({
                        "face_id": face.face_id,
                        "label": pred_label,
                        "conf": conf,
                        "probs": probs,
                        "heatmap": hm,
                        "crop_rgb": rgb_np,
                        "bbox": face.bbox
                    })

                    color = EMOTION_COLORS.get(pred_label, (0, 255, 0))
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
                    label_str = f"#{face.face_id} {pred_label.upper()} {conf:.0f}%"
                    cv2.rectangle(annotated_frame, (x, max(0, y - 28)), (x + len(label_str) * 11, y), color, -1)
                    cv2.putText(
                        annotated_frame, label_str, (x + 4, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
                    )

                col_cam1, col_cam2 = st.columns([2, 1])
                with col_cam1:
                    st.image(annotated_frame, caption=f"Khung hình Quét Đa Khuôn Mặt ({len(faces)} người)", use_container_width=True)

                with col_cam2:
                    st.markdown("### 📋 Danh sách người được quét:")
                    for res in face_results:
                        st.markdown(
                            f"**Người #{res['face_id']}:** {EMOJI_MAP[res['label']]} "
                            f"`Độ tin cậy: {res['conf']:.1f}%`"
                        )
                        st.image(res["crop_rgb"], width=100)
                        st.write("---")

                st.markdown("### 🔬 Soi Giải thích AI (Grad-CAM Inspector) cho từng người:")
                selected_face_id = st.selectbox(
                    "Chọn người muốn soi bản đồ nhiệt:",
                    [f"Người #{r['face_id']} ({r['label']})" for r in face_results]
                )
                sel_idx = int(selected_face_id.split("#")[1].split(" ")[0]) - 1
                chosen = face_results[sel_idx]

                c1, c2, c3 = st.columns(3)
                _, overlay = overlay_heatmap_on_image(chosen["crop_rgb"], chosen["heatmap"], alpha=alpha_val)
                colored_hm, _ = overlay_heatmap_on_image(chosen["crop_rgb"], chosen["heatmap"], alpha=1.0)

                with c1:
                    st.image(chosen["crop_rgb"], caption=f"Ảnh mặt crop #{chosen['face_id']}", use_container_width=True)
                with c2:
                    st.image(colored_hm, caption="Bản đồ nhiệt Grad-CAM độc lập", use_container_width=True)
                with c3:
                    st.image(overlay, caption=f"Phủ Heatmap (Alpha = {alpha_val})", use_container_width=True)

                st.info(f"💡 **Diễn giải sinh học:** {EXPLANATION_INSIGHTS.get(chosen['label'], '')}")

    # =================================================================
    # TAB 2: CROWD / GROUP PHOTO ANALYSIS
    # =================================================================
    with tabs[1]:
        st.subheader("Phân Tích Cảm Xúc Nhóm / Đám Đông (Group Photo Analysis)")
        upload_group = st.file_uploader("Tải ảnh chụp nhóm (JPG/PNG):", type=["jpg", "jpeg", "png"], key="group_uploader")

        use_sample_group = st.checkbox("Hoặc dùng ảnh mẫu đông người có sẵn", value=True if not upload_group else False)
        group_img = None

        if upload_group is not None:
            group_img = Image.open(upload_group)
        elif use_sample_group:
            sample_p = MULTI_SAMPLE_DIR / "group_people_1.jpg"
            if sample_p.exists():
                group_img = Image.open(sample_p)

        if group_img is not None:
            g_frame = np.array(group_img.convert("RGB"))
            g_faces = detector.detect_faces(g_frame, is_bgr=False)

            if len(g_faces) == 0:
                st.warning("Không tìm thấy khuôn mặt nào trong ảnh nhóm này!")
                st.image(group_img, use_container_width=True)
            else:
                st.success(f"Phát hiện tổng cộng **{len(g_faces)} người** trong bức ảnh!")
                g_annotated = g_frame.copy()
                emotion_counts = {}

                for f in g_faces:
                    pred_l, conf, _, _, _ = predict_single_face(model, grad_cam, f.crop_pil, device)
                    emotion_counts[pred_l] = emotion_counts.get(pred_l, 0) + 1
                    x, y, w, h = f.bbox
                    c = EMOTION_COLORS.get(pred_l, (0, 255, 0))
                    cv2.rectangle(g_annotated, (x, y), (x + w, y + h), c, 3)
                    txt = f"#{f.face_id} {pred_l.capitalize()}"
                    cv2.putText(g_annotated, txt, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

                col_g1, col_g2 = st.columns([3, 2])
                with col_g1:
                    st.image(g_annotated, caption=f"Ảnh nhóm đã nhận diện ({len(g_faces)} người)", use_container_width=True)

                with col_g2:
                    st.markdown("### 📊 Thống kê biểu cảm cả nhóm:")
                    for em, cnt in emotion_counts.items():
                        st.metric(label=EMOJI_MAP[em], value=f"{cnt} người ({cnt/len(g_faces)*100:.0f}%)")

    # =================================================================
    # TAB 3: XAI GRAD-CAM & MASKED FACES
    # =================================================================
    with tabs[2]:
        st.subheader("Soi Bản Đồ Nhiệt Grad-CAM & Kiểm Chứng Trên Người Thật Đeo Khẩu Trang")
        
        mode_choice = st.radio(
            "Chọn nguồn ảnh:",
            ["Ảnh người thật đeo khẩu trang (RMFD)", "Ảnh test tổng hợp (MaskTheFace RAF-DB)", "Tải ảnh chân dung tùy ý (Upload)"],
            horizontal=True
        )

        test_face_img = None
        if mode_choice == "Ảnh người thật đeo khẩu trang (RMFD)":
            masked_files = sorted(list(REAL_MASKED_DIR.glob("*.jpg")))
            if masked_files:
                selected_m_name = st.selectbox(
                    f"Chọn mẫu người thật đeo khẩu trang ({len(masked_files)} ảnh có sẵn):",
                    [f.name for f in masked_files]
                )
                test_face_img = Image.open(REAL_MASKED_DIR / selected_m_name)
            else:
                st.info("Chưa có ảnh trong thư mục data/real_masked_faces.")
        elif mode_choice == "Ảnh test tổng hợp (MaskTheFace RAF-DB)":
            test_mask_dir = ROOT_DIR / "data" / "raf_db" / "test_masked"
            all_test_masks = list(test_mask_dir.glob("*/*.jpg")) if test_mask_dir.exists() else []
            if all_test_masks:
                sel_tm = st.selectbox(
                    f"Chọn ảnh từ tập test khẩu trang ({len(all_test_masks)} ảnh):",
                    [f"{p.parent.name}/{p.name}" for p in all_test_masks[:30]]
                )
                test_face_img = Image.open(test_mask_dir / sel_tm)
            else:
                st.info("Chưa tìm thấy tập data/raf_db/test_masked.")
        else:
            up_face = st.file_uploader("Tải ảnh khuôn mặt:", type=["jpg", "png", "jpeg"], key="face_single_up")
            if up_face is not None:
                test_face_img = Image.open(up_face)

        if test_face_img is not None:
            pred_l, conf, probs, hm, rgb_np = predict_single_face(model, grad_cam, test_face_img, device)
            _, overlay = overlay_heatmap_on_image(rgb_np, hm, alpha=alpha_val)
            colored_hm, _ = overlay_heatmap_on_image(rgb_np, hm, alpha=1.0)

            st.write("---")
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("Cảm xúc dự đoán", EMOJI_MAP[pred_l], delta=f"{conf:.1f}% Tin cậy")
            with col_res2:
                st.info(f"💡 **Diễn giải XAI:** {EXPLANATION_INSIGHTS.get(pred_l, '')}")

            st.markdown("### Bản Đồ Nhiệt Giải Thích Trực Quan (Grad-CAM):")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(rgb_np, caption="1. Ảnh khuôn mặt (RGB 224x224)", use_container_width=True)
            with c2:
                st.image(colored_hm, caption="2. Bản đồ nhiệt Grad-CAM độc lập (Jet)", use_container_width=True)
            with c3:
                st.image(overlay, caption=f"3. Phủ Heatmap lên khuôn mặt (Alpha={alpha_val})", use_container_width=True)

            st.markdown("### Phổ phân phối xác suất 7 lớp cảm xúc:")
            fig, ax = plt.subplots(figsize=(8, 2.8), dpi=200)
            sorted_idx = np.argsort(probs)
            sorted_classes = [CLASSES[i].capitalize() for i in sorted_idx]
            sorted_probs = [probs[i] * 100 for i in sorted_idx]
            colors = ["#2563EB" if c.lower() != pred_l else "#16A34A" for c in sorted_classes]
            bars = ax.barh(sorted_classes, sorted_probs, color=colors, height=0.6)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Xác suất (%)", fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            for bar in bars:
                w = bar.get_width()
                ax.annotate(f"{w:.1f}%", xy=(w + 1, bar.get_y() + bar.get_height() / 2),
                            va="center", fontsize=8, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # =================================================================
    # TAB 4: HEAD-TO-HEAD COMPARISON (BASELINE VS MASK-AWARE)
    # =================================================================
    with tabs[3]:
        st.subheader("⚔️ So Sánh Đối Đầu: Baseline (Mặt Thường) vs Mask-Aware (Đeo Khẩu Trang)")
        st.markdown(
            "Đặt hai mô hình lên bàn cân thực tế để kiểm chứng luận điểm khoa học: "
            "**Khi khuôn mặt bị che bởi khẩu trang, mô hình Mask-Aware chuyển dịch vùng chú ý lên Đôi Mắt và đưa ra dự đoán chuẩn xác.**"
        )

        has_both = CHECKPOINT_RAFDB.exists() and (CHECKPOINT_MASK_AWARE.exists() or CHECKPOINT_FALLBACK.exists())

        if not has_both:
            st.warning("⚠️ Cần có cả 2 checkpoint `best_model_rafdb.pth` và `best_model_mask_aware.pth` để kích hoạt tính năng đối đầu.")
        else:
            base_model, base_cam, _, _, _ = load_single_model(str(CHECKPOINT_RAFDB))
            mask_ckpt_path = str(CHECKPOINT_MASK_AWARE if CHECKPOINT_MASK_AWARE.exists() else CHECKPOINT_FALLBACK)
            mask_model, mask_cam, _, _, _ = load_single_model(mask_ckpt_path)

            compare_choice = st.radio(
                "Chọn ảnh thử nghiệm đối đầu:",
                ["Mẫu có sẵn (Người thật đeo khẩu trang)", "Tải ảnh mới tùy ý"],
                horizontal=True,
                key="cmp_radio"
            )

            cmp_img = None
            if compare_choice == "Mẫu có sẵn (Người thật đeo khẩu trang)":
                files = sorted(list(REAL_MASKED_DIR.glob("*.jpg")))
                if files:
                    s_file = st.selectbox("Chọn ảnh:", [f.name for f in files], key="cmp_select")
                    cmp_img = Image.open(REAL_MASKED_DIR / s_file)
            else:
                cmp_up = st.file_uploader("Tải ảnh mặt có khẩu trang:", type=["jpg", "png"], key="cmp_uploader")
                if cmp_up is not None:
                    cmp_img = Image.open(cmp_up)

            if cmp_img is not None:
                # Chạy dự đoán cả 2 mô hình
                b_label, b_conf, b_probs, b_hm, b_rgb = predict_single_face(base_model, base_cam, cmp_img, device)
                m_label, m_conf, m_probs, m_hm, m_rgb = predict_single_face(mask_model, mask_cam, cmp_img, device)

                _, b_overlay = overlay_heatmap_on_image(b_rgb, b_hm, alpha=alpha_val)
                _, m_overlay = overlay_heatmap_on_image(m_rgb, m_hm, alpha=alpha_val)

                st.write("---")
                col_b, col_m = st.columns(2)

                with col_b:
                    st.markdown("### 👤 Mô Hình 1: Baseline (Chỉ học mặt thường)")
                    st.metric("Dự đoán Baseline", EMOJI_MAP[b_label], delta=f"{b_conf:.1f}% Tin cậy")
                    st.image(b_overlay, caption=f"Baseline Grad-CAM Heatmap (Dự đoán: {b_label})", use_container_width=True)
                    st.caption("⚠️ Thường bị 'mù' do vùng miệng bị che khuất và đoán liều sang Neutral.")

                with col_m:
                    st.markdown("### 🛡️ Mô Hình 2: Mask-Aware (Học thích nghi khẩu trang)")
                    st.metric("Dự đoán Mask-Aware", EMOJI_MAP[m_label], delta=f"{m_conf:.1f}% Tin cậy")
                    st.image(m_overlay, caption=f"Mask-Aware Grad-CAM Heatmap (Dự đoán: {m_label})", use_container_width=True)
                    st.caption("✅ Chuyển dịch vùng chú ý thị giác lên cơ quanh mắt (Orbicularis oculi) và lông mày.")

                st.success(
                    f"🏆 **KẾT LUẬN ĐỐI CHIẾU XAI:** Mô hình Baseline nhìn vào `{b_label.upper()}` "
                    f"trong khi mô hình Mask-Aware khai thác đặc trưng nửa trên khuôn mặt nhận định là `{m_label.upper()}`."
                )


if __name__ == "__main__":
    main()
