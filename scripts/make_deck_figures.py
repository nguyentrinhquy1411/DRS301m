# Dùng: python scripts/make_deck_figures.py   (cần docs/images/study_metrics.json từ study_analysis.py)
"""Hình tiếng Anh (cùng bảng màu slide) cho docs/XAI-FER_Slides_v2.pptx — mọi hình đều từ dữ liệu / mô hình thật."""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from dataset import get_transforms, CLASSES  # noqa: E402
from models import load_checkpoint_model  # noqa: E402
from gradcam import GradCAM, overlay_heatmap_on_image  # noqa: E402
from face_detector import MultiFaceDetector  # noqa: E402

OUT = ROOT / "docs" / "images" / "deck"
OUT.mkdir(parents=True, exist_ok=True)
M = json.loads((ROOT / "docs" / "images" / "study_metrics.json").read_text())
INK, GREEN, MINT, TERRA, MUTED, GRID, BG = "#2F3E3A", "#4F7F6E", "#DCEBE3", "#B5604A", "#66756F", "#E3ECE7", "#F7F9F8"
plt.rcParams.update({"font.family": "Calibri", "font.size": 12, "axes.edgecolor": "#C9D3CF", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
RAF_TEST, RAF_MASK = ROOT / "data/raf_db/test", ROOT / "data/raf_db/test_masked"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", name)


def pair_paths(cls, k):
    """k cặp (ảnh gốc, ảnh khẩu trang) của lớp cls."""
    out = []
    for m in sorted((RAF_MASK / cls).glob("*.jpg")):
        o = RAF_TEST / cls / (m.name.split("_surgical")[0] + ".jpg")
        if o.exists():
            out.append((o, m))
        if len(out) == k:
            break
    return out


# 1. Cặp ảnh RAF-DB gốc / MaskTheFace (slide 3)
pairs = [pair_paths(c, 3)[2] for c in ("happy", "surprise", "sad", "angry", "neutral", "fear")]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.3))
for j, (o, m) in enumerate(pairs):
    for i, p in enumerate((o, m)):
        axes[i, j].imshow(Image.open(p).convert("RGB")); axes[i, j].axis("off")
    axes[0, j].set_title(p.parent.name, fontsize=13, color=INK)
axes[0, 0].text(-12, 50, "RAF-DB", rotation=90, va="center", ha="right", fontsize=13, color=MUTED)
axes[1, 0].text(-12, 50, "MaskTheFace", rotation=90, va="center", ha="right", fontsize=13, color=MUTED)
fig.subplots_adjust(wspace=0.04, hspace=0.12)
save(fig, "deck_raf_mask_pairs.png")

# 2. YuNet + 5 landmark + căn chỉnh (slide 5): frame tổng hợp như benchmark_fps.py — 3 mặt RAF-DB bị xoay nghiêng
det = MultiFaceDetector(backend="yunet")
canvas = np.full((520, 1100, 3), 205, np.uint8)
srcs = [pair_paths("happy", 6)[5][0], pair_paths("surprise", 6)[5][1], pair_paths("sad", 6)[5][0]]
for (cx, cy, size, ang), p in zip([(200, 270, 250, 18), (550, 250, 290, -14), (900, 280, 230, 8)], srcs):
    f = cv2.resize(np.asarray(Image.open(p).convert("RGB")), (size, size))
    pad = size // 3
    f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(205, 205, 205))
    R = cv2.getRotationMatrix2D((f.shape[1] / 2, f.shape[0] / 2), ang, 1.0)
    f = cv2.warpAffine(f, R, f.shape[1::-1], borderValue=(205, 205, 205))
    h, w = f.shape[:2]; y0, x0 = cy - h // 2, cx - w // 2
    ys, xs = slice(max(y0, 0), min(y0 + h, canvas.shape[0])), slice(max(x0, 0), min(x0 + w, canvas.shape[1]))
    canvas[ys, xs] = f[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
faces = sorted(det.detect_faces(canvas, is_bgr=False), key=lambda f: f.bbox[0])
vis = canvas.copy()
for f in faces:
    x, y, w, h = f.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (79, 127, 110), 3)
    for (px, py) in f.landmarks:
        cv2.circle(vis, (int(px), int(py)), 6, (181, 96, 74), -1)
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.55], hspace=0.14, wspace=0.06)
ax = fig.add_subplot(gs[0, :]); ax.imshow(vis); ax.axis("off")
ax.set_title(f"Tilted faces: YuNet box + 5 landmarks ({len(faces)} faces)", fontsize=13, color=INK, loc="left")
for j, f in enumerate(faces[:3]):
    a = fig.add_subplot(gs[1, j]); a.imshow(f.crop_rgb); a.axis("off")
    a.set_title("aligned to RAF-DB template" if j == 0 else "", fontsize=11, color=MUTED, loc="left")
save(fig, "deck_yunet_alignment.png")
print("faces detected:", len(faces))

# 3. Quét t trên val (slide 8)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for key, c, lab in [("best_model_rafdb", GREEN, "RAF-DB model"), ("best_model_mask_aware", TERRA, "Mask-Aware model")]:
    sw = M[key]["val_sweep"]; ts = sorted(sw, key=float)
    ax.plot([float(t) for t in ts], [sw[t]["f1"] for t in ts], color=c, lw=2.2, label=lab)
    t0 = str(M[key]["t_star"]); v = sw[t0]["f1"]
    ax.plot([float(t0)], [v], "o", ms=9, color=c, mec=BG, mew=2)
    ax.annotate(f"t* = {t0}", (float(t0), v), xytext=(6, 6), textcoords="offset points", fontsize=11, color=c)
ax.axvline(0, color="#C9D3CF", lw=1, ls="--")
ax.text(0.03, 0.585, "t = 0 (raw logits)", fontsize=10, color=MUTED)
ax.set_xlabel("t   (logits + t · log prior)"); ax.set_ylabel("Validation Macro-F1")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right")
save(fig, "deck_t_sweep_val.png")

# 4. Confusion matrix trên ảnh khẩu trang (slide mới)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
for ax, (key, title) in zip(axes, [("best_model_rafdb", "RAF-DB-only model"), ("best_model_mask_aware", "Mask-Aware model")]):
    cm = np.array(M[key]["test"]["masked"]["cm_tstar"], float); cm /= cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", [BG, MINT, GREEN, INK]), vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            hl = (key == "best_model_rafdb" and (i, j) == (3, 4)) or (key == "best_model_mask_aware" and (i, j) == (2, 6))
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > 0.5 else (TERRA if hl else INK), fontweight="bold" if hl else "normal")
    ax.set_xticks(range(7), CLASSES, rotation=35); ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True label"); ax.set_title(title, fontsize=14, color=INK, loc="left")
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
fig.tight_layout(w_pad=3)
save(fig, "deck_confusion_masked.png")

# 5. Ví dụ Grad-CAM thật (slide 12)
_, tf = get_transforms(224)
loaded = {k: load_checkpoint_model(ROOT / "output" / f"{k}.pth", "cpu") for k in ("best_model_rafdb", "best_model_mask_aware")}
cams = {k: GradCAM(m, m.get_last_conv_layer()) for k, (m, _) in loaded.items()}
def pred(key, p):
    m, ck = loaded[key]
    with torch.no_grad():
        return int((m(tf(Image.open(p).convert("RGB")).unsqueeze(0))[0] + torch.tensor(ck["logit_bias"])).argmax())
rows = []
for cls in ("happy", "sad"):
    for o, m in pair_paths(cls, 40)[2:]:
        if pred("best_model_mask_aware", m) == CLASSES.index(cls):
            rows.append((cls, o, m)); break
cols = [("best_model_rafdb", 0, "RAF-DB · no mask"), ("best_model_rafdb", 1, "RAF-DB · mask"),
        ("best_model_mask_aware", 0, "Mask-Aware · no mask"), ("best_model_mask_aware", 1, "Mask-Aware · mask")]
fig, axes = plt.subplots(len(rows), 5, figsize=(12.5, 5.3))
for r, (cls, o, m) in enumerate(rows):
    lab = CLASSES.index(cls)
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((224, 224))) for p in (o, m)]
    axes[r, 0].imshow(imgs[0]); axes[r, 0].set_title(f"true: {cls}", fontsize=12, color=INK)
    for j, (key, k, title) in enumerate(cols):
        p = (o, m)[k]
        h, _, _ = cams[key].generate_heatmap(tf(Image.open(p).convert("RGB")).unsqueeze(0), target_class=lab)
        axes[r, j + 1].imshow(overlay_heatmap_on_image(imgs[k], h, alpha=0.45)[1])
        pc = CLASSES[pred(key, p)]
        axes[r, j + 1].set_title(f"{title}\npred: {pc}", fontsize=11, color=INK if pc == cls else TERRA)
    for a in axes[r]:
        a.axis("off")
fig.subplots_adjust(wspace=0.04, hspace=0.3)
save(fig, "deck_gradcam_examples.png")
