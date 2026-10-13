# Dùng: python scripts/make_study_figures.py <repo_root> <out_dir>  (ví dụ: . docs/images)
"""Vẽ hình minh hoạ (số liệu thật) cho docs/STUDY_GUIDE_QA.md."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT, S = Path(sys.argv[1]), Path(sys.argv[2])
IMG = ROOT / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)
M = json.loads((S / "study_metrics.json").read_text())
FPS = json.loads((ROOT / "output" / "fps_benchmark_m1.json").read_text())
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, dpi=150); plt.close(fig); print("saved", name)


def bars(ax, groups, series, colors, labels, fmt="{:.1f}", label_idx=None):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (vals, c, l) in enumerate(zip(series, colors, labels)):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=c, label=l)
        if label_idx is None or i in label_idx:
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, groups)


# 1. Phân bố lớp
cnt = {d: [len(list((ROOT / "data" / "raf_db" / d / c).glob("*.jpg"))) for c in CLASSES]
       for d in ("train", "train_masked", "test", "test_masked")}
fig, ax = plt.subplots(figsize=(9, 4.2))
bars(ax, CLASSES, [cnt["train"], cnt["train_masked"]], [C1, C2], ["RAF-DB train gốc (12,271)", "train_masked – MaskTheFace (7,830)"], "{:,.0f}")
ax.set_ylabel("Số ảnh"); grid(ax); ax.legend(frameon=False)
ax.set_title("Phân bố 7 lớp cảm xúc – lệch lớp mạnh: happy / fear ≈ 17 lần", loc="left", color=INK, fontweight="bold")
save(fig, "fig01_class_distribution.png")

# 2. Ablation TTA / bias
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (key, title) in zip(axes, [("best_model_rafdb", "HSEmotion-B0 chỉ RAF-DB"), ("best_model_mask_aware", "HSEmotion-B0 Mask-Aware")]):
    T = M[key]["test"]; sets = ["unmasked", "masked", "combined"]
    series = [[100 * T[s][m]["acc"] for s in sets] for m in ("raw", "tta", "bias", "tta_bias")]
    bars(ax, ["Unmasked", "Masked", "Combined"], series, [C1, C2, C3, C4],
         ["Logits thô", "+ TTA", f"+ prior-bias (t={M[key]['t_star']})", "+ TTA + prior-bias"])
    ax.set_title(title, loc="left", color=INK, fontweight="bold"); grid(ax); ax.set_ylim(40, 95)
axes[0].set_ylabel("Accuracy test (%)"); axes[1].legend(frameon=False, loc="lower left", fontsize=8)
save(fig, "fig02_ablation_tta_bias.png")

# 3. Quét t
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for key, c, lab in [("best_model_rafdb", C1, "Mô hình RAF-DB"), ("best_model_mask_aware", C2, "Mô hình Mask-Aware")]:
    sw = M[key]["val_sweep"]; ts = sorted(sw, key=float)
    t = [float(x) for x in ts]
    axes[0].plot(t, [sw[x]["f1"] for x in ts], color=c, lw=2, label=lab)
    axes[1].plot(t, [100 * sw[x]["acc"] for x in ts], color=c, lw=2, label=lab)
    ts_ = str(M[key]["t_star"])
    for ax, v in ((axes[0], sw[ts_]["f1"]), (axes[1], 100 * sw[ts_]["acc"])):
        ax.plot([float(ts_)], [v], "o", ms=8, color=c, mec="#fcfcfb", mew=2)
        ax.annotate(f"t*={ts_}", (float(ts_), v), xytext=(4, -14), textcoords="offset points", fontsize=8, color=INK2)
for ax, yl in zip(axes, ["Macro-F1 trên VAL", "Accuracy trên VAL (%)"]):
    ax.axvline(0, color="#c3c2b7", lw=1, ls="--"); ax.set_xlabel("t (hệ số cộng t·log π vào logits)"); ax.set_ylabel(yl); grid(ax, "both")
axes[0].legend(frameon=False)
axes[0].set_title("Quét t trên tập validation (chọn t* = argmax Macro-F1)", loc="left", color=INK, fontweight="bold")
save(fig, "fig03_t_sweep_val.png")

# 4. F1 từng lớp trên tập masked
fig, ax = plt.subplots(figsize=(10, 4.2))
def per_class_f1(cm):
    cm = np.array(cm); tp = np.diag(cm); p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    return 2 * p * r / np.maximum(p + r, 1e-9)
bars(ax, CLASSES, [per_class_f1(M["best_model_rafdb"]["test"]["masked"]["cm_tstar"]),
                   per_class_f1(M["best_model_mask_aware"]["test"]["masked"]["cm_tstar"])],
     [C1, C2], ["Mô hình RAF-DB", "Mô hình Mask-Aware"], "{:.2f}")
ax.set_ylabel("F1 trên test_masked (1,975 ảnh)"); grid(ax); ax.legend(frameon=False); ax.set_ylim(0, 1.05)
ax.set_title("F1 từng cảm xúc khi đeo khẩu trang (TTA + prior-bias)", loc="left", color=INK, fontweight="bold")
save(fig, "fig04_per_class_f1_masked.png")

# 5. Confusion matrix masked, 2 mô hình
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (key, title) in zip(axes, [("best_model_rafdb", "RAF-DB (không học khẩu trang)"), ("best_model_mask_aware", "Mask-Aware")]):
    cm = np.array(M[key]["test"]["masked"]["cm_tstar"], float); cm = cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(7), CLASSES, rotation=45); ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật"); ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.spines[:].set_visible(False)
save(fig, "fig05_confusion_masked.png")

# 6. FPS
fig, ax = plt.subplots(figsize=(8, 4))
for g, c, lab in [(False, C1, "Tắt Grad-CAM"), (True, C2, "Bật Grad-CAM (tính lại mỗi 10 frame)")]:
    rs = sorted([r for r in FPS["results"] if r["gradcam"] == g], key=lambda r: r["faces"])
    ax.plot([r["faces"] for r in rs], [r["fps"] for r in rs], "-o", color=c, lw=2, ms=8, mec="#fcfcfb", mew=2, label=lab)
    for r in rs:
        ax.annotate(f"{r['fps']:.1f}", (r["faces"], r["fps"]), xytext=(0, -16 if g else 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.axhline(15, color="#c3c2b7", ls="--", lw=1); ax.text(0.85, 15.4, "ngưỡng 15 FPS", ha="left", fontsize=8, color=INK2)
ax.set_xlabel("Số khuôn mặt trong frame 1280×720"); ax.set_ylabel("FPS"); ax.set_xticks(range(1, 6)); ax.set_ylim(11, 60); grid(ax)
ax.legend(frameon=False); ax.set_title("Tốc độ pipeline trên Apple M1 (MPS, không TTA)", loc="left", color=INK, fontweight="bold")
save(fig, "fig06_fps_vs_faces.png")

# 7. Attention shift: heatmap trung bình
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
for k, (key, lab) in enumerate([("best_model_rafdb", "RAF-DB"), ("best_model_mask_aware", "Mask-Aware")]):
    mm = np.load(S / f"meanmap_{key}.npy")
    for j, cond in enumerate(["mặt thường", "đeo khẩu trang"]):
        ax = axes[2 * k + j]; ax.imshow(mm[j], cmap="jet", vmin=0, vmax=max(mm.max(), 1e-6))
        ax.axhline(111.5, color="white", lw=1, ls="--")
        frac = M[key]["cam_upper_frac"]["unmasked" if j == 0 else "masked"]
        ax.set_title(f"{lab} – {cond}\nnửa trên: {100*frac:.0f}% năng lượng CAM", fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Grad-CAM trung bình trên {M['best_model_mask_aware']['cam_n_pairs']:,} cặp ảnh test (giải thích cho nhãn thật)",
             x=0.01, y=0.99, ha="left", color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(IMG / "fig07_attention_shift_mean_cam.png", dpi=150); plt.close(fig)

# 8. Ví dụ Grad-CAM thật
sys.path.append(str(ROOT / "src"))
import torch
from models import load_checkpoint_model
from gradcam import GradCAM, overlay_heatmap_on_image
from dataset import get_transforms
_, tf = get_transforms(224)
loaded = {k: load_checkpoint_model(ROOT / "output" / f"{k}.pth", "cpu") for k in ("best_model_rafdb", "best_model_mask_aware")}
models = {k: v[0] for k, v in loaded.items()}
bias = {k: torch.tensor(v[1]["logit_bias"]) for k, v in loaded.items()}
def final_pred(key, p):
    with torch.no_grad():
        return int((models[key](tf(Image.open(p).convert("RGB")).unsqueeze(0))[0] + bias[key]).argmax())
rows = []
for cls in ("happy", "surprise", "sad"):
    for m in sorted((ROOT / "data/raf_db/test_masked" / cls).glob("*.jpg"))[2:]:
        if final_pred("best_model_mask_aware", m) == CLASSES.index(cls):
            break
    stem = m.name.split("_surgical")[0]
    rows.append((cls, ROOT / "data/raf_db/test" / cls / f"{stem}.jpg", m))
fig, axes = plt.subplots(len(rows), 5, figsize=(12, 2.7 * len(rows)))
cams = {k: GradCAM(m, m.get_last_conv_layer()) for k, m in models.items()}
for r, (cls, po, pm) in enumerate(rows):
    lab = CLASSES.index(cls)
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((224, 224))) for p in (po, pm)]
    axes[r, 0].imshow(imgs[0]); axes[r, 0].set_title(f"Ảnh gốc – nhãn: {cls}", fontsize=9)
    for j, (key, img, p, title) in enumerate([("best_model_rafdb", imgs[0], po, "RAF-DB · mặt thường"),
                                              ("best_model_rafdb", imgs[1], pm, "RAF-DB · khẩu trang"),
                                              ("best_model_mask_aware", imgs[0], po, "Mask-Aware · mặt thường"),
                                              ("best_model_mask_aware", imgs[1], pm, "Mask-Aware · khẩu trang")]):
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        h, _, _ = cams[key].generate_heatmap(x, target_class=lab)
        pred = final_pred(key, p)
        _, ov = overlay_heatmap_on_image(img, h, alpha=0.45)
        ax = axes[r, j + 1]; ax.imshow(ov); ax.set_title(f"{title}\ndự đoán: {CLASSES[pred]}", fontsize=8)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
save(fig, "fig08_gradcam_examples.png")
