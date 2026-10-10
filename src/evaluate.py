"""
Module: src/evaluate.py
Mục đích:
1. Nạp checkpoint đã huấn luyện (bất kỳ backbone nào, đọc từ metadata checkpoint).
2. Đánh giá trên 3 tập Test độc lập: Mặt gốc / Mặt có khẩu trang / Tổng hợp.
   Tuỳ chọn --tta: trung bình dự đoán ảnh gốc + ảnh lật ngang.
3. Xuất bảng chỉ số chi tiết từng lớp (Precision, Recall, F1-Score).
4. Xuất ma trận nhầm lẫn chuẩn hóa 'output/confusion_matrix_{ckpt}_{tập}.png'
   và file kết quả 'output/{ckpt}_eval.json' (nguồn số liệu duy nhất cho README / báo cáo).
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Đảm bảo UTF-8 cho Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset import get_benchmark_test_loaders
from models import load_checkpoint_model, get_input_size
from inference import predict_loader, get_device, get_logit_bias


def plot_cm(all_labels, all_preds, class_names, title, save_path):
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar_kws={'label': 'Tỷ lệ dự đoán chuẩn hóa'},
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Đã xuất ma trận nhầm lẫn: {save_path.name}")


def evaluate_model(checkpoint_path: Path, output_dir: Path, tta: bool = False, batch_size: int = 64, num_workers: int = 0,
                   use_bias: bool = True):
    device = get_device()
    print("=" * 70)
    print("🧪 ĐÁNH GIÁ MÔ HÌNH TRÊN CÁC TẬP TEST ĐỘC LẬP")
    print("=" * 70)
    if not checkpoint_path.exists():
        print(f"❌ LỖI: Chưa tìm thấy file checkpoint tại {checkpoint_path}!")
        return

    model, ckpt = load_checkpoint_model(checkpoint_path, device)
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    img_size = int(ckpt.get("img_size", get_input_size(model_name)))
    grouped = ckpt.get("split", {}).get("grouped", False)
    print(f"• Thiết bị          : {device.type.upper()}")
    print(f"• Checkpoint        : {checkpoint_path.name}  ({model_name}, {img_size}px, data_mode={ckpt.get('data_mode', '?')})")
    print(f"• Val F1 lúc train  : {ckpt.get('best_val_f1', float('nan')):.4f}"
          + ("" if grouped else "  ⚠️ checkpoint cũ, val bị rò rỉ -> con số này bị thổi phồng"))
    print(f"• TTA (lật ngang)   : {'BẬT' if tta else 'TẮT'}")
    bias = get_logit_bias(ckpt) if use_bias else None
    info = ckpt.get("logit_bias_info", {})
    if bias is not None:
        print(f"• Prior-bias        : BẬT (t={info.get('t', float('nan')):+.2f}, hiệu chỉnh trên val)")
    else:
        print("• Prior-bias        : TẮT" + ("" if use_bias else " (--no-bias)")
              + ("  ⚠️ chưa hiệu chỉnh, chạy src/calibrate.py" if use_bias and grouped else ""))

    orig_loader, masked_loader, comb_loader, class_names = get_benchmark_test_loaders(
        image_size=img_size, batch_size=batch_size, grayscale=model_name in ("baseline", "improved"),
        num_workers=num_workers)

    stem = checkpoint_path.stem + ("_tta" if tta else "") + ("_nobias" if not use_bias and get_logit_bias(ckpt) is not None else "")
    results = {}
    print("\n| Tập kiểm thử (Test Set)       | Accuracy  | Macro F1 | Số ảnh  |")
    print("|-------------------------------|:---------:|:--------:|:-------:|")
    per_set = {}
    for key, title, loader in [("unmasked", "1. Mặt gốc (Unmasked)", orig_loader),
                               ("masked", "2. Có khẩu trang (Masked)", masked_loader),
                               ("combined", "3. Tổng hợp (Combined)", comb_loader)]:
        if len(loader.dataset) == 0:
            continue
        logits, labels = predict_loader(model, loader, device, tta=tta, bias=bias)
        preds = logits.argmax(1).numpy()
        labels = labels.numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        report = classification_report(labels, preds, labels=list(range(len(class_names))),
                                       target_names=class_names, digits=4, zero_division=0, output_dict=True)
        results[key] = {"acc": acc, "macro_f1": f1, "n": int(len(labels)), "per_class": report}
        per_set[key] = (labels, preds, acc, f1)
        print(f"| {title:<29} |  {acc*100:6.2f}% |  {f1:.4f}  | {len(labels):>7,} |")

    for key in ("unmasked", "masked"):
        if key in per_set:
            labels, preds, _, _ = per_set[key]
            print(f"\n🔍 BÁO CÁO CHI TIẾT — {key.upper()}:")
            print(classification_report(labels, preds, labels=list(range(len(class_names))),
                                        target_names=[c.capitalize() for c in class_names], digits=4, zero_division=0))

    for key, (labels, preds, acc, f1) in per_set.items():
        plot_cm(labels, preds, class_names,
                f"CM {key.capitalize()} ({stem})\nAcc: {acc*100:.2f}% | Macro F1: {f1:.4f}",
                output_dir / f"confusion_matrix_{stem}_{key}.png")

    out_json = output_dir / f"{stem}_eval.json"
    out_json.write_text(json.dumps({"checkpoint": checkpoint_path.name, "model_name": model_name, "tta": tta,
                                    "logit_bias_t": info.get("t") if bias is not None else None,
                                    "best_val_f1": ckpt.get("best_val_f1"), "grouped_split": grouped,
                                    "results": results}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"💾 Đã lưu kết quả: {out_json.name}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình nhận diện cảm xúc")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="output", help="Thư mục lưu kết quả")
    parser.add_argument("--tta", action="store_true", help="Test-Time Augmentation (lật ngang)")
    parser.add_argument("--no-bias", action="store_true", help="Bỏ qua prior-bias đã hiệu chỉnh (xem logits thô)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    output_path = ROOT_DIR / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_file = Path(args.checkpoint)
    else:
        ckpt_file = next((p for p in [output_path / "best_model_mask_aware.pth", output_path / "best_model_rafdb.pth"]
                          if p.exists()), output_path / "best_model.pth")

    evaluate_model(ckpt_file, output_path, tta=args.tta, batch_size=args.batch_size, num_workers=args.num_workers,
                   use_bias=not args.no_bias)
