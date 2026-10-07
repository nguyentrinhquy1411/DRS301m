"""
Module: src/evaluate.py
Mục đích:
1. Nạp checkpoint đã huấn luyện ('output/best_model_mask_aware.pth', 'output/best_model_rafdb.pth'...).
2. Đánh giá toàn diện trên cả 3 tập Test độc lập:
   - Tập Mặt Gốc (Unmasked - 3,068 ảnh)
   - Tập Mặt Có Khẩu Trang (Masked - 1,975 ảnh)
   - Tập Tổng Hợp (Combined - 5,043 ảnh)
3. Xuất bảng chỉ số chi tiết từng lớp (Precision, Recall, F1-Score).
4. Xuất ma trận nhầm lẫn chuẩn hóa: 'output/confusion_matrix_{model}_{eval_mode}.png'.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Đảm bảo UTF-8 cho Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset import (
    get_transforms,
    get_benchmark_test_loaders,
    RAF_TEST_DIR,
    RAF_TEST_MASKED_DIR,
    CLASSES
)
from models import BaselineCNN, ImprovedCNN, PretrainedFER


@torch.no_grad()
def eval_loader_metrics(model, loader, device):
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, macro_f1, weighted_f1, all_preds, all_labels


def plot_cm(all_labels, all_preds, class_names, title, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
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
    print(f"🖼️ Đã xuất biểu đồ ma trận nhầm lẫn tại: {save_path.name}")


def evaluate_model(checkpoint_path: Path, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🧪 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN CÁC TẬP TEST ĐỘC LẬP")
    print("=" * 70)
    print(f"• Thiết bị đánh giá : {device.type.upper()}")
    print(f"• Checkpoint nạp vào: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"❌ LỖI: Chưa tìm thấy file checkpoint tại {checkpoint_path}!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenet_v3_large")
    data_mode = checkpoint.get("data_mode", checkpoint.get("dataset", "raf_db"))
    print(f"• Tên mô hình       : {model_name.upper()}")
    print(f"• Chế độ huấn luyện : {str(data_mode).upper()}")

    # Khởi tạo mô hình
    if model_name.lower() in ["mobilenet_v3_large", "resnet18"]:
        model = PretrainedFER(backbone_name=model_name.lower(), num_classes=7, pretrained=False).to(device)
        img_size = 224
    elif model_name.lower() == "baseline":
        model = BaselineCNN(num_classes=7).to(device)
        img_size = 48
    else:
        model = ImprovedCNN(num_classes=7).to(device)
        img_size = 48

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orig_loader, masked_loader, comb_loader, class_names = get_benchmark_test_loaders(image_size=img_size, batch_size=64)

    # Đánh giá 3 tập
    acc_orig, f1_orig, _, preds_orig, labels_orig = eval_loader_metrics(model, orig_loader, device)
    acc_mask, f1_mask, _, preds_mask, labels_mask = eval_loader_metrics(model, masked_loader, device)
    acc_comb, f1_comb, _, preds_comb, labels_comb = eval_loader_metrics(model, comb_loader, device)

    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK:")
    print("=" * 70)
    print(f"| Tập kiểm thử (Test Set)       | Accuracy  | Macro F1 | Số ảnh  |")
    print(f"|-------------------------------|:---------:|:--------:|:-------:|")
    print(f"| 1. Mặt gốc (Unmasked)         |  {acc_orig*100:6.2f}% |  {f1_orig:.4f}  |  {len(labels_orig):,}  |")
    print(f"| 2. Mặt có khẩu trang (Masked) |  {acc_mask*100:6.2f}% |  {f1_mask:.4f}  |  {len(labels_mask):,}  |")
    print(f"| 3. Tổng hợp (Combined)        |  {acc_comb*100:6.2f}% |  {f1_comb:.4f}  |  {len(labels_comb):,}  |")
    print("=" * 70)

    # Báo cáo chi tiết trên tập Masked
    print("\n🔍 BÁO CÁO CHI TIẾT TRÊN TẬP MẶT CÓ KHẨU TRANG (MASKED):")
    print(classification_report(labels_mask, preds_mask, target_names=[c.capitalize() for c in class_names], digits=4))

    # Xuất confusion matrices
    stem = checkpoint_path.stem
    plot_cm(
        labels_orig, preds_orig, class_names,
        f"CM Unmasked ({stem})\nAcc: {acc_orig*100:.2f}% | Macro F1: {f1_orig:.4f}",
        output_dir / f"confusion_matrix_{stem}_unmasked.png"
    )
    plot_cm(
        labels_mask, preds_mask, class_names,
        f"CM Masked ({stem})\nAcc: {acc_mask*100:.2f}% | Macro F1: {f1_mask:.4f}",
        output_dir / f"confusion_matrix_{stem}_masked.png"
    )
    plot_cm(
        labels_comb, preds_comb, class_names,
        f"CM Combined ({stem})\nAcc: {acc_comb*100:.2f}% | Macro F1: {f1_comb:.4f}",
        output_dir / f"confusion_matrix_{stem}_combined.png"
    )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình nhận diện cảm xúc")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="output", help="Thư mục lưu ma trận nhầm lẫn")

    args = parser.parse_args()
    output_path = ROOT_DIR / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_file = Path(args.checkpoint)
    else:
        # Tự động chọn checkpoint tốt nhất
        mask_ckpt = output_path / "best_model_mask_aware.pth"
        rafdb_ckpt = output_path / "best_model_rafdb.pth"
        if mask_ckpt.exists():
            ckpt_file = mask_ckpt
        elif rafdb_ckpt.exists():
            ckpt_file = rafdb_ckpt
        else:
            ckpt_file = output_path / "best_model.pth"

    evaluate_model(checkpoint_path=ckpt_file, output_dir=output_path)
