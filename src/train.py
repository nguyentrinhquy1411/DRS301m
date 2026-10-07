"""
Module: src/train.py
Mục đích:
1. Huấn luyện và Fine-Tuning mô hình trên BỘ DỮ LIỆU THẬT 100%:
   - RAF-DB gốc (15,339 ảnh).
   - Mask-Aware Combined FER (22,129 ảnh gồm cả khuôn mặt gốc và khuôn mặt đeo khẩu trang).
2. Chiến lược Two-Stage Transfer Learning & Continuous Fine-Tuning:
   - Hỗ trợ kế thừa trọng số đã huấn luyện (--init-checkpoint) từ model mặt gốc để thích nghi sang khẩu trang.
   - Warmup Classifier Head & Deep Fine-Tuning với CosineAnnealingLR.
3. Tự động áp dụng Weighted Cross-Entropy Loss khắc phục mất cân bằng dữ liệu.
4. Đánh giá Benchmark 3 chiều: Mặt thường vs Mặt đeo khẩu trang vs Tổng hợp.
5. Lưu checkpoint và biểu đồ lịch sử huấn luyện.
"""

import sys
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

# Đảm bảo UTF-8 cho Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset import (
    get_dataloaders,
    get_benchmark_test_loaders,
    RAF_TRAIN_DIR,
    RAF_TEST_DIR,
    FER_TRAIN_DIR,
    FER_TEST_DIR
)
from models import BaselineCNN, ImprovedCNN, PretrainedFER


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Vòng lặp huấn luyện 1 epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  Huấn luyện", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Đánh giá mô hình trên tập Validation hoặc Test."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    return epoch_loss, epoch_f1, epoch_acc


def plot_history(history, save_path: Path):
    """Vẽ biểu đồ Loss và Macro F1 qua các epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Đồ thị Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563EB", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#DC2626", lw=2)
    ax1.set_title("Biểu đồ Hàm mất mát (Loss)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Đồ thị Macro F1
    ax2.plot(epochs, history["train_f1"], label="Train Macro F1", color="#2563EB", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro F1", color="#16A34A", lw=2)
    ax2.set_title("Biểu đồ Chỉ số Macro F1-Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Đã lưu biểu đồ lịch sử huấn luyện tại: {save_path.name}")


def run_training(args):
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"🚀 THIẾT BỊ HUẤN LUYỆN: {device.type.upper()}")
    if device.type == "cuda":
        print(f"🎮 Card đồ họa        : {torch.cuda.get_device_name(0)}")
    print(f"📁 Bộ dữ liệu          : {args.dataset.upper()} (Chế độ: {args.data_mode.upper()})")
    print(f"🧠 Kiến trúc mô hình   : {args.model.upper()}")
    print(f"⚖️ Áp dụng Class Weight : {'BẬT (Weighted Cross-Entropy)' if args.use_weights else 'TẮT'}")
    print(f"📦 Batch Size          : {args.batch_size}")
    if args.init_checkpoint:
        print(f"📥 Trọng số khởi tạo   : {args.init_checkpoint}")
    print("=" * 70)

    # 1. Định vị thư mục dữ liệu
    if args.dataset.lower() == "raf_db":
        train_path, test_path = RAF_TRAIN_DIR, RAF_TEST_DIR
        img_size = 224
    else:
        train_path, test_path = FER_TRAIN_DIR, FER_TEST_DIR
        img_size = 48 if args.model in ["baseline", "improved"] else 224

    # 2. Nạp dữ liệu
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        train_dir=train_path,
        test_dir=test_path,
        data_mode=args.data_mode,
        batch_size=args.batch_size,
        image_size=img_size,
        num_workers=args.num_workers
    )

    print(f"• Số mẫu: Train={len(train_loader.dataset):,} | Val={len(val_loader.dataset):,} | Test={len(test_loader.dataset):,}")
    print(f"• Danh sách nhãn: {class_names}")

    # 3. Khởi tạo mô hình
    is_pretrained_type = args.model.lower() in ["mobilenet_v3_large", "resnet18"]
    if args.model.lower() == "baseline":
        model = BaselineCNN(num_classes=7).to(device)
    elif args.model.lower() == "improved":
        model = ImprovedCNN(num_classes=7).to(device)
    else:
        # PretrainedFER
        model = PretrainedFER(
            backbone_name=args.model.lower(),
            num_classes=7,
            pretrained=True,
            freeze_backbone=False
        ).to(device)

    # Nạp trọng số khởi tạo nếu có (Transfer từ unmasked model)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if init_path.exists():
            ckpt = torch.load(init_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✅ Đã nạp thành công weights từ checkpoint: {init_path.name}")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint tại {init_path}, khởi tạo chuẩn mặc định.")

    # 4. Hàm mất mát có trọng số
    if args.use_weights:
        weights_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    # Đặt tên checkpoint phù hợp theo chế độ
    if args.data_mode == "combined":
        checkpoint_path = output_dir / "best_model_mask_aware.pth"
    elif args.data_mode == "masked":
        checkpoint_path = output_dir / "best_model_masked_only.pth"
    else:
        checkpoint_path = output_dir / "best_model_rafdb.pth"

    general_checkpoint_path = output_dir / "best_model.pth"

    start_time = time.time()

    # =================================================================
    # GIAI ĐOẠN 1: WARMUP CLASSIFIER HEAD (NẾU CHƯA CÓ INIT CHECKPOINT)
    # =================================================================
    warmup_epochs = args.warmup_epochs if not args.init_checkpoint else 0
    if is_pretrained_type and warmup_epochs > 0:
        print("\n" + "-" * 70)
        print(f"🔥 GIAI ĐOẠN 1: WARMUP CLASSIFIER HEAD ({warmup_epochs} EPOCHS - BACKBONE FROZEN)")
        print("-" * 70)
        model.freeze_backbone()
        optimizer_warmup = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=1e-4
        )

        for ep in range(1, warmup_epochs + 1):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer_warmup, device)
            val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)
            print(f"   [Warmup {ep}/{warmup_epochs}] Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc*100:.2f}%")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

    # =================================================================
    # GIAI ĐOẠN 2: DEEP FINE-TUNING TOÀN MẠNG
    # =================================================================
    total_ft_epochs = args.epochs
    print("\n" + "-" * 70)
    print(f"⚡ GIAI ĐOẠN 2: DEEP FINE-TUNING TOÀN MẠNG ({total_ft_epochs} EPOCHS)")
    print("-" * 70)
    
    if is_pretrained_type:
        model.unfreeze_backbone()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_ft_epochs, eta_min=1e-6)
    patience_counter = 0

    for epoch in range(1, total_ft_epochs + 1):
        lr_current = optimizer.param_groups[0]['lr']
        print(f"\n👉 Epoch {epoch:02d}/{total_ft_epochs:02d} (LR: {lr_current:.6f})")

        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"   [Train] Loss: {train_loss:.4f} | Macro F1: {train_f1:.4f}")
        print(f"   [Val]   Loss: {val_loss:.4f} | Macro F1: {val_f1:.4f} | Accuracy: {val_acc*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_payload = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "dataset": args.dataset,
                "data_mode": args.data_mode,
                "best_val_f1": best_val_f1,
                "classes": class_names,
                "epoch": epoch
            }
            torch.save(save_payload, checkpoint_path)
            torch.save(save_payload, general_checkpoint_path)
            print(f"   🌟 ĐÃ LƯU CHECKPOINT TỐT NHẤT -> {checkpoint_path.name} (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if args.patience > 0:
                print(f"   ⏳ Val F1 không tăng ({patience_counter}/{args.patience} epochs)")
                if patience_counter >= args.patience:
                    print(f"\n🛑 KÍCH HOẠT EARLY STOPPING: Val F1 không cải thiện trong {args.patience} epochs liên tiếp.")
                    print(f"   Dừng sớm tại epoch {epoch:02d} để chống Overfitting!")
                    break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT TRONG: {total_time/60:.2f} phút")
    print(f"🏆 Best Validation Macro F1 : {best_val_f1:.4f}")

    # =================================================================
    # ĐÁNH GIÁ BENCHMARK TOÀN DIỆN VỚI CHECKPOINT TỐT NHẤT
    # =================================================================
    print("\n" + "=" * 70)
    print("🧪 BẢNG BENCHMARK ĐÁNH GIÁ 3 CHIỀU VỚI CHECKPOINT TỐT NHẤT:")
    print("=" * 70)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion_eval = nn.CrossEntropyLoss()

    if args.dataset.lower() == "raf_db":
        orig_loader, masked_loader, comb_loader, _ = get_benchmark_test_loaders(image_size=img_size, batch_size=args.batch_size)
        
        _, f1_orig, acc_orig = evaluate(model, orig_loader, criterion_eval, device)
        _, f1_mask, acc_mask = evaluate(model, masked_loader, criterion_eval, device)
        _, f1_comb, acc_comb = evaluate(model, comb_loader, criterion_eval, device)

        print(f"| Tập kiểm thử (Test Set)       | Accuracy  | Macro F1 | Số ảnh  |")
        print(f"|-------------------------------|:---------:|:--------:|:-------:|")
        print(f"| 1. Mặt gốc (Unmasked)         |  {acc_orig*100:6.2f}% |  {f1_orig:.4f}  |  3,068  |")
        print(f"| 2. Mặt có khẩu trang (Masked) |  {acc_mask*100:6.2f}% |  {f1_mask:.4f}  |  1,975  |")
        print(f"| 3. Tổng hợp (Combined)        |  {acc_comb*100:6.2f}% |  {f1_comb:.4f}  |  5,043  |")
    else:
        _, test_f1, test_acc = evaluate(model, test_loader, criterion_eval, device)
        print(f"• Test Accuracy : {test_acc*100:.2f}% | Macro F1: {test_f1:.4f}")

    print("=" * 70)

    plot_file = output_dir / f"training_history_{args.model}_{args.data_mode}.png"
    plot_history(history, save_path=plot_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện & Fine-tuning mô hình nhận diện cảm xúc Mask-Aware FER")
    parser.add_argument("--model", type=str, default="mobilenet_v3_large",
                        choices=["mobilenet_v3_large", "resnet18", "improved", "baseline"],
                        help="Chọn kiến trúc mô hình (mặc định: mobilenet_v3_large)")
    parser.add_argument("--dataset", type=str, default="raf_db", choices=["raf_db", "fer2013"],
                        help="Chọn bộ dữ liệu (mặc định: raf_db)")
    parser.add_argument("--data-mode", type=str, default="combined", choices=["combined", "original", "masked"],
                        help="Chế độ dữ liệu: combined (gốc + khẩu trang), original, masked (mặc định: combined)")
    parser.add_argument("--init-checkpoint", type=str, default="output/best_model_rafdb.pth",
                        help="Đường dẫn checkpoint để nạp weights khởi đầu (Transfer Learning)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Số epoch khởi động head (mặc định: 3)")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch fine-tuning chính (mặc định: 10)")
    parser.add_argument("--patience", type=int, default=4, help="Số epoch Early Stopping (mặc định: 4, 0 để tắt)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho fine-tuning (mặc định: 0.0001)")
    parser.add_argument("--use-weights", action="store_true", default=True, help="Bật Weighted Cross-Entropy Loss")
    parser.add_argument("--num-workers", type=int, default=0, help="Số CPU workers nạp dữ liệu (mặc định: 0 cho Windows)")
    parser.add_argument("--output-dir", type=str, default="output", help="Thư mục lưu mô hình và biểu đồ")

    args = parser.parse_args()
    run_training(args)
