"""
Module: src/train.py
Mục đích:
1. Huấn luyện / Fine-Tuning trên RAF-DB gốc ('original') hoặc RAF-DB + khẩu trang ('combined').
2. Chia train/val theo nhóm ảnh gốc, cố định & dùng chung giữa các giai đoạn (xem dataset.py)
   -> Val F1 phản ánh đúng năng lực tổng quát hoá, early stopping / chọn checkpoint không bị ảo.
3. Recipe huấn luyện:
   - Backbone pre-train trên khuôn mặt (HSEmotion EfficientNet) hoặc ImageNet.
   - Warmup head (đóng băng backbone) -> fine-tune toàn mạng với LR backbone < LR head.
   - Linear warmup + Cosine theo từng bước, AdamW, gradient clipping.
   - Loss: Logit-Adjusted CE (mặc định) / Weighted CE / CE + Label Smoothing.
   - Mixup / CutMix, RandomErasing, LowerFaceOcclusion.
   - EMA trọng số, Mixed Precision (AMP) khi train trên GPU (đánh giá luôn fp32).
   - Hiệu chỉnh prior-bias trên VAL sau khi train (xem calibrate.py).
   - (Tuỳ chọn) Flip-Consistency loss kiểu EAC chống nhãn nhiễu.
   - (Tuỳ chọn) Knowledge Distillation từ một checkpoint teacher lớn hơn.
4. Đánh giá benchmark 3 chiều (gốc / khẩu trang / tổng hợp), có và không có TTA.
"""

import sys
import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import matplotlib
matplotlib.use("Agg")
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
    compute_class_weights,
    compute_class_prior,
)
from models import build_model, load_checkpoint_model, get_input_size, PRETRAINED_BACKBONES
from inference import predict_loader, get_logit_bias
from calibrate import calibrate_checkpoint


# =====================================================================
# TIỆN ÍCH
# =====================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FERLoss(nn.Module):
    """
    Cross-Entropy nhận nhãn mềm (phục vụ Mixup/CutMix + Label Smoothing), với 3 chế độ:
    - 'ce'           : CE thường
    - 'weighted_ce'  : CE có trọng số lớp (căn bậc hai)
    - 'logit_adjust' : Logit Adjustment (Menon et al., ICLR 2021) - cộng tau*log(prior) vào logits
                       khi train; lúc suy luận dùng logits gốc -> cân bằng lớp mà không làm méo
                       xác suất như class weights, thường cải thiện macro F1 cho fear/disgust.
    """
    def __init__(self, mode: str, num_classes: int, class_weights=None, class_prior=None,
                 tau: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.register_buffer("weights", class_weights if mode == "weighted_ce" else torch.ones(num_classes))
        adj = tau * torch.log(class_prior) if mode == "logit_adjust" else torch.zeros(num_classes)
        self.register_buffer("adjust", adj)

    def soft_targets(self, targets: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(targets, self.num_classes).float()
        return onehot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits.float() + self.adjust, dim=1)
        per_sample = -(soft_targets * log_p * self.weights).sum(dim=1)
        norm = (soft_targets * self.weights).sum(dim=1)
        return per_sample.sum() / norm.sum().clamp_min(1e-8)


def mix_batch(images, soft_targets, mixup_alpha: float, cutmix_alpha: float, prob: float):
    """Áp dụng Mixup hoặc CutMix ngẫu nhiên với xác suất `prob`."""
    if prob <= 0 or random.random() > prob or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return images, soft_targets
    perm = torch.randperm(images.size(0), device=images.device)
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = images.shape[2:]
        cut_h, cut_w = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[perm]
    return images, lam * soft_targets + (1 - lam) * soft_targets[perm]


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# VÒNG LẶP HUẤN LUYỆN
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, args, scaler=None,
                    ema=None, teacher=None):
    model.train()
    running_loss, n_seen = 0.0, 0
    all_preds, all_labels = [], []
    use_amp = device.type == "cuda" and args.amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    has_features = hasattr(model, "forward_features")

    pbar = tqdm(loader, desc="  Huấn luyện", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        targets = criterion.soft_targets(labels)
        images, targets = mix_batch(images, targets, args.mixup_alpha, args.cutmix_alpha, args.mix_prob)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if args.flip_consistency > 0 and has_features:
                fmap = model.forward_features(images)
                outputs = model.forward_head(fmap)
                fmap_flip = model.forward_features(torch.flip(images, dims=[3]))
                cons = F.mse_loss(fmap.float(), torch.flip(fmap_flip, dims=[3]).float())
            else:
                outputs = model(images)
                cons = None

            loss = criterion(outputs, targets)
            if cons is not None:
                loss = loss + args.flip_consistency * cons
            if teacher is not None:
                with torch.no_grad():
                    t_in = images
                    if teacher.input_size != images.shape[-1]:
                        t_in = F.interpolate(images, size=teacher.input_size, mode="bilinear", align_corners=False)
                    t_logits = teacher(t_in).float()
                T = args.kd_temp
                kd = F.kl_div(F.log_softmax(outputs.float() / T, 1), F.softmax(t_logits / T, 1),
                              reduction="batchmean") * T * T
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model)

        running_loss += loss.item() * images.size(0)
        n_seen += images.size(0)
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / max(1, n_seen), epoch_f1


def evaluate(model, loader, device, tta=False, bias=None):
    """Trả về (loss CE thường, macro F1, accuracy). Luôn fp32."""
    logits, labels = predict_loader(model, loader, device, tta=tta, bias=bias)
    loss = F.cross_entropy(logits, labels).item()
    preds = logits.argmax(dim=1)
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    acc = (preds == labels).float().mean().item()
    return loss, f1, acc


def plot_history(history, save_path: Path):
    """Vẽ biểu đồ Loss và Macro F1 qua các epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    ax1.plot(epochs, history["train_loss"], label="Train Loss (có aug/mix)", color="#2563EB", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#DC2626", lw=2)
    ax1.set_title("Biểu đồ Hàm mất mát (Loss)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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


def make_optimizer(model, args, head_only: bool):
    is_pretrained = hasattr(model, "head_parameters")
    if head_only:
        return optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.head_lr, weight_decay=args.weight_decay)
    if not is_pretrained:
        return optim.AdamW(model.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    return optim.AdamW([
        {"params": model.backbone_parameters(), "lr": args.lr},
        {"params": model.head_parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


# =====================================================================
# CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def run_training(args):
    seed_everything(args.seed)
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_train_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_name = args.model.lower()
    img_size = get_input_size(model_name)
    grayscale = model_name in ("baseline", "improved")
    is_pretrained_type = model_name in PRETRAINED_BACKBONES

    print("=" * 70)
    print(f"🚀 THIẾT BỊ HUẤN LUYỆN: {device.type.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"📁 Chế độ dữ liệu      : {args.data_mode.upper()}")
    print(f"🧠 Kiến trúc mô hình   : {model_name.upper()} (ảnh vào {img_size}x{img_size})")
    print(f"⚖️ Loss                : {args.loss} | label smoothing={args.label_smoothing}")
    print(f"🔀 Mixup/CutMix        : alpha={args.mixup_alpha}/{args.cutmix_alpha}, p={args.mix_prob}")
    print(f"📦 Batch Size / Epochs : {args.batch_size} / {args.freeze_epochs} (head) + {args.epochs} (full)")
    if args.init_checkpoint:
        print(f"📥 Trọng số khởi tạo   : {args.init_checkpoint}")
    print("=" * 70)

    # 1. Dữ liệu
    train_loader, val_loader, _, class_names, train_samples = get_dataloaders(
        data_mode=args.data_mode,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
        image_size=img_size,
        num_workers=args.num_workers,
        grayscale=grayscale,
        occlusion_p=args.occlusion_p,
        erasing_p=args.erasing_p,
        cache_images=args.cache_images,
    )
    print(f"• Số mẫu: Train={len(train_loader.dataset):,} | Val={len(val_loader.dataset):,} (chia theo nhóm ảnh gốc)")

    # 2. Mô hình
    model_kwargs = {"head": args.head} if (is_pretrained_type and args.head) else {}
    model = build_model(model_name, num_classes=len(class_names), pretrained=True, **model_kwargs)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint khởi tạo: {init_path}")
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        if ckpt.get("model_name") != model_name:
            raise ValueError(f"Checkpoint khởi tạo là '{ckpt.get('model_name')}', khác --model '{model_name}'")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Đã nạp weights khởi tạo từ: {init_path.name}")
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    teacher = None
    if args.teacher_checkpoint:
        teacher, t_ckpt = load_checkpoint_model(args.teacher_checkpoint, device)
        teacher.input_size = int(t_ckpt.get("img_size", get_input_size(t_ckpt.get("model_name", ""))))
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"🎓 Knowledge Distillation từ teacher: {t_ckpt.get('model_name')} (alpha={args.kd_alpha}, T={args.kd_temp})")

    # 3. Loss
    criterion = FERLoss(
        args.loss, len(class_names),
        class_weights=compute_class_weights(train_samples),
        class_prior=compute_class_prior(train_samples),
        tau=args.la_tau,
        label_smoothing=args.label_smoothing,
    ).to(device)

    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.amp and not torch.cuda.is_bf16_supported()) else None
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True) if args.ema_decay > 0 else None

    # Tên checkpoint theo chế độ dữ liệu (app.py / webcam_app.py đọc các tên này)
    default_names = {"combined": "best_model_mask_aware.pth", "masked": "best_model_masked_only.pth", "original": "best_model_rafdb.pth"}
    checkpoint_path = output_dir / (args.output_name or default_names[args.data_mode])

    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    start_time = time.time()
    steps_per_epoch = len(train_loader)

    def eval_and_save(epoch_tag: str, train_loss, train_f1):
        nonlocal best_val_f1
        eval_model = ema.module if ema is not None else model
        val_loss, val_f1, val_acc = evaluate(eval_model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        print(f"   [{epoch_tag}] Train Loss {train_loss:.4f} F1 {train_f1:.4f} | Val Loss {val_loss:.4f} F1 {val_f1:.4f} Acc {val_acc*100:.2f}%")
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": eval_model.state_dict(),
                "model_name": model_name,
                "head": getattr(eval_model, "head_type", None),
                "img_size": img_size,
                "dataset": "raf_db",
                "data_mode": args.data_mode,
                "best_val_f1": best_val_f1,
                "best_val_acc": val_acc,
                "classes": class_names,
                "epoch": epoch_tag,
                "split": {"val_ratio": args.val_ratio, "seed": args.split_seed, "grouped": True},
                "args": vars(args),
            }, checkpoint_path)
            print(f"   🌟 ĐÃ LƯU CHECKPOINT TỐT NHẤT -> {checkpoint_path.name} (Val F1: {best_val_f1:.4f})")
        return improved

    # =================================================================
    # GIAI ĐOẠN 1: WARMUP CLASSIFIER HEAD (BACKBONE FROZEN)
    # =================================================================
    freeze_epochs = args.freeze_epochs if (is_pretrained_type and not args.init_checkpoint) else 0
    if freeze_epochs > 0:
        print("\n" + "-" * 70)
        print(f"🔥 GIAI ĐOẠN 1: WARMUP CLASSIFIER HEAD ({freeze_epochs} EPOCHS - BACKBONE FROZEN)")
        print("-" * 70)
        model.freeze_backbone()
        opt = make_optimizer(model, args, head_only=True)
        sch = build_scheduler(opt, freeze_epochs * steps_per_epoch, 0)
        for ep in range(1, freeze_epochs + 1):
            tl, tf1 = train_one_epoch(model, train_loader, criterion, opt, sch, device, args, scaler, ema, teacher)
            eval_and_save(f"Head {ep}/{freeze_epochs}", tl, tf1)
        model.unfreeze_backbone()
        if ema is not None:   # EMA bắt đầu lại từ trọng số hiện tại cho giai đoạn fine-tune
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)

    # =================================================================
    # GIAI ĐOẠN 2: FINE-TUNING TOÀN MẠNG
    # =================================================================
    print("\n" + "-" * 70)
    print(f"⚡ GIAI ĐOẠN 2: FINE-TUNING TOÀN MẠNG ({args.epochs} EPOCHS | LR backbone {args.lr} / head {args.head_lr})")
    print("-" * 70)
    optimizer = make_optimizer(model, args, head_only=False)
    scheduler = build_scheduler(optimizer, args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        tl, tf1 = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args, scaler, ema, teacher)
        if eval_and_save(f"Epoch {epoch:02d}/{args.epochs}", tl, tf1):
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n🛑 EARLY STOPPING: Val F1 không cải thiện trong {args.patience} epochs liên tiếp (dừng tại epoch {epoch}).")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT TRONG: {total_time/60:.2f} phút")
    print(f"🏆 Best Validation Macro F1 : {best_val_f1:.4f}")

    plot_history(history, output_dir / f"training_history_{model_name}_{args.data_mode}.png")

    # =================================================================
    # ĐÁNH GIÁ BENCHMARK VỚI CHECKPOINT TỐT NHẤT
    # =================================================================
    if args.calibrate:
        calibrate_checkpoint(checkpoint_path, device=device, batch_size=args.batch_size, num_workers=args.num_workers)
    best_model, best_ckpt = load_checkpoint_model(checkpoint_path, device)
    bias = get_logit_bias(best_ckpt)
    loaders = get_benchmark_test_loaders(image_size=img_size, batch_size=args.batch_size, grayscale=grayscale,
                                         num_workers=args.num_workers)[:3]
    names = ["Mặt gốc (Unmasked)", "Có khẩu trang (Masked)", "Tổng hợp (Combined)"]
    results = {}
    print("\n" + "=" * 70)
    print("🧪 BENCHMARK TEST (checkpoint tốt nhất theo Val F1)")
    print("=" * 70)
    print(f"| {'Tập kiểm thử':<24} | Acc thô  | F1 thô   | Acc TTA+bias | F1 TTA+bias | Số ảnh |")
    print(f"|{'-'*26}|----------|----------|--------------|-------------|--------|")
    for name, loader in zip(names, loaders):
        if len(loader.dataset) == 0:
            continue
        _, f1, acc = evaluate(best_model, loader, device, tta=False)
        _, f1_t, acc_t = evaluate(best_model, loader, device, tta=True, bias=bias)
        results[name] = {"acc": acc, "macro_f1": f1, "acc_tta_bias": acc_t, "macro_f1_tta_bias": f1_t,
                         "n": len(loader.dataset)}
        print(f"| {name:<24} | {acc*100:6.2f}% | {f1:.4f}   |    {acc_t*100:6.2f}%   |   {f1_t:.4f}    | {len(loader.dataset):>6,} |")
    print("=" * 70)

    metrics_path = checkpoint_path.with_name(checkpoint_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps({"best_val_f1": best_val_f1, "test": results, "args": vars(args),
                                        "train_minutes": round(total_time / 60, 2)}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"💾 Đã lưu kết quả benchmark: {metrics_path.name}")


def build_parser():
    p = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện cảm xúc Mask-Aware FER")
    p.add_argument("--model", type=str, default="hsemotion_b0",
                   choices=list(PRETRAINED_BACKBONES) + ["improved", "baseline"],
                   help="Kiến trúc mô hình (mặc định: hsemotion_b0)")
    p.add_argument("--head", type=str, default=None, choices=["linear", "mlp"],
                   help="Kiểu classifier head (mặc định: linear cho timm, mlp cho torchvision)")
    p.add_argument("--data-mode", type=str, default="original", choices=["combined", "original", "masked"],
                   help="original: RAF-DB gốc | combined: gốc + khẩu trang | masked: chỉ khẩu trang")
    p.add_argument("--init-checkpoint", type=str, default="",
                   help="Checkpoint khởi tạo (VD: output/best_model_rafdb.pth khi train combined)")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--output-name", type=str, default="", help="Tên file checkpoint (mặc định theo data-mode)")

    p.add_argument("--freeze-epochs", type=int, default=1, help="Số epoch chỉ train head (0 để bỏ qua)")
    p.add_argument("--epochs", type=int, default=30, help="Số epoch fine-tune toàn mạng")
    p.add_argument("--warmup-epochs", type=float, default=1.0, help="Số epoch tăng LR tuyến tính đầu giai đoạn 2")
    p.add_argument("--patience", type=int, default=8, help="Early stopping (0 để tắt)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    p.add_argument("--head-lr", type=float, default=1e-3, help="LR classifier head")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--ema-decay", type=float, default=0.998, help="0 để tắt EMA")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision khi train trên GPU")
    p.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True,
                   help="Giải mã toàn bộ ảnh train/val vào RAM một lần (nhanh hơn nhiều khi num_workers=0)")
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                   help="Hiệu chỉnh prior-bias trên VAL sau khi train (xem calibrate.py)")

    p.add_argument("--loss", type=str, default="logit_adjust", choices=["ce", "weighted_ce", "logit_adjust"])
    p.add_argument("--la-tau", type=float, default=1.0, help="Hệ số tau của Logit Adjustment")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--mix-prob", type=float, default=0.3, help="Xác suất áp dụng Mixup/CutMix cho mỗi batch")
    p.add_argument("--occlusion-p", type=float, default=None,
                   help="Xác suất che nửa dưới mặt (mặc định 0 với original, 0.3 với combined/masked)")
    p.add_argument("--erasing-p", type=float, default=0.25, help="Xác suất RandomErasing")
    p.add_argument("--flip-consistency", type=float, default=0.0,
                   help="Trọng số flip-consistency loss kiểu EAC (VD 1.0; tốn ~1.5x thời gian)")

    p.add_argument("--teacher-checkpoint", type=str, default="", help="Checkpoint teacher cho Knowledge Distillation")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temp", type=float, default=4.0)

    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42, help="Seed phép chia val (giữ nguyên giữa các giai đoạn!)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 4,
                   help="Số workers DataLoader (0 trên Windows để tránh lỗi shm.dll)")
    return p


if __name__ == "__main__":
    run_training(build_parser().parse_args())
