"""
Module: src/calibrate.py
Mục đích: Hiệu chỉnh độ lệch prior (prior-bias) cho checkpoint đã train — KHÔNG cần train lại.

Mô hình train bằng Logit-Adjusted loss dự đoán như thể 7 lớp cân bằng, nên trên RAF-DB
(lệch mạnh về happy/neutral) nó đoán thừa fear/disgust: recall cao nhưng precision rất thấp.
Script này:
1. Tính logits (fp32, TTA lật ngang) trên tập VAL theo đúng phép chia nhóm đã dùng khi train.
2. Dò t trong [-0.5, 1.5] để argmax(logits + t*log(prior_train)) đạt macro F1 cao nhất trên VAL.
3. Ghi `logit_bias = t*log(prior)` vào checkpoint. evaluate.py, app.py, webcam_app.py tự áp dụng.

Chạy:  python src/calibrate.py --checkpoint output/best_model_rafdb.pth
       python src/calibrate.py --checkpoint output/best_model_mask_aware.pth
"""

import sys
import argparse
from pathlib import Path
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset import CLASSES, SamplesDataset, get_transforms, get_train_val_samples, compute_class_prior, make_loader
from models import load_checkpoint_model, get_input_size
from inference import predict_loader, calibrate_prior_bias, get_device


def calibrate_checkpoint(ckpt_path, device=None, batch_size: int = 128, num_workers: int = 0,
                         force: bool = False) -> float:
    ckpt_path = Path(ckpt_path)
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    split = ckpt.get("split", {})
    if not split.get("grouped", False) and not force:
        raise SystemExit("❌ Checkpoint cũ (chia val ngẫu nhiên, bị rò rỉ): tập val hiện tại chứa ảnh mô hình đã học, "
                         "hiệu chỉnh sẽ sai. Hãy train lại (hoặc dùng --force nếu chấp nhận).")

    model, _ = load_checkpoint_model(ckpt_path, device)
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    img_size = int(ckpt.get("img_size", get_input_size(model_name)))
    data_mode = ckpt.get("data_mode", "original")

    train_samples, val_samples = get_train_val_samples(data_mode, split.get("val_ratio", 0.15), split.get("seed", 42))
    _, eval_tf = get_transforms(img_size, grayscale=model_name in ("baseline", "improved"))
    val_loader = make_loader(SamplesDataset(val_samples, CLASSES, transform=eval_tf), batch_size, False, num_workers)
    prior = compute_class_prior(train_samples)

    logits, labels = predict_loader(model, val_loader, device, tta=True)
    t, table = calibrate_prior_bias(logits, labels, prior)
    f1_0, acc_0 = table[0.0]
    f1_t, acc_t = table[t]

    print("=" * 66)
    print(f"🎯 HIỆU CHỈNH PRIOR-BIAS: {ckpt_path.name} ({model_name}, data_mode={data_mode}, val={len(val_samples):,} ảnh)")
    print("=" * 66)
    for k in sorted(table):
        if abs(k * 100) % 25 == 0 or k == t:
            mark = "  <== chọn" if k == t else ""
            print(f"   t={k:+.2f}  val macro F1={table[k][0]:.4f}  acc={table[k][1]*100:.2f}%{mark}")
    print(f"• Trước (t=0) : F1 {f1_0:.4f} | Acc {acc_0*100:.2f}%")
    print(f"• Sau  (t={t:+.2f}): F1 {f1_t:.4f} | Acc {acc_t*100:.2f}%")

    ckpt["logit_bias"] = (t * torch.log(prior.clamp_min(1e-8))).tolist()
    ckpt["logit_bias_info"] = {"t": t, "prior": prior.tolist(), "tta": True,
                               "val_macro_f1": [f1_0, f1_t], "val_acc": [acc_0, acc_t]}
    torch.save(ckpt, ckpt_path)
    print(f"💾 Đã ghi logit_bias vào {ckpt_path.name}")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hiệu chỉnh prior-bias trên tập val cho checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="Cho phép hiệu chỉnh checkpoint cũ (val bị rò rỉ)")
    args = ap.parse_args()
    calibrate_checkpoint(args.checkpoint, batch_size=args.batch_size, num_workers=args.num_workers, force=args.force)
