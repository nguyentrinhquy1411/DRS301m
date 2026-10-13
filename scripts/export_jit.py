"""
Script: scripts/export_jit.py
Mục đích:
Xuất mô hình PyTorch checkpoint (.pth) sang TorchScript JIT (.pt) để có thể
nạp và suy luận trực tiếp trong môi trường C++ / LibTorch / R torch mà KHÔNG cần
phụ thuộc vào Python runtime hay thư viện timm.

Hỗ trợ:
- Xuất best_model_mask_aware.pth -> output/best_model_mask_aware_jit.pt
- Xuất best_model_rafdb.pth      -> output/best_model_rafdb_jit.pt
- Tự động đóng gói vector logit_bias đã hiệu chỉnh vào buffer của mô hình TorchScript.
- Xuất metadata cấu hình sang JSON (.json) để R đọc trực tiếp.
"""

import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from models import load_checkpoint_model, get_input_size  # noqa: E402
from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


class JITFERWrapper(nn.Module):
    """Bọc mô hình FER để TorchScript thực hiện forward và cộng logit_bias tự động."""
    def __init__(self, base_model: nn.Module, logit_bias: list[float] | None = None):
        super().__init__()
        self.base_model = base_model
        if logit_bias is not None and len(logit_bias) == len(CLASSES):
            self.register_buffer("logit_bias", torch.tensor(logit_bias, dtype=torch.float32).view(1, -1))
        else:
            self.register_buffer("logit_bias", torch.zeros(1, len(CLASSES), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return logits + self.logit_bias


def export_checkpoint_to_jit(ckpt_path: Path, output_path: Path | None = None):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ Không tìm thấy checkpoint: {ckpt_path}")
        return False

    device = torch.device("cpu")
    print(f"📦 Đang nạp {ckpt_path.name} ...")
    model, ckpt = load_checkpoint_model(ckpt_path, device=device)
    model.eval()

    model_name = ckpt.get("model_name", "hsemotion_b0")
    img_size = int(ckpt.get("img_size", get_input_size(model_name)))
    logit_bias = ckpt.get("logit_bias", None)

    wrapper = JITFERWrapper(model, logit_bias).eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    # Trace mô hình
    print(f"⚡ Đang TorchScript tracing ({model_name}, size={img_size}x{img_size}) ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    if output_path is None:
        output_path = ckpt_path.with_name(ckpt_path.stem + "_jit.pt")
    output_path = Path(output_path)

    traced.save(str(output_path))
    print(f"✅ Đã lưu TorchScript model: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")

    # Lưu metadata kèm theo cho R đọc
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    meta = {
        "model_name": model_name,
        "input_size": img_size,
        "classes": CLASSES,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "best_val_f1": ckpt.get("best_val_f1", 0.0),
        "logit_bias": logit_bias,
        "split": ckpt.get("split", {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📋 Đã lưu metadata: {meta_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Xuất PyTorch checkpoint sang TorchScript JIT cho R")
    parser.add_argument("--checkpoint", type=str, default="", help="Đường dẫn checkpoint cụ thể")
    args = parser.parse_args()

    if args.checkpoint:
        export_checkpoint_to_jit(Path(args.checkpoint))
    else:
        # Xuất cả 2 checkpoint quan trọng nhất
        ckpts = [
            ROOT_DIR / "output" / "best_model_mask_aware.pth",
            ROOT_DIR / "output" / "best_model_rafdb.pth"
        ]
        for c in ckpts:
            if c.exists():
                export_checkpoint_to_jit(c)


if __name__ == "__main__":
    main()
