"""
Test: tests/test_jit_parity.py
Mục đích:
Kiểm tra tính nhất quán số học (Numerical Parity) giữa mô hình PyTorch gốc (.pth)
và mô hình TorchScript JIT (.pt) sẽ được nạp trong môi trường R.

Yêu cầu:
- Sai số xác suất lớn nhất (Max Diff) phải < 1e-4 across all 7 classes.
- Nhãn cảm xúc dự đoán (Argmax class) phải trùng khớp 100%.
"""

import sys
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from inference import FERPredictor  # noqa: E402
from dataset import CLASSES         # noqa: E402


def test_model_pair(ckpt_path: str, jit_path: str, num_samples: int = 5):
    print("=" * 75)
    print(f"🧪 KIỂM THỬ TÍNH NHẤT QUÁN SỐ HỌC (NUMERICAL PARITY TEST)")
    print(f"• PyTorch Checkpoint   : {ckpt_path}")
    print(f"• TorchScript JIT Model : {jit_path}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Thiết bị kiểm thử    : {device}")

    # Nạp 2 mô hình
    orig_predictor = FERPredictor(ckpt_path, device=device)
    jit_model = torch.jit.load(jit_path, map_location=device).eval()

    test_subsets = [
        ("Ảnh thường (Unmasked)", "data/raf_db/test/*/*.jpg"),
        ("Ảnh khẩu trang (Masked)", "data/raf_db/test_masked/*/*.jpg"),
    ]

    all_max_diffs = []
    np.random.seed(42)

    for subset_name, pattern in test_subsets:
        test_imgs = glob.glob(str(ROOT_DIR / pattern))
        if not test_imgs:
            print(f"⚠️  Bỏ qua {subset_name} (không tìm thấy file)")
            continue

        selected_samples = list(np.random.choice(test_imgs, min(num_samples, len(test_imgs)), replace=False))
        print(f"\n📂 Tập dữ liệu: {subset_name} ({len(selected_samples)} ảnh ngẫu nhiên)")

        for i, img_path in enumerate(selected_samples):
            with Image.open(img_path) as im:
                img_rgb = np.asarray(im.convert("RGB"))

            # 1. Dự đoán từ PyTorch gốc (có logit_bias)
            orig_probs = orig_predictor.predict([img_rgb], tta=False)[0]

            # 2. Dự đoán từ TorchScript JIT (đã tích hợp logit_bias bên trong wrapper)
            tensor, _ = orig_predictor.preprocess([img_rgb])
            with torch.no_grad():
                jit_logits = jit_model(tensor)
                jit_probs = torch.softmax(jit_logits, dim=1)[0].cpu().numpy()

            pred_orig = CLASSES[np.argmax(orig_probs)]
            pred_jit = CLASSES[np.argmax(jit_probs)]
            max_diff = float(np.max(np.abs(orig_probs - jit_probs)))
            all_max_diffs.append(max_diff)

            status = "✅ MATCH" if pred_orig == pred_jit and max_diff < 1e-4 else "❌ MISMATCH"
            print(f"[{i+1:02d}/{len(selected_samples):02d}] {Path(img_path).name:<24} | Orig: {pred_orig:<8} ({np.max(orig_probs)*100:5.1f}%) | JIT: {pred_jit:<8} ({np.max(jit_probs)*100:5.1f}%) | Diff: {max_diff:.2e} | {status}")

            assert pred_orig == pred_jit, f"Lệch nhãn tại {img_path}: {pred_orig} != {pred_jit}"
            assert max_diff < 1e-4, f"Sai số quá lớn ({max_diff:.2e}) tại {img_path}"

    print("-" * 75)
    print(f"📊 Sai số lớn nhất: {max(all_max_diffs):.2e}")
    print(f"🎉 KẾT QUẢ: 100% MẪU TRÙNG KHỚP TUYỆT ĐỐI GIỮA PYTORCH VÀ TORCHSCRIPT!")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    print("\n🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN PYTORCH VS TORCHSCRIPT (JIT)")
    test_model_pair("output/best_model_mask_aware.pth", "output/best_model_mask_aware_jit.pt", num_samples=5)
    test_model_pair("output/best_model_rafdb.pth", "output/best_model_rafdb_jit.pt", num_samples=5)

