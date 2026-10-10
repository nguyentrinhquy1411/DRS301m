"""
Script: scripts/check_leakage.py
Kiểm tra rò rỉ dữ liệu giữa train / val / test, KHÔNG chỉ dựa vào tên file mà cả NỘI DUNG ảnh.

1. Ảnh khẩu trang được sinh ra từ ảnh gốc nào? -> so perceptual hash (dHash) của NỬA TRÊN khuôn mặt
   (vùng mắt/trán không bị khẩu trang che) với toàn bộ ảnh gốc train + test.
   - Ảnh trong test_masked mà khớp với ảnh gốc TRAIN  => RÒ RỈ vào tập test.
   - Ảnh trong train_masked mà khớp với ảnh gốc TEST  => RÒ RỈ vào tập test.
   - Đối chiếu với khoá nhóm theo tên file (dataset.source_group) để chắc phép chia train/val
     theo nhóm đang gom đúng ảnh gốc + bản khẩu trang của nó.
2. Ảnh gốc trùng giữa train và test: dHash toàn ảnh để lọc ứng viên, rồi XÁC NHẬN bằng tương quan
   pixel đã chuẩn hoá (dHash một mình báo nhầm nhiều vì mọi ảnh RAF-DB đều căn theo cùng template).
   Đây là tính chất của split chính thức RAF-DB (~0.5% ảnh test có bản trùng trong train), chỉ cảnh báo.
3. Phép chia val cố định (data/splits/*.json) không chung nhóm với train.

Chạy:  python scripts/check_leakage.py --report output/leakage_report.txt
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset import (  # noqa: E402
    RAF_TRAIN_DIR, RAF_TRAIN_MASKED_DIR, RAF_TEST_DIR, RAF_TEST_MASKED_DIR,
    source_group, get_train_val_samples,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def dhash(img: Image.Image, upper_frac: float = 1.0) -> np.uint64:
    g = img.convert("L")
    if upper_frac < 1.0:
        w, h = g.size
        g = g.crop((0, 0, w, int(h * upper_frac)))
    a = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hashes(paths: list[Path], upper_frac: float, desc: str) -> np.ndarray:
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(tqdm(paths, desc=desc, leave=False)):
        with Image.open(p) as im:
            out[i] = dhash(im, upper_frac)
    return out


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 256):
    """Với mỗi hash query, trả về (index ref gần nhất, khoảng cách Hamming)."""
    idx = np.zeros(len(query), dtype=np.int64)
    dist = np.zeros(len(query), dtype=np.int32)
    for s in range(0, len(query), chunk):
        d = popcount64(query[s:s + chunk, None] ^ ref[None, :])
        idx[s:s + chunk] = d.argmin(1)
        dist[s:s + chunk] = d.min(1)
    return idx, dist


def norm_pixels(path: Path, size: int = 48) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def correlation(a: Path, b: Path) -> float:
    va, vb = norm_pixels(a), norm_pixels(b)
    return float(va @ vb) / va.size


LOG = []


def out(msg: str = ""):
    print(msg)
    LOG.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=int, default=6, help="Ngưỡng Hamming (0-64) coi là cùng ảnh nguồn")
    ap.add_argument("--upper-frac", type=float, default=0.42, help="Tỷ lệ phần trên khuôn mặt dùng để so ảnh khẩu trang")
    ap.add_argument("--dup-corr", type=float, default=0.93, help="Tương quan pixel tối thiểu để xác nhận 2 ảnh gốc là một")
    ap.add_argument("--report", type=str, default="", help="Ghi toàn bộ kết quả ra file (VD output/leakage_report.txt)")
    args = ap.parse_args()
    code = run(args)
    if args.report:
        Path(args.report).write_text("\n".join(LOG) + "\n", encoding="utf-8")
        print(f"💾 Đã ghi báo cáo: {args.report}")
    return code


def run(args):

    dirs = {"train": RAF_TRAIN_DIR, "test": RAF_TEST_DIR, "train_masked": RAF_TRAIN_MASKED_DIR, "test_masked": RAF_TEST_MASKED_DIR}
    files = {k: list_images(v) for k, v in dirs.items()}
    for k, v in files.items():
        out(f"• {k:<13}: {len(v):>6,} ảnh  ({dirs[k]})")
    if not files["train"] or not files["test"]:
        out("❌ Thiếu ảnh RAF-DB gốc, không kiểm tra được.")
        return 1

    problems = 0
    orig_paths = files["train"] + files["test"]
    orig_split = np.array(["train"] * len(files["train"]) + ["test"] * len(files["test"]))
    orig_upper = hashes(orig_paths, args.upper_frac, "hash nửa trên ảnh gốc")

    # 1. Ảnh khẩu trang thuộc về ảnh gốc nào
    group_to_orig = {source_group(str(p)): i for i, p in enumerate(orig_paths)}
    for mk, expected in (("train_masked", "train"), ("test_masked", "test")):
        mpaths = files[mk]
        if not mpaths:
            continue
        mh = hashes(mpaths, args.upper_frac, f"hash {mk}")
        idx, dist = nearest(mh, orig_upper)
        # Ảnh gốc mà TÊN FILE trỏ tới có thực sự là nguồn không? (ưu tiên khi hoà khoảng cách,
        # vì RAF-DB có vài ảnh trùng nhau -> nearest neighbour có thể rơi vào bản trùng)
        own = np.array([group_to_orig.get(source_group(str(m)), -1) for m in mpaths])
        own_dist = np.array([popcount64(np.array([h ^ orig_upper[o]]))[0] if o >= 0 else 99 for h, o in zip(mh, own)])
        name_ok = own_dist <= args.max_dist
        matched = name_ok | (dist <= args.max_dist)
        wrong = ~name_ok & (dist <= args.max_dist) & (orig_split[idx] != expected)
        out(f"\n[{mk}] khớp được ảnh gốc: {matched.sum():,}/{len(mpaths):,} (Hamming ≤ {args.max_dist})")
        out(f"   - tên file trỏ đúng ảnh gốc nguồn   : {name_ok.sum():,}/{len(mpaths):,}")
        out(f"   - nguồn thuộc split khác '{expected}' : {wrong.sum():,}")
        if wrong.any():
            problems += 1
            out("   ❌ RÒ RỈ! Ví dụ:")
            for j in np.where(wrong)[0][:5]:
                out(f"      {mpaths[j].name}  <-  {orig_paths[idx[j]]} (d={dist[j]})")
        if (matched & ~name_ok).any():
            problems += 1
            out("   ⚠️ Tên file KHÔNG trỏ về ảnh gốc nguồn -> phép chia train/val theo nhóm sẽ sai. Ví dụ:")
            for j in np.where(matched & ~name_ok)[0][:5]:
                out(f"      {mpaths[j].name}  <-  {orig_paths[idx[j]].name} (d={dist[j]})")
        if (~matched).any():
            out(f"   ℹ️ {(~matched).sum():,} ảnh không khớp ảnh gốc nào (khẩu trang che quá cao? thử --upper-frac 0.35)")
        out(f"   - phân bố khoảng cách nearest: {sorted(Counter(dist.tolist()).items())[:10]}")

    # 2. Ảnh gốc trùng giữa train và test
    full = hashes(orig_paths, 1.0, "hash toàn ảnh gốc")
    n_tr = len(files["train"])
    idx, dist = nearest(full[n_tr:], full[:n_tr])
    cand = np.where(dist <= args.max_dist)[0]
    corr = {j: correlation(files["test"][j], files["train"][idx[j]]) for j in cand}
    confirmed = sorted((j for j in cand if corr[j] >= args.dup_corr), key=lambda j: -corr[j])
    suspect = sorted((j for j in cand if 0.88 <= corr[j] < args.dup_corr), key=lambda j: -corr[j])
    n_test = len(dist)
    out(f"\n[train vs test gốc] ứng viên dHash (Hamming ≤ {args.max_dist}): {len(cand):,} -> "
        f"xác nhận trùng (tương quan ≥ {args.dup_corr}): {len(confirmed):,}/{n_test:,} ({len(confirmed)/n_test:.2%}), "
        f"nghi ngờ (0.88–{args.dup_corr}): {len(suspect):,}")
    for tag, js in (("trùng", confirmed), ("nghi ngờ", suspect)):
        for j in js:
            te, tr = files["test"][j], files["train"][idx[j]]
            same = "cùng nhãn" if te.parent.name == tr.parent.name else f"KHÁC nhãn ({te.parent.name} vs {tr.parent.name})"
            out(f"      [{tag}] {te.name} ~ {tr.name}  corr={corr[j]:.3f}  {same}")
    if confirmed:
        out("   ⚠️ CẢNH BÁO (không phải lỗi pipeline): split chính thức RAF-DB không tách theo danh tính, một số ảnh test"
            " là cùng người/cùng buổi chụp với ảnh train. Đã đo trên hsemotion_b0: bỏ các ảnh này khỏi test chỉ làm"
            " accuracy giảm ~0.1–0.2 điểm. Nên nêu trong báo cáo.")

    # 3. Phép chia train/val theo nhóm
    tr, va = get_train_val_samples("combined")
    overlap = {source_group(p) for p, _ in tr} & {source_group(p) for p, _ in va}
    out(f"\n[train vs val] train={len(tr):,} val={len(va):,} nhóm chung={len(overlap)}")
    problems += bool(overlap)

    out("\n" + ("✅ KHÔNG PHÁT HIỆN RÒ RỈ." if problems == 0 else f"❌ CÓ {problems} VẤN ĐỀ CẦN XỬ LÝ (xem ở trên)."))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
