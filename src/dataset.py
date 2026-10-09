"""
Module: src/dataset.py
Mục đích:
1. Data Pipeline cho RAF-DB:
   - RAF-DB gốc (15,339 ảnh người thật in-the-wild, bản aligned 100x100).
   - RAF-DB Masked (khẩu trang tổng hợp bằng MaskTheFace từ chính ảnh RAF-DB).
2. Hỗ trợ 3 chế độ nạp dữ liệu: 'original' | 'masked' | 'combined'.
3. CHIA TRAIN/VAL THEO NHÓM ẢNH GỐC (chống rò rỉ dữ liệu):
   ảnh gốc `raf_train_00012.jpg` và mọi bản đeo khẩu trang sinh ra từ nó
   (`raf_train_00012_*.jpg`) luôn nằm cùng một phía train hoặc val.
   Danh sách nhóm val được lưu vào `data/splits/` và DÙNG CHUNG cho mọi chế độ / mọi giai đoạn
   -> checkpoint RAF-DB dùng làm khởi tạo cho Mask-Aware không bao giờ "thấy" ảnh val.
4. Augmentation mạnh: RandomResizedCrop, Affine, ColorJitter, Grayscale, RandomErasing
   và LowerFaceOcclusion (che ngẫu nhiên nửa dưới khuôn mặt, mô phỏng khẩu trang / tay che miệng).
5. Trọng số lớp & phân phối prior phục vụ Weighted CE / Logit Adjustment.
"""

import os
import re
import sys
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np
from PIL import Image, ImageDraw

# Đảm bảo UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
# Có thể trỏ sang thư mục dữ liệu khác bằng biến môi trường FER_DATA_DIR
DATA_DIR = Path(os.environ.get("FER_DATA_DIR", ROOT_DIR / "data"))

RAF_TRAIN_DIR = DATA_DIR / "raf_db" / "train"
RAF_TRAIN_MASKED_DIR = DATA_DIR / "raf_db" / "train_masked"
RAF_TEST_DIR = DATA_DIR / "raf_db" / "test"
RAF_TEST_MASKED_DIR = DATA_DIR / "raf_db" / "test_masked"
REAL_MASKED_DIR = DATA_DIR / "real_masked_faces"
SPLIT_DIR = DATA_DIR / "splits"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Chuẩn ImageNet cho mô hình Pre-trained (HSEmotion cũng dùng chuẩn này)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_GROUP_RE = re.compile(r"^(raf_(?:train|test)_\d+)")


def source_group(path: str) -> str:
    """Khoá nhóm = ảnh RAF-DB gốc mà file này được sinh ra từ đó.
    'raf_train_00012.jpg' và 'raf_train_00012_surgical_blue.jpg' -> 'raf_train_00012'."""
    stem = Path(path).stem
    m = _GROUP_RE.match(stem)
    return m.group(1) if m else stem


def to_rgb(img):
    """Hàm cấp module để pickling tương thích hoàn toàn với Windows Multiprocessing."""
    return img.convert("RGB")


class LowerFaceOcclusion:
    """Che ngẫu nhiên nửa dưới khuôn mặt (từ khoảng sống mũi xuống cằm) bằng một mảng màu
    đặc hoặc nhiễu -> buộc mô hình khai thác vùng mắt/lông mày, mô phỏng khẩu trang thật,
    tay che miệng... đa dạng hơn bộ ảnh khẩu trang tổng hợp cố định."""
    COLORS = [(255, 255, 255), (220, 230, 240), (120, 170, 220), (60, 60, 60), (20, 20, 20), (200, 200, 200)]

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        top = h * random.uniform(0.50, 0.62)
        inset = w * random.uniform(0.0, 0.12)
        sag = h * random.uniform(0.0, 0.08)   # mép trên hơi võng như dây khẩu trang
        poly = [(inset, top + sag), (w / 2, top), (w - inset, top + sag), (w, h), (0, h)]
        if random.random() < 0.75:
            ImageDraw.Draw(img).polygon(poly, fill=random.choice(self.COLORS))
        else:
            noise = Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).polygon(poly, fill=255)
            img.paste(noise, (0, 0), mask)
        return img


# ---------------------------------------------------------------------
# 1. HÀM TẠO CÁC PHÉP BIẾN ĐỔI ẢNH (TRANSFORMS)
# ---------------------------------------------------------------------
def get_transforms(image_size: int = 224, grayscale: bool = False, occlusion_p: float = 0.0,
                   erasing_p: float = 0.25):
    """
    - train_transform: RandomResizedCrop + Flip + Affine + ColorJitter + (LowerFaceOcclusion) + RandomErasing
    - eval_transform : Resize + ToTensor + Normalize
    grayscale=True dùng cho BaselineCNN / ImprovedCNN (1 kênh, 48x48).
    """
    if grayscale:
        mean, std = [0.5], [0.5]
        color = [transforms.Grayscale(num_output_channels=1)]
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        color = [transforms.Lambda(to_rgb)]

    train_transform = transforms.Compose(color + [
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.0 if grayscale else 0.1),
        LowerFaceOcclusion(p=occlusion_p),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.2), value="random"),
    ])

    eval_transform = transforms.Compose(color + [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------
# 2. CLASS DATASET DỰA TRÊN DANH SÁCH MẪU (SAMPLES DATASET)
# ---------------------------------------------------------------------
class SamplesDataset(Dataset):
    """Dataset linh hoạt nạp từ danh sách tuple (img_path, class_idx)."""
    def __init__(self, samples: list, classes: list, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = classes
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_folder_samples(root: Path) -> list:
    """Liệt kê (đường dẫn, nhãn) theo cấu trúc root/<tên lớp>/*.jpg, nhãn theo thứ tự CLASSES cố định."""
    root = Path(root)
    if not root.exists():
        return []
    extra = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in CLASSES)
    if extra:
        raise RuntimeError(f"Thư mục {root} có lớp lạ {extra}, chuẩn là {CLASSES}")
    samples = []
    for idx, c in enumerate(CLASSES):
        if (root / c).is_dir():
            samples += [(str(p), idx) for p in sorted((root / c).iterdir()) if p.suffix.lower() in IMG_EXTENSIONS]
    return samples


# ---------------------------------------------------------------------
# 3. TÍNH TOÁN TRỌNG SỐ LỚP & PRIOR
# ---------------------------------------------------------------------
def class_counts(samples: list, num_classes: int = len(CLASSES)) -> np.ndarray:
    return np.bincount([s[1] for s in samples], minlength=num_classes).astype(np.float64)


def compute_class_weights(samples: list, smooth_factor: float = 0.5) -> torch.Tensor:
    """
    Trọng số làm dịu (Square-root Smoothing): w_c = (N / (C * N_c)) ** smooth_factor
    giúp nâng đỡ các lớp thiểu số (fear, disgust) mà không đè bẹp các lớp đa số.
    """
    counts = np.maximum(class_counts(samples), 1)
    raw_weights = counts.sum() / (len(counts) * counts)
    weights = np.power(raw_weights, smooth_factor)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_prior(samples: list) -> torch.Tensor:
    counts = np.maximum(class_counts(samples), 1)
    return torch.tensor(counts / counts.sum(), dtype=torch.float32)


# ---------------------------------------------------------------------
# 4. CHIA TRAIN / VAL THEO NHÓM (CỐ ĐỊNH, DÙNG CHUNG)
# ---------------------------------------------------------------------
def get_val_groups(val_ratio: float = 0.15, seed: int = 42) -> set:
    """
    Tạo (hoặc đọc lại) danh sách nhóm ảnh gốc thuộc tập val.
    - Tính trên TOÀN BỘ ảnh train (gốc + khẩu trang) để kết quả giống nhau với mọi data_mode.
    - Phân tầng theo nhãn để tỷ lệ các lớp ở val giống train.
    - Lưu ra file JSON để mọi lần chạy / mọi giai đoạn dùng đúng một phép chia.
    """
    split_file = SPLIT_DIR / f"val_groups_r{int(val_ratio * 100)}_s{seed}.json"
    if split_file.exists():
        return set(json.loads(split_file.read_text())["val_groups"])

    orig_samples = _image_folder_samples(RAF_TRAIN_DIR)
    masked_samples = _image_folder_samples(RAF_TRAIN_MASKED_DIR)
    orig_groups = {source_group(p) for p, _ in orig_samples}
    orphan = sum(source_group(p) not in orig_groups for p, _ in masked_samples)
    if orig_samples and orphan:
        print(f"⚠️ {orphan} ảnh khẩu trang không map được về ảnh gốc qua tên file -> "
              f"phép chia theo nhóm có thể chưa chặn hết rò rỉ. Chạy scripts/check_leakage.py để kiểm tra.")

    group_label = {}
    for path, label in orig_samples + masked_samples:
        g = source_group(path)
        if group_label.setdefault(g, label) != label:
            raise RuntimeError(f"Nhóm {g} có nhiều nhãn khác nhau — kiểm tra lại dữ liệu khẩu trang!")
    if not group_label:
        raise RuntimeError(f"❌ Không tìm thấy ảnh train trong {RAF_TRAIN_DIR} hoặc {RAF_TRAIN_MASKED_DIR}")

    rng = np.random.default_rng(seed)
    val_groups = []
    for c in range(len(CLASSES)):
        groups_c = sorted(g for g, l in group_label.items() if l == c)
        rng.shuffle(groups_c)
        val_groups.extend(groups_c[:int(round(len(groups_c) * val_ratio))])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps({"val_ratio": val_ratio, "seed": seed, "val_groups": sorted(val_groups)}, indent=0))
    print(f"💾 Đã lưu phép chia val cố định: {split_file}")
    return set(val_groups)


def get_train_val_samples(data_mode: str = "combined", val_ratio: float = 0.15, seed: int = 42):
    samples = []
    if data_mode in ("original", "combined"):
        samples += _image_folder_samples(RAF_TRAIN_DIR)
    if data_mode in ("masked", "combined"):
        samples += _image_folder_samples(RAF_TRAIN_MASKED_DIR)
    if not samples:
        raise RuntimeError(f"❌ Không tìm thấy mẫu train nào với chế độ data_mode='{data_mode}'!")

    val_groups = get_val_groups(val_ratio, seed)
    train = [s for s in samples if source_group(s[0]) not in val_groups]
    val = [s for s in samples if source_group(s[0]) in val_groups]
    assert not ({source_group(p) for p, _ in train} & {source_group(p) for p, _ in val}), "Rò rỉ train/val!"
    return train, val


def get_test_samples(data_mode: str = "combined") -> list:
    samples = []
    if data_mode in ("original", "combined"):
        samples += _image_folder_samples(RAF_TEST_DIR)
    if data_mode in ("masked", "combined"):
        samples += _image_folder_samples(RAF_TEST_MASKED_DIR)
    return samples


# ---------------------------------------------------------------------
# 5. HÀM TẠO CÁC DATALOADER (TRAIN / VAL / TEST)
# ---------------------------------------------------------------------
def _loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def get_dataloaders(
    data_mode: str = "combined",  # 'original', 'masked', 'combined'
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
    image_size: int = 224,
    num_workers: int = 4,
    grayscale: bool = False,
    occlusion_p: float | None = None,
    erasing_p: float = 0.25,
):
    """
    Trả về: train_loader, val_loader, test_loader, class_names, train_samples
    occlusion_p mặc định: 0.0 với 'original' (giữ baseline thuần), 0.3 với 'combined'/'masked'.
    """
    if occlusion_p is None:
        occlusion_p = 0.0 if data_mode == "original" else 0.3
    train_transform, eval_transform = get_transforms(image_size, grayscale, occlusion_p, erasing_p)

    train_samples, val_samples = get_train_val_samples(data_mode, val_ratio, seed)
    test_samples = get_test_samples(data_mode)

    train_dataset = SamplesDataset(train_samples, CLASSES, transform=train_transform)
    val_dataset = SamplesDataset(val_samples, CLASSES, transform=eval_transform)
    test_dataset = SamplesDataset(test_samples, CLASSES, transform=eval_transform)

    train_loader = _loader(train_dataset, batch_size, True, num_workers, drop_last=True)
    val_loader = _loader(val_dataset, batch_size, False, num_workers)
    test_loader = _loader(test_dataset, batch_size, False, num_workers)
    return train_loader, val_loader, test_loader, CLASSES, train_samples


# ---------------------------------------------------------------------
# 6. HÀM TẠO TEST LOADERS ĐỂ BENCHMARK TOÀN DIỆN
# ---------------------------------------------------------------------
def get_benchmark_test_loaders(image_size: int = 224, batch_size: int = 64, grayscale: bool = False,
                               num_workers: int = 0):
    """
    Trả về 3 Test DataLoader riêng biệt để đối chiếu:
    1. orig_loader: ảnh mặt gốc (3,068)
    2. masked_loader: ảnh mặt có khẩu trang
    3. combined_loader: cả hai
    """
    _, eval_transform = get_transforms(image_size=image_size, grayscale=grayscale)
    orig = SamplesDataset(get_test_samples("original"), CLASSES, transform=eval_transform)
    masked = SamplesDataset(get_test_samples("masked"), CLASSES, transform=eval_transform)
    combined = SamplesDataset(orig.samples + masked.samples, CLASSES, transform=eval_transform)
    loaders = [_loader(d, batch_size, False, num_workers) for d in (orig, masked, combined)]
    return (*loaders, CLASSES)


# ---------------------------------------------------------------------
# 7. KHỐI TỰ KIỂM TRA (SELF-TEST)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("🔍 KIỂM THỬ DATA PIPELINE (COMBINED)...")
    print("=" * 65)
    train_loader, val_loader, test_loader, class_names, train_samples = get_dataloaders(
        data_mode="combined", batch_size=32, num_workers=0
    )
    print(f"• Train : {len(train_loader.dataset):,} ảnh | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")
    print(f"• Class Weights : {compute_class_weights(train_samples).tolist()}")
    print(f"• Class Prior   : {compute_class_prior(train_samples).tolist()}")
    images, labels = next(iter(train_loader))
    print(f"• Batch images  : {tuple(images.shape)}  dải [{images.min():.2f}, {images.max():.2f}]")
    print("=" * 65)
