"""
Module: src/dataset.py
Mục đích:
1. Data Pipeline chuyên dụng cho BỘ DỮ LIỆU THẬT 100%:
   - RAF-DB (Real-world Affective Faces Database - 15,339 ảnh người thật in-the-wild).
   - RAF-DB Masked (Tập ảnh được tổng hợp khẩu trang qua MaskTheFace với Dlib landmarks).
   - Real Masked Faces (Tập ảnh khuôn mặt đeo khẩu trang thực tế).
2. Hỗ trợ 3 chế độ nạp dữ liệu:
   - 'original': Chỉ ảnh mặt gốc không khẩu trang.
   - 'masked': Chỉ ảnh mặt có khẩu trang.
   - 'combined': Kết hợp cả hai để huấn luyện mô hình Mask-Aware FER toàn diện.
3. Chuẩn hóa RGB 3 kênh (224x224) theo phân phối chuẩn ImageNet phục vụ
   mô hình Pre-trained & Fine-Tuning (MobileNetV3, ResNet).
4. Hỗ trợ tính toán tự động trọng số lớp (Class Weights) cho Weighted Cross-Entropy Loss.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import numpy as np

# Đảm bảo UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Đường dẫn mặc định trỏ đến bộ dữ liệu thật RAF-DB
RAF_TRAIN_DIR = DATA_DIR / "raf_db" / "train"
RAF_TRAIN_MASKED_DIR = DATA_DIR / "raf_db" / "train_masked"
RAF_TEST_DIR = DATA_DIR / "raf_db" / "test"
RAF_TEST_MASKED_DIR = DATA_DIR / "raf_db" / "test_masked"
REAL_MASKED_DIR = DATA_DIR / "real_masked_faces"

# Dự phòng trỏ đến FER-2013 nếu cần
FER_TRAIN_DIR = DATA_DIR / "train"
FER_TEST_DIR = DATA_DIR / "test"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Chuẩn ImageNet cho mô hình Pre-trained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_rgb(img):
    """Hàm cấp module để pickling tương thích hoàn toàn với Windows Multiprocessing."""
    return img.convert("RGB")


# ---------------------------------------------------------------------
# 1. HÀM TẠO CÁC PHÉP BIẾN ĐỔI ẢNH (TRANSFORMS)
# ---------------------------------------------------------------------
def get_transforms(image_size: int = 224):
    """
    Tạo pipeline transform RGB cho mô hình Pre-trained:
    - train_transform: Resize 224x224 + Data Augmentation tự nhiên (Flip, Rotate, Jitter) + Normalize
    - eval_transform : Resize 224x224 + ToTensor + Normalize
    """
    train_transform = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


# ---------------------------------------------------------------------
# 2. CLASS DATASET DỰA TRÊN DANH SÁCH MẪU (SAMPLES DATASET)
# ---------------------------------------------------------------------
class SamplesDataset(Dataset):
    """Dataset linh hoạt nạp từ danh sách tuple (img_path, class_idx)."""
    def __init__(self, samples: list, classes: list, transform=None):
        self.samples = samples
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


class SubsetWithTransform(Dataset):
    """Dataset bọc một subset và áp dụng transform riêng biệt."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        img_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        image = self.subset.dataset.loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------
# 3. TÍNH TOÁN TRỌNG SỐ LỚP (CLASS WEIGHTS)
# ---------------------------------------------------------------------
def compute_class_weights(dataset_or_samples, smooth_factor: float = 0.5) -> torch.Tensor:
    """
    Tính toán trọng số làm dịu (Square-root Smoothing):
    w_c = (N / (C * N_c)) ** smooth_factor
    giúp nâng đỡ các lớp thiểu số (fear, disgust) mà không đè bẹp các lớp đa số (happy, neutral),
    tránh hiện tượng đoán liều làm tụt Accuracy tổng thể.
    """
    if hasattr(dataset_or_samples, "samples"):
        targets = [s[1] for s in dataset_or_samples.samples]
    elif isinstance(dataset_or_samples, list):
        targets = [s[1] for s in dataset_or_samples]
    else:
        raise ValueError("Unsupported input for compute_class_weights")

    classes, counts = np.unique(targets, return_counts=True)
    total_samples = len(targets)
    num_classes = len(classes)
    
    raw_weights = total_samples / (num_classes * counts.astype(np.float32))
    # Áp dụng căn bậc hai làm dịu để tránh chênh lệch quá đà
    weights = np.power(raw_weights, smooth_factor)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------
# 4. HÀM TẠO CÁC DATALOADER (TRAIN / VAL / TEST)
# ---------------------------------------------------------------------
def get_dataloaders(
    train_dir: Path = RAF_TRAIN_DIR,
    test_dir: Path = RAF_TEST_DIR,
    data_mode: str = "combined",  # 'original', 'masked', 'combined'
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
    image_size: int = 224,
    num_workers: int = 0
):
    """
    Nạp dữ liệu ảnh người thật, hỗ trợ nạp kết hợp mặt gốc và mặt khẩu trang.
    Tách 85% Train / 15% Val và trả về:
    train_loader, val_loader, test_loader, class_names, class_weights
    """
    train_transform, eval_transform = get_transforms(image_size=image_size)

    # 1. Thu thập danh sách mẫu Train theo chế độ data_mode
    train_samples = []
    classes = CLASSES

    if data_mode in ["original", "combined"] and Path(train_dir).exists():
        orig_train = ImageFolder(root=str(train_dir))
        train_samples.extend(orig_train.samples)
        classes = orig_train.classes

    if data_mode in ["masked", "combined"] and RAF_TRAIN_MASKED_DIR.exists():
        masked_train = ImageFolder(root=str(RAF_TRAIN_MASKED_DIR))
        train_samples.extend(masked_train.samples)
        classes = masked_train.classes

    if not train_samples:
        raise RuntimeError(f"❌ Không tìm thấy mẫu train nào với chế độ data_mode='{data_mode}'!")

    # 2. Thu thập danh sách mẫu Test
    test_samples = []
    if data_mode == "original" and Path(test_dir).exists():
        orig_test = ImageFolder(root=str(test_dir))
        test_samples.extend(orig_test.samples)
    elif data_mode == "masked" and RAF_TEST_MASKED_DIR.exists():
        masked_test = ImageFolder(root=str(RAF_TEST_MASKED_DIR))
        test_samples.extend(masked_test.samples)
    else:  # "combined"
        if Path(test_dir).exists():
            orig_test = ImageFolder(root=str(test_dir))
            test_samples.extend(orig_test.samples)
        if RAF_TEST_MASKED_DIR.exists():
            masked_test = ImageFolder(root=str(RAF_TEST_MASKED_DIR))
            test_samples.extend(masked_test.samples)

    # 3. Tính class weights trên toàn bộ tập train
    class_weights = compute_class_weights(train_samples)

    # 4. Tách Train / Validation theo tỷ lệ ngẫu nhiên nhưng cố định seed
    total_train = len(train_samples)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size

    rng = np.random.default_rng(seed)
    indices = np.arange(total_train)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sub_samples = [train_samples[i] for i in train_indices]
    val_sub_samples = [train_samples[i] for i in val_indices]

    train_dataset = SamplesDataset(train_sub_samples, classes, transform=train_transform)
    val_dataset = SamplesDataset(val_sub_samples, classes, transform=eval_transform)
    test_dataset = SamplesDataset(test_samples, classes, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, classes, class_weights


# ---------------------------------------------------------------------
# 5. HÀM TẠO TEST LOADERS ĐỂ BENCHMARK TOÀN DIỆN
# ---------------------------------------------------------------------
def get_benchmark_test_loaders(image_size: int = 224, batch_size: int = 64):
    """
    Trả về 3 Test DataLoader riêng biệt để đối chiếu:
    1. orig_loader: 3,068 ảnh mặt gốc
    2. masked_loader: 1,975 ảnh mặt có khẩu trang
    3. combined_loader: 5,043 ảnh cả hai
    """
    _, eval_transform = get_transforms(image_size=image_size)
    orig_ds = ImageFolder(str(RAF_TEST_DIR), transform=eval_transform)
    masked_ds = ImageFolder(str(RAF_TEST_MASKED_DIR), transform=eval_transform)

    combined_samples = orig_ds.samples + masked_ds.samples
    combined_ds = SamplesDataset(combined_samples, orig_ds.classes, transform=eval_transform)

    orig_loader = DataLoader(orig_ds, batch_size=batch_size, shuffle=False)
    masked_loader = DataLoader(masked_ds, batch_size=batch_size, shuffle=False)
    combined_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=False)

    return orig_loader, masked_loader, combined_loader, orig_ds.classes


# ---------------------------------------------------------------------
# 6. KHỐI TỰ KIỂM TRA (SELF-TEST)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("🔍 KIỂM THỬ DATA PIPELINE TRÊN BỘ DỮ LIỆU KẾT HỢP (COMBINED)...")
    print("=" * 65)

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        data_mode="combined",
        batch_size=32,
        image_size=224,
        num_workers=0
    )

    print(f"Danh sách nhãn ({len(class_names)}): {class_names}")
    print(f"• Tập Train Combined : {len(train_loader)} batches ({len(train_loader.dataset):,} ảnh)")
    print(f"• Tập Val Combined   : {len(val_loader)} batches ({len(val_loader.dataset):,} ảnh)")
    print(f"• Tập Test Combined  : {len(test_loader)} batches ({len(test_loader.dataset):,} ảnh)")
    print(f"• Class Weights      : {class_weights.tolist()}")

    images, labels = next(iter(train_loader))
    print("-" * 65)
    print(f"• Batch images shape : {images.shape} (RGB 224x224)")
    print(f"• Batch labels shape : {labels.shape}")
    print(f"• Dtype              : {images.dtype}")
    print(f"• Dải pixel (Min-Max): [{images.min().item():.2f}, {images.max().item():.2f}]")
    print("=" * 65)
    print("✅ DATA PIPELINE DỮ LIỆU KẾT HỢP ĐÃ HOÀN HẢO!")
