import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
# ---------------------------------------------------------------------
# Đảm bảo in tiếng Việt và Emoji mượt mà trên Terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tự động định vị thư mục gốc dự án dù bạn đứng ở đâu để chạy lệnh
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# 7 cảm xúc chuẩn theo thứ tự nhãn của FER-2013
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---------------------------------------------------------------------
# 2. HÀM ĐẾM SỐ LƯỢNG ẢNH
# ---------------------------------------------------------------------
def count_images(directory: Path) -> dict[str, int]:
    """
    Duyệt qua 7 thư mục con trong 'directory' và đếm số lượng file ảnh (.jpg, .png).
    """
    counts: dict[str, int] = {}
    for cls in CLASSES:
        cls_dir = directory / cls
        if cls_dir.exists():
            # Gom cả file .jpg và .png nếu có
            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            counts[cls] = len(images)
        else:
            counts[cls] = 0
    return counts


# ---------------------------------------------------------------------
# 3. HÀM TÍNH TOÁN TRỌNG SỐ LỚP (CLASS WEIGHTS)
# ---------------------------------------------------------------------
def calculate_class_weights(train_counts: dict[str, int]) -> dict[str, float]:
    """
    BẢN CHẤT TOÁN HỌC:
    Công thức: w_c = N / (C * N_c)
    - N: Tổng số ảnh trong tập Train
    - C: Số lượng lớp (7 lớp)
    - N_c: Số ảnh của riêng lớp c

    Ý nghĩa: Lớp ít ảnh (như Disgust) sẽ có w_c rất cao -> phạt nặng khi model đoán sai.
    """
    total_samples = sum(train_counts.values())
    num_classes = len(train_counts)

    weights: dict[str, float] = {}
    for cls, count in train_counts.items():
        if count > 0:
            # Làm tròn 4 chữ số thập phân
            weights[cls] = round(total_samples / (num_classes * count), 4)
        else:
            weights[cls] = 1.0

    return weights


# ---------------------------------------------------------------------
# 4. HÀM VẼ BIỂU ĐỒ PHÂN BỐ DỮ LIỆU
# ---------------------------------------------------------------------
def plot_distribution(train_counts: dict[str, int], test_counts: dict[str, int], save_path: Path):
    """
    Vẽ biểu đồ cột đôi so sánh số lượng mẫu giữa Train và Test của 7 cảm xúc.
    """
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())

    x = np.arange(len(CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    rects1 = ax.bar(x - width/2, [train_counts[c] for c in CLASSES], width,
                    label=f"Train ({total_train:,} ảnh)", color="#2563EB", alpha=0.9)
    rects2 = ax.bar(x + width/2, [test_counts[c] for c in CLASSES], width,
                    label=f"Test ({total_test:,} ảnh)", color="#F59E0B", alpha=0.9)

    ax.set_ylabel("Số lượng ảnh (Images)", fontsize=12, fontweight="bold")
    ax.set_title("Phân bố số lượng dữ liệu 7 lớp cảm xúc (FER-2013)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Hiển thị số lượng cụ thể trên đỉnh từng cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:,}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="semibold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n📊 Đã lưu biểu đồ phân tích tại: {save_path.name}")


# ---------------------------------------------------------------------
# 5. ĐIỀU KHIỂN CHÍNH
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU GIẢI PHẪU DỮ LIỆU FER-2013")
    print("=" * 70)

    train_counts = count_images(TRAIN_DIR)
    test_counts = count_images(TEST_DIR)
    weights = calculate_class_weights(train_counts)

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())

    # In bảng thống kê chi tiết
    header = f"{'Cảm xúc (Class)':<15} {'Train':<10} {'Tỷ lệ Train (%)':<18} {'Test':<10} {'Class Weight':<15}"
    print(header)
    print("-" * 70)

    for cls in CLASSES:
        n_train = train_counts[cls]
        pct_train = (n_train / total_train) * 100 if total_train > 0 else 0
        n_test = test_counts[cls]
        w = weights[cls]
        print(f"{cls:<15} {n_train:<10,} {f'{pct_train:.2f}%':<18} {n_test:<10,} {w:<15.4f}")

    print("=" * 70)
    print(f"Tổng số ảnh Train : {total_train:,}")
    print(f"Tổng số ảnh Test  : {total_test:,}")
    print(f"Tổng cộng toàn bộ : {total_train + total_test:,} ảnh")
    print("=" * 70)

    # In sẵn tensor để sau này nạp vào PyTorch
    weight_list = [weights[c] for c in CLASSES]
    print("\n💡 Trọng số nạp vào PyTorch Loss Function:")
    print(f"class_weights = torch.tensor({weight_list}, dtype=torch.float)")

    # Lưu biểu đồ vào thư mục gốc dự án
    chart_file = OUTPUT_DIR / "class_distribution.png"
    plot_distribution(train_counts, test_counts, save_path=chart_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
