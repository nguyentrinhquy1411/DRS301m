"""
Script: scripts/download_raf_db.py
Mục đích:
Tải và trích xuất BỘ DỮ LIỆU THẬT 100% ngoài đời thực:
RAF-DB (Real-world Affective Faces Database - 15,339 ảnh người thật in-the-wild).
Dữ liệu lưu tại:
- data/raf_db/train/ (12,271 ảnh)
- data/raf_db/test/ (3,068 ảnh)
Được phân thành 7 thư mục cảm xúc chuẩn:
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""

import sys
import io
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RAF_DB_DIR = ROOT_DIR / "data" / "raf_db"

# Ánh xạ nhãn chuẩn của RAF-DB (1 đến 7) sang tên cảm xúc:
# 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
LABEL_MAP = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

URLS = {
    "test": "https://huggingface.co/datasets/elipaluma/RAF-DB_Kaggle/resolve/main/data/test-00000-of-00001.parquet",
    "train": "https://huggingface.co/datasets/elipaluma/RAF-DB_Kaggle/resolve/main/data/train-00000-of-00001.parquet"
}


def download_and_extract_split(split_name: str, url: str):
    print(f"\n📥 Đang tải tập {split_name.upper()} từ HuggingFace Parquet...")
    df = pd.read_parquet(url)
    total = len(df)
    print(f"• Tổng số ảnh {split_name}: {total:,} ảnh")

    split_dir = RAF_DB_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Tạo trước các thư mục con 7 nhãn
    for emotion in LABEL_MAP.values():
        (split_dir / emotion).mkdir(parents=True, exist_ok=True)

    print(f"• Đang trích xuất ảnh thật vào {split_dir}...")
    success = 0
    for idx, row in tqdm(df.iterrows(), total=total, desc=f"Extracting {split_name}"):
        lbl_num = int(row["label"])
        emotion = LABEL_MAP.get(lbl_num, "unknown")
        if emotion == "unknown":
            continue

        try:
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
            elif hasattr(img_data, "read"):
                img_bytes = img_data.read()
            else:
                img_bytes = bytes(img_data)

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            dst_path = split_dir / emotion / f"raf_{split_name}_{idx:05d}.jpg"
            img.save(dst_path, format="JPEG", quality=95)
            success += 1
        except Exception as e:
            continue

    print(f"✅ Đã trích xuất thành công {success:,} / {total:,} ảnh {split_name}!")


def main():
    print("=" * 65)
    print("🌍 BẮT ĐẦU TẢI BỘ DỮ LIỆU THẬT RAF-DB (REAL-WORLD FACES)")
    print("=" * 65)
    
    # 1. Tải test trước (nhanh, 3,068 ảnh)
    download_and_extract_split("test", URLS["test"])
    
    # 2. Tải train (12,271 ảnh)
    download_and_extract_split("train", URLS["train"])

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT NẠP BỘ DỮ LIỆU THẬT RAF-DB VÀO THƯ MỤC:")
    print(f"📁 {RAF_DB_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
