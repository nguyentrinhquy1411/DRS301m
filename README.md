# 🎭 Real-Time Multi-Face Explainable Facial Expression Recognition (XAI-FER)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Occlusions via Pre-Trained Deep Models (HSEmotion EfficientNet / MobileNetV3, YuNet + face alignment, Grad-CAM).

---

## 🌟 Key Highlights

1. **100% Real-World In-The-Wild Benchmark**:
   - **RAF-DB (Real-world Affective Faces Database)**: 15,339 real-world images across 7 emotion categories (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
   - **Mask-Aware FER**: trained on RAF-DB + synthetic masked faces (MaskTheFace) + random lower-face occlusion augmentation.
2. **Transfer Learning**:
   - Default backbone **HSEmotion EfficientNet-B0** (face-pretrained on VGGFace2 → AffectNet), also MobileNetV3-Large / ResNet-18 / EfficientNet-B2.
   - Logit-Adjusted loss for class imbalance, Mixup/CutMix, RandomErasing, EMA, AMP, optional EAC-style flip consistency & knowledge distillation.
3. **Real-Time Multi-Face Detection**:
   - OpenCV **YuNet** detector (5 landmarks) + **face alignment to the RAF-DB template**, so webcam crops match the training distribution. Haar Cascade fallback.
   - IoU face tracking + temporal smoothing of probabilities; horizontal-flip TTA.
4. **Explainable AI (XAI)**:
   - Interactive **Grad-CAM** visual explanation overlays showing which facial regions the model focused on (e.g. eye corners, eyebrows, mouth).
5. **Interactive Web Application**:
   - Built with **Streamlit** supporting Single-Face, Multi-Face scanning, and Head-to-Head Model Comparison (Mask-Aware vs. Baseline).

---

## 📁 Project Structure

```text
├── src/
│   ├── dataset.py          # Data loaders for RAF-DB & Masked benchmarks
│   ├── eda.py              # Exploratory Data Analysis & visualizer
│   ├── evaluate.py         # Evaluation metrics & Confusion Matrix generator
│   ├── face_detector.py    # YuNet detector + RAF-DB face alignment (Haar fallback)
│   ├── gradcam.py          # Grad-CAM explainability implementation
│   ├── inference.py        # Shared predictor (TTA, Grad-CAM) + face tracker for the apps
│   ├── models.py           # PretrainedFER (HSEmotion/EfficientNet, MobileNetV3, ResNet) & CNN models
│   └── train.py            # Fine-tuning pipeline
├── scripts/
│   ├── download_raf_db.py  # Script to download/prepare RAF-DB
│   └── check_leakage.py    # Content-based train/val/test leakage check
├── output/                 # Checkpoints, confusion matrices, training histories
├── docs/                   # Project proposal & methodology documentation
├── app.py                  # Real-Time Streamlit Web App
└── pyproject.toml          # Project configuration & dependencies
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nguyentrinhquy1411/DRS301m.git
cd DRS301m

# Install dependencies using uv or pip
pip install -r requirements.txt
# or with uv:
uv sync
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

### 3. Training & Evaluation (GPU machine)

Expected data layout: `data/raf_db/{train,test,train_masked,test_masked}/<class>/*.jpg`.
Masked images must keep the source file name as a prefix (`raf_train_00012_*.jpg`) so the
train/val split can keep an image and its masked copies on the same side.

```bash
# 0. Check for data leakage (content-based, does not trust file names)
python scripts/check_leakage.py

# 1. Train on original RAF-DB faces  -> output/best_model_rafdb.pth
python src/train.py --model hsemotion_b0 --data-mode original --epochs 30

# 2. Fine-tune Mask-Aware from step 1 -> output/best_model_mask_aware.pth
python src/train.py --model hsemotion_b0 --data-mode combined --epochs 20 \
    --init-checkpoint output/best_model_rafdb.pth

# 3. Evaluate (writes confusion matrices + output/<ckpt>[_tta]_eval.json)
python src/evaluate.py --checkpoint output/best_model_rafdb.pth --tta
python src/evaluate.py --checkpoint output/best_model_mask_aware.pth --tta
```

Useful options: `--model hsemotion_b2` (larger, 260px), `--flip-consistency 1.0` (EAC-style
noisy-label robustness, ~1.5× slower), `--teacher-checkpoint <ckpt>` (knowledge distillation into a
smaller student such as `mobilenet_v3_large`), `--loss weighted_ce` (old behaviour), `--no-amp`.
The validation split is fixed per `--split-seed` and stored in `data/splits/`; keep it unchanged
across steps 1 and 2.

---

## 📊 Benchmark Results

> ⚠️ The current checkpoints in `output/` were trained with the **old pipeline**, whose random
> train/val split put an image and its masked copy on different sides (validation leakage, val F1 0.826
> vs. test F1 0.718). The numbers below are **test-set** results of those checkpoints, taken from
> `output/confusion_matrix_best_model_mask_aware_*.png`. Re-train with the new pipeline and update
> this table from the `*_eval.json` files.

| Model | Evaluation Set | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| Mask-Aware MobileNetV3 (old pipeline) | Unmasked (RAF-DB Test) | 79.86% | 0.7179 |
| Mask-Aware MobileNetV3 (old pipeline) | Masked (synthetic) | 73.27% | 0.6281 |
| Mask-Aware HSEmotion-B0 (new pipeline) | Unmasked / Masked | _to be trained_ | |

---

## 👥 Authors
- **DSR301m Team Project**
