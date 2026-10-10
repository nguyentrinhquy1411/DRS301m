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
│   ├── check_leakage.py    # Content-based train/val/test leakage check
│   └── benchmark_fps.py    # Real-time pipeline FPS benchmark
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
python scripts/check_leakage.py --report output/leakage_report.txt

# 1. Train on original RAF-DB faces  -> output/best_model_rafdb.pth
python src/train.py --model hsemotion_b0 --data-mode original --epochs 30

# 2. Fine-tune Mask-Aware from step 1 -> output/best_model_mask_aware.pth
python src/train.py --model hsemotion_b0 --data-mode combined --epochs 20 \
    --init-checkpoint output/best_model_rafdb.pth

# (train.py already calibrates the prior bias on VAL at the end; for older checkpoints run:)
python src/calibrate.py --checkpoint output/best_model_mask_aware.pth

# 3. Evaluate (writes confusion matrices + output/<ckpt>[_tta]_eval.json)
python src/evaluate.py --checkpoint output/best_model_rafdb.pth --tta
python src/evaluate.py --checkpoint output/best_model_mask_aware.pth --tta
```

Useful options: `--model hsemotion_b2` (larger, 260px), `--flip-consistency 1.0` (EAC-style
noisy-label robustness, ~1.5× slower), `--teacher-checkpoint <ckpt>` (knowledge distillation into a
smaller student such as `mobilenet_v3_large`), `--loss weighted_ce` (old behaviour), `--no-amp`, `--no-cache-images`.
Evaluation always runs in fp32 (fp16 autocast was measured to cost ~2 accuracy points).

**Prior-bias calibration.** The logit-adjusted loss makes the model predict as if the 7 classes were
balanced, but RAF-DB train *and* test are dominated by happy/neutral, so the raw model over-predicts
fear/disgust (high recall, low precision). `src/calibrate.py` picks `t` on the **validation** split so that
`argmax(logits + t·log(prior))` maximises macro F1, and stores the bias in the checkpoint; `evaluate.py`
and both apps apply it automatically (`--no-bias` shows raw logits).
The validation split is fixed per `--split-seed` and stored in `data/splits/`; keep it unchanged
across steps 1 and 2.

---

## 📊 Benchmark Results

All numbers are on the official RAF-DB test split, fp32. The validation split is grouped by source
image (no leakage); `t` is always chosen on validation, never on test.

| Model | Test set | Raw logits | TTA + prior bias |
| :--- | :--- | :---: | :---: |
| HSEmotion-B0, RAF-DB only (`best_model_rafdb.pth`, t=0.70) | Unmasked (3,068) | 84.22% / F1 0.764 | **89.15% / F1 0.833** |
| HSEmotion-B0, RAF-DB only | Masked (synthetic, 1,975) | 47.90% / F1 0.451 ¹ | **56.61% / F1 0.534** |
| HSEmotion-B0 Mask-Aware (`best_model_mask_aware.pth`, t=0.85) | Unmasked | 84.06% / F1 0.761 ² | **88.43% / F1 0.822** |
| HSEmotion-B0 Mask-Aware | Masked (synthetic) | 73.22% / F1 0.647 ² | **79.29% / F1 0.708** |
| HSEmotion-B0 Mask-Aware | Combined (5,043) | 79.81% / F1 0.717 ² | **84.85% / F1 0.779** |
| MobileNetV3 Mask-Aware (old pipeline, leaky val) | Unmasked / Masked | 79.86% / 73.27% | — |

¹ measured under fp16 autocast at the end of training (slightly pessimistic).
² with TTA, before prior-bias calibration.

---

## ⚡ Real-Time Performance

`python scripts/benchmark_fps.py --no-tta --cam-every 10` — Apple M1 (16 GB, MPS), Mask-Aware HSEmotion-B0,
1280x720 frames, full pipeline (YuNet + alignment, batched prediction, tracker, Grad-CAM), camera capture excluded.
Raw data: [`output/fps_benchmark_m1.json`](output/fps_benchmark_m1.json).

| Faces in frame | 1 | 2 | 3 | 4 | 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FPS, no Grad-CAM | **55.0** | 37.4 | 28.4 | 20.2 | 17.4 |
| FPS, Grad-CAM refreshed every 10 frames | **37.4** | 25.9 | 20.7 | 17.3 | 15.1 |

Webcam/video modes disable flip-TTA (≤0.1 accuracy points: 88.43% → 88.43%, F1 0.822 → 0.818) and refresh Grad-CAM
every 10 frames (one Grad-CAM pass costs ~100 ms on M1). Together: 1 face with Grad-CAM goes from 7.7 to 37.4 FPS.
Still-image modes (snapshot, group photo, XAI inspector) keep TTA.

---

## 🔒 Data Leakage Audit

Full output: [`output/leakage_report.txt`](output/leakage_report.txt) (`python scripts/check_leakage.py --report ...`).
The check matches images by **content** (perceptual hash, then normalized pixel correlation), not by file name.

| Check | Result |
| :--- | :--- |
| `test_masked` → source image | 1,973/1,975 matched to an original **test** image, **0 from train** (2 too occluded to match) |
| `train_masked` → source image | 7,816/7,830 matched to the original their file name points to, 0 from test; 2 hash mismatches confirmed as false alarms by pixel correlation, 12 too occluded to match |
| Train vs. validation | 17,099 / 3,002 images, **0 shared source images** (split grouped by source image, fixed in `data/splits/`) |
| Train vs. test (original RAF-DB) | 49/3,068 test images (1.6%) show the same person / same shoot as a train image |

**Verdict: no leakage introduced by the pipeline.** The 49 train/test near-duplicates come from the official
RAF-DB split, which is not identity-disjoint. Removing them from the test set changes accuracy by only
0.05 points (RAF-DB model, 89.15% → 89.10%) and 0.19 points (Mask-Aware, 88.43% → 88.24%), so the results
above are unaffected.

The old pipeline (MobileNetV3 row above) split train/val randomly *after* mixing originals with their masked
copies, so a validation image's masked twin was often in train: val F1 0.826 vs. test F1 0.718.

---

## 👥 Authors
- **DSR301m Team Project**
