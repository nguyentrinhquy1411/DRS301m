# 🎭 Real-Time Multi-Face Explainable Facial Expression Recognition (XAI-FER)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Real Occlusions via Pre-Trained Deep Models (MobileNetV3-Large, MediaPipe, Grad-CAM).

---

## 🌟 Key Highlights

1. **100% Real-World In-The-Wild Benchmark**:
   - **RAF-DB (Real-world Affective Faces Database)**: 15,339 real-world images across 7 emotion categories (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
   - **Mask-Aware FER**: Robust against occlusions (medical masks, hands, eyeglasses) trained on combined unmasked + masked faces.
2. **Transfer Learning SOTA**:
   - Fine-tuned **MobileNetV3-Large** with Weighted Cross-Entropy Loss to handle class imbalance.
3. **Real-Time Multi-Face Detection**:
   - High-speed multi-face detection powered by **MediaPipe BlazeFace** with fallback OpenCV Haar Cascades.
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
│   ├── face_detector.py    # MediaPipe BlazeFace & OpenCV Haar detector
│   ├── gradcam.py          # Grad-CAM explainability implementation
│   ├── models.py           # PretrainedFER (MobileNetV3, ResNet) & CNN models
│   └── train.py            # Deep fine-tuning pipeline
├── scripts/
│   └── download_raf_db.py  # Script to download/prepare RAF-DB
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

### 3. Training & Evaluation

```bash
# Train Mask-Aware model
python src/train.py --model mobilenet_v3_large --data-mode combined --epochs 25

# Evaluate trained checkpoints
python src/evaluate.py --checkpoint output/best_model_mask_aware.pth
```

---

## 📊 Benchmark Results

| Model | Evaluation Set | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Mask-Aware MobileNetV3** | Unmasked (RAF-DB Test) | **83.25%** | **0.7812** |
| **Mask-Aware MobileNetV3** | Masked (Synthetic + Real) | **76.80%** | **0.7245** |
| **Baseline RAF-DB** | Masked (Degraded by occlusion) | 59.40% | 0.5420 |

---

## 👥 Authors
- **DSR301m Team Project**
