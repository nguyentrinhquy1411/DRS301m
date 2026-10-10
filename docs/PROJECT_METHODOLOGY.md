# 🎭 Project Methodology: Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Occlusions via Pre-Trained Deep Models

---

## 📌 THÔNG TIN TỔNG QUAN DỰ ÁN

* **Tên đề tài:** Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Occlusions using Pre-trained Deep Models (XAI-FER Mask-Aware)
* **Quy mô nhóm:** 3 thành viên
* **Định hướng thực hiện:** Huấn luyện và đánh giá trên ảnh người thật in-the-wild (**RAF-DB**), mở rộng sang khuôn mặt đeo khẩu trang bằng ảnh khẩu trang **tổng hợp** (MaskTheFace) sinh ra từ chính ảnh RAF-DB, tận dụng backbone tiền huấn luyện trên khuôn mặt, và triển khai ứng dụng thời gian thực quét nhiều khuôn mặt đồng thời kèm giải thích Grad-CAM.
* **Điểm khác biệt cốt lõi (Key Contribution):**
  1. **Mask-Aware FER:** mô hình giữ nguyên độ chính xác trên mặt thường (88.43% so với 89.15%) nhưng tăng **+22.7 điểm** accuracy trên mặt đeo khẩu trang (79.29% so với 56.61%) so với mô hình chỉ train trên RAF-DB.
  2. **Quy trình đánh giá không rò rỉ dữ liệu:** chia train/val theo ảnh gốc, kiểm tra rò rỉ bằng nội dung ảnh (`scripts/check_leakage.py`), mọi siêu tham số hiệu chỉnh đều chọn trên tập val, không bao giờ trên test.
  3. **Backbone tiền huấn luyện trên khuôn mặt:** HSEmotion EfficientNet-B0 (VGGFace2 → AffectNet) thay cho backbone ImageNet, đạt **89.15% / macro F1 0.833** trên RAF-DB test.
  4. **Pipeline thời gian thực nhất quán với lúc train:** detector YuNet + căn chỉnh khuôn mặt theo đúng template của RAF-DB aligned, theo dõi từng khuôn mặt qua các frame, Grad-CAM tương tác.

---

## 1. ĐẶT VẤN ĐỀ VÀ CHIẾN LƯỢC BÁO CÁO (PITCHING STRATEGY)

### 1.1. Problem Statement (Vấn đề thực tế)
1. **Hạn chế của mô hình Black-Box:** Các hệ thống nhận diện cảm xúc khuôn mặt (FER) truyền thống đưa ra nhãn phân loại mà không giải thích được lý do, gây thiếu tin cậy trong các ứng dụng giáo dục, y tế và giám sát tương tác.
2. **Khoảng cách giữa phòng thí nghiệm và đời thực (In-The-Wild Gap):** Mô hình train từ đầu trên ảnh xám nhỏ 48x48 thường sụp đổ hiệu năng ngoài đời thực — nơi khuôn mặt có màu sắc tự nhiên, góc nghiêng đa dạng, ánh sáng phức tạp và bị che khuất (kính mắt, khẩu trang, tay che miệng).
3. **Khẩu trang che mất nửa dưới khuôn mặt:** Mô hình FER thông thường dựa nhiều vào miệng; khi bị che, nó đoán dồn về `neutral` (thực nghiệm: recall `neutral` 0.82 nhưng precision chỉ 0.37, recall `happy` tụt còn 0.44).
4. **Môi trường thực tế có nhiều người (Multi-Face Scenario):** Camera và ứng dụng thực tế đòi hỏi phát hiện, căn chỉnh và dự đoán cảm xúc của **nhiều khuôn mặt cùng lúc** với nhãn ổn định giữa các frame.

### 1.2. Giải pháp đề xuất (Proposed Solution)
1. **Dữ liệu:** RAF-DB (15,339 ảnh người thật in-the-wild) + RAF-DB Masked (khẩu trang tổng hợp bằng MaskTheFace từ chính ảnh RAF-DB, giữ nguyên nhãn) + augmentation che ngẫu nhiên nửa dưới khuôn mặt.
2. **Mô hình:** fine-tune HSEmotion EfficientNet-B0 qua 2 giai đoạn: RAF-DB gốc → RAF-DB gốc + khẩu trang (Mask-Aware), khởi tạo từ checkpoint giai đoạn 1.
3. **Huấn luyện chống mất cân bằng lớp:** Logit-Adjusted loss khi train + hiệu chỉnh prior-bias trên tập val sau khi train.
4. **Pipeline thời gian thực:** YuNet (5 landmark) → căn chỉnh theo template RAF-DB → dự đoán theo batch (TTA lật ngang) → tracker IoU + làm mượt xác suất → Grad-CAM cho khuôn mặt được chọn.

### 1.3. Tuyên ngôn khi thuyết trình (Elevator Pitch)
> *"Our system recognises facial emotions of multiple people in real time — even when they wear a mask. A face-pretrained EfficientNet fine-tuned on RAF-DB plus masked faces reaches 88.4% on normal faces and 79.3% on masked faces, 22.7 points above the same model trained without masks. Every number is measured on a leak-free evaluation protocol, and every prediction can be explained with an interactive Grad-CAM heatmap."*

---

## 2. BỘ DỮ LIỆU & KIẾN TRÚC HỆ THỐNG

### 2.1. Thống kê Dữ liệu
* **RAF-DB Train (`data/raf_db/train`):** **12,271 ảnh** (bản aligned 100x100, split chính thức)
  * `happy`: 4,772 | `neutral`: 2,524 | `sad`: 1,982 | `surprise`: 1,290 | `disgust`: 717 | `angry`: 705 | `fear`: 281
* **RAF-DB Test (`data/raf_db/test`):** **3,068 ảnh**
  * `happy`: 1,185 | `neutral`: 680 | `sad`: 478 | `surprise`: 329 | `angry`: 162 | `disgust`: 160 | `fear`: 74
* **RAF-DB Masked (khẩu trang tổng hợp, MaskTheFace):** `train_masked` **7,830 ảnh**, `test_masked` **1,975 ảnh**. Tên file giữ tiền tố ảnh gốc (`raf_train_00012_*.jpg`) để truy được ảnh nguồn.
* **Ảnh người thật đeo khẩu trang thật (`data/real_masked_faces`):** bộ ảnh nhỏ chỉ dùng để **demo định tính** trong app (tab XAI / So sánh), **không** dùng để tính số liệu.
* **Chia train/val:** 15% val, **theo nhóm ảnh gốc**, phân tầng theo nhãn, cố định trong `data/splits/` và dùng chung cho cả 2 giai đoạn. Chế độ combined: 17,099 train / 3,002 val.

```text
               Video / Webcam Feed (Nhiều người trong khung hình)
                                      │
                                      ▼
               Phase 1: Multi-Face Detection + Face Alignment
        (OpenCV YuNet, 5 landmark → similarity transform về template RAF-DB)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
          Aligned Face #1                           Aligned Face #N
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                 Phase 2: Preprocessing (Resize 224x224, ImageNet Norm)
                                      │
                                      ▼
                 Phase 3: Face-Pretrained Backbone (HSEmotion EfficientNet-B0)
         ┌─────────────────────────────────────────────────────────┐
         │ • Giai đoạn 1: RAF-DB gốc (warmup head → fine-tune)     │
         │ • Giai đoạn 2: RAF-DB + khẩu trang (Mask-Aware)         │
         │ • Logit-Adjusted loss, Mixup/CutMix, RandomErasing,     │
         │   che nửa dưới mặt ngẫu nhiên, EMA, AMP                 │
         └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                 Phase 4: Inference = TTA lật ngang + prior-bias (chọn trên val)
                          + Tracker IoU & làm mượt xác suất theo frame
                                      │
                                      ▼
                 Phase 5: Leak-free Evaluation (Acc, Macro F1, Confusion Matrix
                          trên Unmasked / Masked / Combined) + Leakage Audit
                                      │
                                      ▼
                 Phase 6: Explainable AI (Grad-CAM)
                                      │
                                      ▼
                 Phase 7: Streamlit App + Desktop Webcam App
```

---

## 3. CHI TIẾT CÁC GIAI ĐOẠN THỰC HIỆN

### PHASE 1: DATA PIPELINE & PREPROCESSING
*Người phụ trách chính: Thành viên 1* — `src/dataset.py`
* Nạp RAF-DB gốc + RAF-DB Masked theo 3 chế độ: `original`, `masked`, `combined`.
* **Chia train/val theo nhóm ảnh gốc:** ảnh gốc và mọi bản khẩu trang của nó luôn cùng phía; phép chia lưu ra JSON để mọi giai đoạn dùng chung.
* Augmentation trên tập train: `RandomResizedCrop(0.8–1.0)`, `RandomHorizontalFlip`, `RandomAffine(±10°)`, `ColorJitter`, `RandomGrayscale`, **LowerFaceOcclusion** (che ngẫu nhiên nửa dưới mặt, p=0.3 ở chế độ combined), `RandomErasing`.
* Nạp sẵn ảnh đã giải mã vào RAM (`--cache-images`) để train nhanh trên Windows.

---

### PHASE 2: REAL-TIME MULTI-FACE DETECTION & ALIGNMENT
*Người phụ trách chính: Thành viên 1* — `src/face_detector.py`
* **OpenCV YuNet** phát hiện mọi khuôn mặt kèm 5 landmark (2 mắt, mũi, 2 khóe miệng); dự phòng Haar Cascade.
* **Căn chỉnh khuôn mặt** bằng similarity transform về template 5 điểm của RAF-DB aligned. Template được **đo thực nghiệm** bằng cách chạy YuNet trên ~1,000 ảnh RAF-DB test (độ lệch chuẩn ~0.03), nên ảnh crop từ webcam có cùng phân phối với ảnh lúc train.
* Detect trên frame thu nhỏ (640px) cho nhanh, căn chỉnh trên frame gốc để giữ chi tiết.

---

### PHASE 3: PRE-TRAINED BACKBONES & TRANSFER LEARNING
*Người phụ trách chính: Thành viên 2* — `src/models.py`, `src/train.py`
* **Backbone chính: HSEmotion EfficientNet-B0** (tiền huấn luyện trên khuôn mặt VGGFace2, fine-tune cảm xúc trên AffectNet). Lớp phân loại 7 lớp được khởi tạo từ classifier AffectNet (bỏ lớp *contempt*). Hỗ trợ thêm: EfficientNet-B2 (HSEmotion), MobileNetV3-Large, ResNet-18.
* **Huấn luyện 2 giai đoạn:**
  * Giai đoạn 1 (`--data-mode original`): 1 epoch chỉ train head (đóng băng backbone) → fine-tune toàn mạng 25 epochs, LR backbone $10^{-4}$ / head $10^{-3}$, AdamW, warmup tuyến tính + cosine.
  * Giai đoạn 2 (`--data-mode combined`): khởi tạo từ checkpoint giai đoạn 1, fine-tune 20 epochs trên ảnh gốc + khẩu trang.
* **Loss:** Logit-Adjusted Cross-Entropy (cộng $\tau \log \pi_c$ vào logits khi train) + label smoothing 0.1; Mixup/CutMix; EMA trọng số; AMP khi train.
* **Hiệu chỉnh prior-bias (`src/calibrate.py`):** Logit-Adjusted loss khiến mô hình dự đoán như thể 7 lớp cân bằng, trong khi RAF-DB lệch mạnh về `happy`/`neutral` → mô hình đoán thừa `fear`/`disgust`. Sau khi train, chọn $t$ **trên tập val** sao cho $\arg\max(\text{logits} + t \log \pi)$ đạt macro F1 cao nhất ($t=0.70$ cho mô hình RAF-DB, $t=0.85$ cho Mask-Aware), lưu vào checkpoint.
* Đánh giá luôn chạy fp32 (đo được: đánh giá dưới fp16 autocast làm tụt ~2 điểm accuracy).

---

### PHASE 4: ĐÁNH GIÁ TOÀN DIỆN & SO SÁNH HIỆU NĂNG
*Người phụ trách: Thành viên 2* — `src/evaluate.py`

Tất cả số liệu trên **split test chính thức của RAF-DB**, fp32, có TTA lật ngang và prior-bias (chọn trên val):

| Mô hình | Mặt thường (3,068) | Mặt đeo khẩu trang (1,975) | Tổng hợp (5,043) |
| :--- | :---: | :---: | :---: |
| HSEmotion-B0, chỉ RAF-DB (`best_model_rafdb.pth`) | **89.15%** / F1 0.833 | 56.61% / F1 0.534 | 76.40% / F1 0.725 |
| **HSEmotion-B0 Mask-Aware (Proposed ⭐)** (`best_model_mask_aware.pth`) | 88.43% / F1 0.822 | **79.29% / F1 0.708** | **84.85% / F1 0.779** |
| MobileNetV3 Mask-Aware (pipeline cũ, val bị rò rỉ) | 79.86% / F1 0.718 | 73.27% / F1 0.628 | — |

**Đóng góp của từng cải tiến** (mô hình RAF-DB, mặt thường):

| Bước | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| MobileNetV3 (pipeline cũ) | 79.86% | 0.718 |
| HSEmotion-B0, logits thô (fp32) | 84.22% | 0.764 |
| + TTA lật ngang + prior-bias $t=0.70$ | **89.15%** | **0.833** |

**Nhận xét chính:**
* Mask-Aware gần như không mất độ chính xác trên mặt thường (−0.7 điểm) nhưng tăng **+22.7 điểm** trên mặt đeo khẩu trang.
* Khi bị che miệng, mô hình chỉ train RAF-DB dồn dự đoán về `neutral` (recall 0.82, precision 0.37); Mask-Aware khắc phục phần lớn hiện tượng này.
* Lớp khó nhất vẫn là `disgust` và `fear` trên mặt đeo khẩu trang (F1 0.47 và 0.53) vì các biểu cảm này phụ thuộc nhiều vào mũi và miệng — đây là giới hạn tự nhiên của bài toán.
* Tốc độ suy luận (FPS) phụ thuộc phần cứng và số khuôn mặt; cần đo trực tiếp trên máy demo trước khi đưa vào báo cáo.

**Kiểm tra rò rỉ dữ liệu** (`scripts/check_leakage.py`, báo cáo đầy đủ ở `output/leakage_report.txt`): đối chiếu ảnh bằng nội dung (perceptual hash + tương quan pixel), không dựa vào tên file.
* 1,973/1,975 ảnh `test_masked` truy được về ảnh gốc thuộc **test**, 0 ảnh từ train.
* Train và val (17,099 / 3,002 ảnh) **không có ảnh gốc chung**.
* 49/3,068 ảnh test (1.6%) là cùng người / cùng buổi chụp với ảnh train — tính chất của split chính thức RAF-DB (không tách theo danh tính). Loại các ảnh này khỏi test chỉ làm accuracy giảm 0.05–0.19 điểm.
* Pipeline cũ chia train/val ngẫu nhiên **sau khi** trộn ảnh gốc với bản khẩu trang của nó → val F1 0.826 nhưng test F1 chỉ 0.718. Đây là lý do số liệu cũ không còn được dùng.

---

### PHASE 5: EXPLAINABLE AI VỚI GRAD-CAM
*Người phụ trách chính: Thành viên 3* — `src/gradcam.py`
* Hook vào feature map cuối cùng sau activation: `bn2` (EfficientNet), `features[-1]` (MobileNetV3), `layer4` (ResNet). Gradient được bắt bằng tensor hook để tương thích với activation in-place.
* Tạo heatmap theo lớp được dự đoán, phủ lên ảnh khuôn mặt **đã căn chỉnh**.
* Phân tích vùng kích hoạt: khóe miệng và đuôi mắt (`happy`), lông mày nhíu (`angry`), mắt mở to (`surprise`); so sánh mô hình thường với Mask-Aware trên mặt đeo khẩu trang để thấy vùng chú ý dịch lên mắt và lông mày.

---

### PHASE 6: ỨNG DỤNG REAL-TIME MULTI-FACE
*Người phụ trách chính: Thành viên 3* — `app.py`, `webcam_app.py`, `src/inference.py`
* **Streamlit app (`app.py`):** Live webcam, phân tích video tải lên (xuất video đã gán nhãn), chụp snapshot, phân tích ảnh nhóm, XAI Inspector, so sánh RAF-DB với Mask-Aware.
* **Desktop app (`webcam_app.py`):** nạp sẵn cả 2 mô hình, đổi mô hình tức thì bằng phím `1` / `2` / `M`, bật/tắt Grad-CAM bằng `G`.
* **Ổn định theo thời gian:** tracker IoU giữ ID cho từng khuôn mặt, làm mượt xác suất bằng EMA để nhãn không nhảy giữa các frame.

---

## 4. PHÂN CHIA CÔNG VIỆC TRONG NHÓM (3 THÀNH VIÊN)

| Thành viên | Vai trò chính | Trách nhiệm chuyên môn chi tiết | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Data & Detection Pipeline Engineer** | • EDA trên RAF-DB và RAF-DB Masked.<br>• Data pipeline chia train/val theo nhóm ảnh gốc, augmentation che khuất.<br>• Kiểm tra rò rỉ dữ liệu bằng nội dung ảnh.<br>• Detector YuNet + căn chỉnh khuôn mặt theo template RAF-DB. | • `dataset.py`, `face_detector.py`<br>• `scripts/check_leakage.py`, `output/leakage_report.txt`<br>• Báo cáo phân tích dữ liệu. |
| **Thành viên 2** | **Deep Learning & Transfer Learning Engineer** | • Backbone HSEmotion EfficientNet và các backbone so sánh.<br>• Huấn luyện 2 giai đoạn (RAF-DB → Mask-Aware), Logit-Adjusted loss.<br>• Hiệu chỉnh prior-bias trên val.<br>• Đánh giá Acc, Macro F1, Confusion Matrix trên 3 tập test. | • `models.py`, `train.py`, `calibrate.py`, `evaluate.py`<br>• `best_model_rafdb.pth`, `best_model_mask_aware.pth`<br>• Bảng so sánh thực nghiệm. |
| **Thành viên 3** | **XAI & Real-Time Application Engineer** | • Grad-CAM trên các backbone pre-trained.<br>• Trực quan hóa heatmap trên mặt thường và mặt đeo khẩu trang.<br>• Ứng dụng Streamlit + desktop app quét đa khuôn mặt, tracker.<br>• Slide thuyết trình và kịch bản demo. | • `gradcam.py`, `inference.py`<br>• `app.py`, `webcam_app.py`<br>• Slide, video demo, tài liệu hướng dẫn. |

---

## 5. LỘ TRÌNH THỰC HIỆN DỰ ÁN (6 TUẦN)

```text
Tuần 1: Khám phá Dữ liệu & Đề cương
  ├── Phân tích phân bố 15,339 ảnh RAF-DB, hoàn thiện Proposal
  └── Sinh RAF-DB Masked bằng MaskTheFace (giữ tiền tố tên ảnh gốc)
Tuần 2: Detection & Baseline
  ├── YuNet + căn chỉnh khuôn mặt theo template RAF-DB
  └── Baseline MobileNetV3 / ResNet-18
Tuần 3: Leak-free Pipeline & Face-Pretrained Backbone
  ├── Chia train/val theo nhóm ảnh gốc, kiểm tra rò rỉ dữ liệu
  └── Fine-tune HSEmotion EfficientNet-B0 trên RAF-DB gốc
Tuần 4: Mask-Aware & Đánh giá
  ├── Fine-tune Mask-Aware (gốc + khẩu trang), hiệu chỉnh prior-bias trên val
  └── Accuracy, Macro F1, Confusion Matrix trên Unmasked / Masked / Combined
Tuần 5: Explainable AI (Grad-CAM)
  ├── Grad-CAM trên mặt thường và mặt đeo khẩu trang
  └── So sánh vùng chú ý giữa mô hình RAF-DB và Mask-Aware
Tuần 6: Ứng dụng Real-Time & Hoàn thiện Báo cáo
  ├── Streamlit app + desktop app, đo FPS trên máy demo
  └── Quay video demo và hoàn tất báo cáo bảo vệ
```

---

## 6. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG VÀ KẾT QUẢ ĐẠT ĐƯỢC

1. **Về mặt kỹ thuật định lượng:**
   * Mục tiêu ban đầu 78–83% accuracy trên RAF-DB test → **đạt 89.15% / macro F1 0.833** (mô hình RAF-DB) và **88.43% / F1 0.822** (Mask-Aware).
   * Trên mặt đeo khẩu trang (tổng hợp): **79.29% / F1 0.708**, cao hơn mô hình không học khẩu trang 22.7 điểm.
   * Lớp thiểu số trên mặt thường (Mask-Aware): `fear` precision 0.74 / recall 0.61, `disgust` 0.69 / 0.68 — không còn hiện tượng đoán thừa sau hiệu chỉnh prior-bias.
   * Quy trình đánh giá được kiểm chứng không rò rỉ dữ liệu (`output/leakage_report.txt`).
2. **Về mặt giải thích khoa học (XAI):**
   * Grad-CAM thể hiện vùng quyết định hợp lý về mặt giải phẫu (nụ cười Duchenne, cơ cau mày, mắt mở to), và vùng chú ý dịch lên mắt/lông mày khi miệng bị che.
3. **Về sản phẩm ứng dụng:**
   * Nhận diện đồng thời nhiều khuôn mặt từ webcam/video với nhãn ổn định theo thời gian; FPS cần đo trên máy demo.
   * Click chọn bất kỳ khuôn mặt nào để xem bản đồ nhiệt Grad-CAM chi tiết.

**Hạn chế cần nêu trong báo cáo:**
* Dữ liệu khẩu trang là **tổng hợp** (MaskTheFace); chưa có tập khẩu trang thật đủ lớn để đánh giá định lượng.
* Split chính thức RAF-DB không tách theo danh tính (1.6% ảnh test trùng người với train, ảnh hưởng ≤ 0.2 điểm).
* `disgust` và `fear` khi đeo khẩu trang vẫn khó (F1 0.47 và 0.53) do phụ thuộc vào vùng mũi/miệng.
