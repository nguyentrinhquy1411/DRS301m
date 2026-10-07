# 🎭 Project Proposal & Methodology: Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Real Occlusions via Pre-Trained Deep Models

---

## 📌 THÔNG TIN TỔNG QUAN DỰ ÁN

* **Tên đề tài:** Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Real Occlusion using Pre-trained Deep Models (XAI-FER Real Multi-Face)
* **Quy mô nhóm:** 3 thành viên
* **Định hướng thực hiện:** Tuyệt đối không dùng dữ liệu vẽ/sinh nhân tạo. Hệ thống được xây dựng và kiểm định 100% trên bộ dữ liệu ảnh chụp người thật ngoài đời thực (**In-The-Wild Faces**), tận dụng sức mạnh của các mô hình học sâu SOTA tiền huấn luyện (Pre-trained Backbones / Transfer Learning), và triển khai ứng dụng thời gian thực quét nhiều khuôn mặt đồng thời (Real-Time Multi-Face Scanning).
* **Điểm khác biệt cốt lõi (Key Contribution / Unique Selling Point):** 
  1. **Dữ liệu thật 100% (Real-World Benchmark):** Sử dụng **RAF-DB (Real-world Affective Faces Database)** gồm 15,339 ảnh người thật in-the-wild (RGB $100 \times 100$, ánh sáng thực, biểu cảm tự nhiên, che khuất tự nhiên như kính mắt, tay chạm mặt, râu tóc) và **RMFD (Real-World Masked Face Dataset)** gồm ảnh người thật đeo khẩu trang thật ngoài đời.
  2. **Transfer Learning & Fine-Tuning SOTA:** Khai thác các kiến trúc thị giác máy tính hàng đầu (**MobileNetV3-Large** siêu nhẹ ~21 MB và **ResNet-18**) qua 2 chiến lược: *Feature Extraction* (đóng băng backbone) và *Deep Fine-Tuning* kết hợp Weighted Cross-Entropy Loss xử lý mất cân bằng lớp.
  3. **Quét nhiều khuôn mặt thời gian thực (Real-Time Multi-Face Pipeline):** Tích hợp bộ phát hiện đa khuôn mặt siêu nhẹ **MediaPipe BlazeFace** (< 15 ms/frame), nhận diện cảm xúc đồng thời cho $N$ người trong khung hình video/webcam.
  4. **Minh bạch hóa AI (Explainable AI - Grad-CAM):** Trích xuất bản đồ nhiệt từ Conv layer cuối cùng, giải thích rõ AI đang dựa vào vùng mắt, mũi hay khóe miệng của người thật để đưa ra quyết định; cho phép người dùng click chọn từng người trong đám đông để xem giải thích trực quan.

---

## 1. ĐẶT VẤN ĐỀ VÀ CHIẾN LƯỢC BÁO CÁO (PITCHING STRATEGY)

### 1.1. Problem Statement (Vấn đề thực tế)
1. **Hạn chế của mô hình Black-Box:** Các hệ thống nhận diện cảm xúc khuôn mặt (FER) truyền thống đưa ra nhãn phân loại mà không giải thích được lý do tại sao, gây thiếu tin cậy trong các ứng dụng giáo dục, y tế và giám sát tương tác.
2. **Khoảng cách giữa phòng thí nghiệm và đời thực (In-The-Wild Gap):** Các mô hình tự huấn luyện từ đầu trên tập ảnh xám nhỏ 48x48 trong môi trường nhân tạo thường bị sụp đổ hiệu năng khi ra ngoài đời thực — nơi khuôn mặt có màu sắc tự nhiên, góc nghiêng đa dạng, ánh sáng phức tạp và che khuất thực tế (kính mắt, khẩu trang y tế/vải, tay che miệng).
3. **Môi trường thực tế có nhiều người (Multi-Face Scenario):** Camera giám sát và ứng dụng thực tế đòi hỏi hệ thống phải phát hiện, cắt và dự đoán cảm xúc của **nhiều khuôn mặt cùng lúc trong một khung hình** với tốc độ mượt mà ($\ge 30$ FPS).

### 1.2. Giải pháp đề xuất (Proposed Solution)
1. **Nền tảng dữ liệu người thật 100%:** Sử dụng **RAF-DB** (15,339 ảnh người thật in-the-wild) làm chuẩn huấn luyện và đánh giá, bổ trợ bởi tập ảnh người thật đeo khẩu trang thật **RMFD**.
2. **Pipeline Pre-trained & Transfer Learning:** Tinh chỉnh mô hình **MobileNetV3-Large** gọn nhẹ trên tập dữ liệu người thật, tối ưu hóa tốc độ và độ chính xác.
3. **Pipeline Quét Đa Khuôn Mặt Thời Gian Thực:** Sử dụng **MediaPipe Face Detection** quét đồng thời mọi người trong khung hình, suy luận cảm xúc song song và vẽ bounding box, emoji, độ tin cậy.
4. **Giải thích quyết định qua Grad-CAM Inspector:** Click chọn từng khuôn mặt trong khung hình để mở bản đồ nhiệt giải thích trực quan.

### 1.3. Tuyên ngôn khi thuyết trình Proposal (Elevator Pitch)
> *"Instead of relying on toy datasets or artificial masks, our project delivers a robust, real-time multi-face emotion recognition system grounded in 100% real-world in-the-wild human data (RAF-DB & RMFD). By leveraging lightweight pre-trained architectures (MobileNetV3) with targeted fine-tuning and integrating MediaPipe BlazeFace detection alongside interactive Grad-CAM explainability, we deliver transparent, high-speed multi-person emotion analytics ready for real-world deployment."*

---

## 2. BỘ DỮ LIỆU THỰC TẾ & KIẾN TRÚC HỆ THỐNG

### 2.1. Thống kê Dữ liệu Thật (RAF-DB & RMFD)
* **Tập Huấn luyện RAF-DB (`data/raf_db/train`):** **12,271 ảnh người thật**
  * `happy`: 4,772 | `neutral`: 2,524 | `sad`: 1,982 | `surprise`: 1,290 | `disgust`: 717 | `angry`: 705 | `fear`: 281
* **Tập Kiểm thử RAF-DB (`data/raf_db/test`):** **3,068 ảnh người thật**
  * `happy`: 1,185 | `neutral`: 680 | `sad`: 478 | `surprise`: 329 | `angry`: 162 | `disgust`: 160 | `fear`: 74
* **Tập Che khuất Thật RMFD (`data/real_masked_faces`):** 50 ảnh người thật đeo khẩu trang thật ngoài đời.

```text
               Video / Webcam Feed (Nhiều người trong khung hình)
                                      │
                                      ▼
               Phase 1: Multi-Face Detection & Tracking
                      (MediaPipe BlazeFace Detector)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
            Face Crop #1                              Face Crop #N
         (Real Human Face)                         (Real Human Face)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                 Phase 2: RGB Preprocessing & Normalization
                    (Resize 224x224, ImageNet Normalization)
                                      │
                                      ▼
                 Phase 3 & 4: Pre-trained Backbone Strategy
         ┌─────────────────────────────────────────────────────────┐
         │ • Strategy A: Feature Extraction (Backbone Frozen)      │
         │ • Strategy B: Deep Fine-Tuning (Top Layers Unfrozen)    │
         │   (MobileNetV3-Large ~21MB / ResNet-18 ~44MB)          │
         │ • Loss: Weighted Cross-Entropy (Xử lý Imbalance)        │
         └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                 Phase 5: Multi-Metric Evaluation & Testing
           (Accuracy, Macro F1, Confusion Matrix trên RAF-DB Test & RMFD)
                                      │
                                      ▼
                 Phase 6: Explainable AI (Grad-CAM Analysis)
             (Visualizing Neural Attention Maps on Real Faces)
                                      │
                                      ▼
                 Phase 7: Real-Time Multi-Face Web Application
        (Streamlit Live Dashboard: Multi-Face BBoxes + Interactive XAI)
```

---

## 3. CHI TIẾT CÁC GIAI ĐOẠN THỰC HIỆN

### PHASE 1: REAL-WORLD DATA PIPELINE & PREPROCESSING
*Người phụ trách chính: Thành viên 1*
* Nạp tập dữ liệu người thật RAF-DB (12,271 train / 3,068 test).
* Xây dựng Custom Dataset Loader hỗ trợ ảnh màu RGB, resize $224 \times 224$, chuẩn hóa theo ImageNet Mean/Std.
* Áp dụng Data Augmentation trên tập Train: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, `ColorJitter(brightness=0.2, contrast=0.2)`.

---

### PHASE 2: REAL-TIME MULTI-FACE DETECTION MODULE
*Người phụ trách chính: Thành viên 1*
* Tích hợp **MediaPipe Face Detection** (BlazeFace).
* Trích xuất đồng thời danh sách bounding boxes $[x, y, w, h]$ của tất cả khuôn mặt trong khung hình với độ trễ $< 15$ ms.
* Cắt (crop) khuôn mặt kèm padding tự nhiên, chuẩn bị cho batch inference.

---

### PHASE 3: PRE-TRAINED BACKBONES & TRANSFER LEARNING
*Người phụ trách chính: Thành viên 2*
* Tận dụng backbone **MobileNetV3-Large** (~21 MB) và **ResNet-18** (~44 MB) đã huấn luyện trên hàng triệu ảnh.
* Gắn Classifier Head mới: `AdaptiveAvgPool2d` $\rightarrow$ `Linear(D, 256)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(256, 7)`.
* Thực hiện 2 chiến lược:
  * **Feature Extraction:** Đóng băng backbone, chỉ train head trong 3 epochs với lr = $10^{-3}$.
  * **Deep Fine-Tuning:** Mở khóa toàn bộ mô hình, train với lr = $10^{-4}$ kết hợp `CosineAnnealingLR` và `Weighted Cross-Entropy Loss`.

---

### PHASE 4: ĐÁNH GIÁ TOÀN DIỆN & SO SÁNH HIỆU NĂNG
*Người phụ trách: Thành viên 2*

| Mô hình / Chiến lược | Cấu hình huấn luyện | Accuracy (RAF-DB Test) | Macro F1-Score | Tốc độ suy luận (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline CNN** | Train from scratch | ~62% | ~0.56 | > 60 FPS |
| **MobileNetV3** | Feature Extraction (Frozen) | ~72% | ~0.68 | ~50 FPS |
| **MobileNetV3 (Proposed ⭐)** | Deep Fine-Tuned + Weighted Loss | **78 - 83%** | **0.75 - 0.80** | **~45 FPS** |

---

### PHASE 5: EXPLAINABLE AI VỚI GRAD-CAM
*Người phụ trách chính: Thành viên 3*
* Hook vào lớp Conv cuối cùng của MobileNetV3 (`features[-1]`) hoặc ResNet (`layer4[-1]`).
* Trích xuất activation map và gradient tương ứng với nhãn cảm xúc dự đoán, tạo Heatmap (Jet color map) phủ lên ảnh khuôn mặt người thật.
* Phân tích các vùng kích hoạt sinh học: khóe cười nụ cười Duchenne (`Happy`), nhăn mày sống mũi (`Angry`), mắt to há miệng (`Surprise`).

---

### PHASE 6: ĐÓNG GÓI ỨNG DỤNG WEB REAL-TIME MULTI-FACE (STREAMLIT)
*Người phụ trách chính: Thành viên 3*
* **Chế độ 1 - Real-Time Multi-Face Webcam:** Quét camera trực tiếp, vẽ bounding box + emoji + cảm xúc + % tin cậy quanh từng người trong khung hình.
* **Chế độ 2 - Phân tích Ảnh/Video Nhóm:** Tải ảnh chụp tập thể nhiều người hoặc ảnh người thật đeo khẩu trang RMFD, nhận diện và thống kê biểu cảm cả nhóm.
* **Chế độ 3 - Kính lúp Grad-CAM Inspector:** Click chọn bất kỳ khuôn mặt nào trên giao diện để phóng to và soi bản đồ nhiệt Grad-CAM giải thích AI.

---

## 4. PHÂN CHIA CÔNG VIỆC TRONG NHÓM (3 THÀNH VIÊN)

| Thành viên | Vai trò chính | Trách nhiệm chuyên môn chi tiết | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Data & Detection Pipeline Engineer** | • Thực hiện EDA trên bộ dữ liệu người thật RAF-DB.<br>• Xây dựng Custom Dataset Loader RGB 224x224.<br>• Tích hợp MediaPipe Face Detector quét nhiều khuôn mặt.<br>• Tính toán phân bố nhãn và trọng số Class Weights. | • Module dữ liệu (`dataset.py`)<br>• Module phát hiện đa khuôn mặt (`face_detector.py`)<br>• Báo cáo phân tích dữ liệu người thật in-the-wild. |
| **Thành viên 2** | **Deep Learning & Transfer Learning Engineer** | • Thiết lập MobileNetV3-Large và ResNet-18 với custom head.<br>• Pipeline huấn luyện 2 pha (Warmup Head $\to$ Fine-Tuning).<br>• Cài đặt Weighted Cross-Entropy Loss xử lý mất cân bằng lớp.<br>• Đánh giá định lượng trên RAF-DB test (Acc, Macro F1, Confusion Matrix). | • Module mô hình (`models.py`)<br>• Script huấn luyện (`train.py`)<br>• Checkpoint tối ưu (`best_model_rafdb.pth`)<br>• Báo cáo so sánh thực nghiệm các mô hình. |
| **Thành viên 3** | **XAI & Real-Time Application Engineer** | • Hiện thực hóa Grad-CAM trên kiến trúc Pre-trained.<br>• Trực quan hóa bản đồ nhiệt trên ảnh người thật và người đeo khẩu trang.<br>• Phát triển ứng dụng Streamlit quét webcam đa khuôn mặt thời gian thực.<br>• Xây dựng tính năng click chọn khuôn mặt xem Grad-CAM tương tác.<br>• Soạn slide thuyết trình và chuẩn bị kịch bản demo. | • Module giải thích XAI (`gradcam.py`)<br>• Ứng dụng Web hoàn chỉnh (`app.py`)<br>• Giao diện webcam quét đa khuôn mặt trực tiếp.<br>• Slide thuyết trình, video demo và tài liệu hướng dẫn. |

---

## 5. LỘ TRÌNH THỰC HIỆN DỰ ÁN (6 TUẦN)

```text
Tuần 1: Khám phá Dữ liệu Thật RAF-DB & Đề cương
  ├── Phân tích phân bố 15,339 ảnh RAF-DB, hoàn thiện tài liệu Proposal
  └── Nạp tập ảnh người thật đeo khẩu trang thật RMFD
Tuần 2: Pre-trained Backbone & Multi-Face Detection
  ├── Tích hợp MediaPipe Face Detector quét nhiều người đồng thời
  └── Thiết lập MobileNetV3-Large và ResNet-18 với custom head
Tuần 3: Two-Stage Transfer Learning Pipeline
  ├── Thực nghiệm Warmup Classifier Head (đóng băng backbone)
  └── Thực nghiệm Deep Fine-Tuning với Weighted Loss trên tập train RAF-DB
Tuần 4: Đánh giá Toàn diện & Ma trận Nhầm lẫn
  ├── Đo lường Accuracy, Macro F1, Recall từng lớp trên RAF-DB test
  └── Kiểm thử độ bền vững trên tập ảnh che khuất thật RMFD
Tuần 5: Explainable AI (Grad-CAM Inspector)
  ├── Trích xuất Grad-CAM từ mô hình Fine-tuned trên ảnh người thật
  └── Phân tích trực quan các vùng giải phẫu cơ mặt biểu cảm sinh học
Tuần 6: Web App Quét Đa Khuôn Mặt Thời Gian Thực & Hoàn thiện Báo cáo
  ├── Đóng gói ứng dụng Streamlit Real-Time Multi-Face Scanning
  └── Thử nghiệm thực tế luồng webcam, quay video demo và hoàn tất báo cáo bảo vệ
```

---

## 6. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG VÀ KẾT QUẢ KỲ VỌNG

1. **Về mặt kỹ thuật định lượng (Quantitative Metrics):**
   * Mô hình Fine-Tuned MobileNetV3 đạt Accuracy từ **78 - 83%** trên tập kiểm thử người thật RAF-DB test.
   * Macro F1-Score đạt $\ge 0.75$ nhờ tối ưu hóa Weighted Loss, cải thiện rõ rệt trên các cảm xúc thiểu số (*Disgust, Fear*).
2. **Về mặt giải thích khoa học (Qualitative & XAI Insights):**
   * Grad-CAM thể hiện rõ ràng các vùng quyết định hợp lý về mặt giải phẫu sinh học trên người thật (nụ cười Duchenne, cơ cau mày, mắt mở to).
3. **Về sản phẩm ứng dụng (Real-Time Multi-Face Product):**
   * Hệ thống quét và nhận diện đồng thời từ $2 - 5$ khuôn mặt trong khung hình webcam mượt mà ($\ge 25 - 40$ FPS).
   * Giao diện trực quan, cho phép người dùng click chọn bất kỳ khuôn mặt nào để xem bản đồ nhiệt Grad-CAM chi tiết.
