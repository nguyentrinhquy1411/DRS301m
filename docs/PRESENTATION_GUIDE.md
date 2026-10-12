# 🎭 KỊCH BẢN VÀ HƯỚNG DẪN THUYẾT TRÌNH CHI TIẾT (PRESENTATION GUIDE)
## Đề tài: Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Occlusions via Pre-Trained Deep Models (XAI-FER Mask-Aware)

---

## 📌 1. TỔNG QUAN PHÂN CHIA VAI DIỄN & THỜI LƯỢNG

* **Tổng thời lượng:** 18 – 20 phút (Trình bày: 14 – 15 phút | Live Demo: 3 – 4 phút | Q&A: 5 – 10 phút).
* **Quy mô nhóm:** 3 thành viên (Speaker 1, Speaker 2, Speaker 3).
* **Nguyên tắc phối hợp:** Chuyển giao mạch lạc (handover cues), mỗi người phụ trách trọn vẹn một mảng kỹ thuật cốt lõi, không giẫm chân lên nhau.

| Thành viên | Phân vai chuyên môn | Khối lượng nội dung | Slide | Thời gian |
| :--- | :--- | :--- | :---: | :---: |
| **Speaker 1** | **Data & Detection Pipeline Engineer** | • Đặt vấn đề & Thách thức In-The-Wild<br>• Dữ liệu RAF-DB & Pipeline MaskTheFace<br>• Giải quyết triệt để rò rỉ dữ liệu (Leak-free Audit)<br>• YuNet Detection & Căn chỉnh hình học 5 landmark | Slide 1 – 5 | **Phút 00:00 – 05:00** (5 phút) |
| **Speaker 2** | **Deep Learning & Evaluation Engineer** | • Face-Pretrained Backbone (HSEmotion EfficientNet-B0)<br>• Chiến lược huấn luyện 2 giai đoạn & Logit Adjustment<br>• Hiệu chỉnh Prior-Bias hậu huấn luyện (`calibrate.py`)<br>• Benchmark định lượng toàn diện (+22.7% Masked Acc)<br>• Confusion Matrix & Ablation Study | Slide 6 – 11 | **Phút 05:00 – 10:30** (5.5 phút) |
| **Speaker 3** | **XAI & Real-Time System Engineer** | • Giải phẫu Grad-CAM & Attention Shift khi đeo khẩu trang<br>• Multi-Face Tracker IoU + EMA Smoothing chống flicker<br>• Kỹ thuật tối ưu Real-time FPS (55 FPS, Grad-CAM cache)<br>• Hệ sinh thái ứng dụng Streamlit & Desktop App<br>• **Trực tiếp điều khiển Live Demo** & Kết luận | Slide 12 – 17 | **Phút 10:30 – 15:30** (5 phút) + Demo |

---

## 🎯 2. ELEVATOR PITCH (LỜI MỞ ĐẦU ĐINH CAO - 30 GIÂY)

> *"Kính thưa quý thầy cô và hội đồng, biểu cảm khuôn mặt là cầu nối cảm xúc quan trọng nhất của con người. Nhưng điều gì sẽ xảy ra khi một nửa khuôn mặt bị che khuất bởi chiếc khẩu trang, giữa môi trường thực tế đông người và nhiễu loạn? Hệ thống của chúng em không chỉ giải quyết triệt để bài toán nhận diện cảm xúc đa khuôn mặt thời gian thực khi có khẩu trang với độ chính xác 88.4% trên mặt thường và 79.3% trên mặt đeo khẩu trang (vượt trội 22.7% so với mô hình thông thường), mà còn mở chiếc hộp đen của AI để giải thích trực quan: Mô hình đang nhìn vào đâu để đưa ra phán đoán đó."*

---

## 🎬 3. KỊCH BẢN CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE SCRIPT)

---

### 🎙️ PHẦN 1: SPEAKER 1 — DATA PIPELINE, AUDIT & DETECTION (00:00 – 05:00)

#### Slide 1: Trang tiêu đề & Giới thiệu thành viên
* **Visuals:** Logo trường, tên đề tài, ảnh đại diện và vai trò của 3 thành viên, icon biểu cảm 🎭.
* **Speaker 1 nói:**
  > *"Kính thưa quý thầy cô trong hội đồng và các bạn sinh viên, em là [Tên Speaker 1], đại diện cho nhóm thực hiện đề tài: **Hệ thống Nhận diện Cảm xúc Đa khuôn mặt Thời gian thực Kết hợp Giải thích AI trong Điều kiện Che khuất (XAI-FER Mask-Aware)**. Đồng hành cùng em hôm nay có bạn [Tên Speaker 2] - phụ trách Mô hình hóa Deep Learning & Đánh giá định lượng, và bạn [Tên Speaker 3] - phụ trách Explainable AI & Kiến trúc ứng dụng thời gian thực. Sau đây, em xin phép bắt đầu phần trình bày của nhóm."*

---

#### Slide 2: Đặt vấn đề & Bối cảnh thực tế (Problem Statement & Motivation)
* **Visuals:** 
  * Biểu đồ so sánh "Lab FER" (ảnh xám 48x48 đơn sắc FER-2013) vs "In-The-Wild FER" (ảnh màu góc nghiêng, ánh sáng phức tạp, nhiều người).
  * Hình ảnh minh họa khuôn mặt đeo khẩu trang: Nửa dưới bị che khuất hoàn toàn.
  * Hộp đen AI (Black-box) đặt dấu hỏi chấm về sự tin cậy.
* **Speaker 1 nói:**
  > *"Trong kỷ nguyên chuyển đổi số, nhận diện cảm xúc khuôn mặt (FER) đang đóng vai trò then chốt trong y tế thông minh, giáo dục trực tuyến và tương tác người - máy. Tuy nhiên, khi chuyển từ phòng thí nghiệm ra môi trường thực tế, các hệ thống FER truyền thống gặp phải 3 rào cản chí mạng:
  > 
  > 1. **Khoảng cách phòng lab và đời thực (In-the-wild gap):** Các mô hình cũ thường train trên tập FER-2013 với ảnh xám nhỏ 48x48 được căn chỉnh nhân tạo, nên sụp đổ hoàn toàn trước ảnh đời thực có góc nghiêng, biểu cảm tự nhiên và ánh sáng phức tạp.
  > 2. **Sự che khuất do khẩu trang (Mask Occlusion):** Khẩu trang y tế che mất toàn bộ nửa dưới khuôn mặt — nơi chứa khóe miệng, môi và nụ cười. Thực nghiệm cho thấy, mô hình thông thường bị 'mù' trước khẩu trang: recall của cảm xúc Vui vẻ (`happy`) tụt từ 94% xuống còn 44%, và mô hình dồn 82% dự đoán sai về nhãn Bình thường (`neutral`).
  > 3. **Vấn đề hộp đen (Black-box AI):** Dự đoán cảm xúc mà không có bằng chứng thị giác khiến con người không thể tin tưởng, đặc biệt trong các ứng dụng chẩn đoán tâm lý.
  > 
  > Mục tiêu của nhóm em là xây dựng một hệ thống **In-the-wild, Mask-Aware, Real-Time và Explainable** giải quyết triệt để cả 3 vấn đề trên."*

---

#### Slide 3: Dữ liệu In-The-Wild & Pipeline sinh khẩu trang (Dataset & MaskTheFace)
* **Visuals:** 
  * Thống kê RAF-DB: 15,339 ảnh màu phân bố trên 7 nhãn cảm xúc (`happy`, `neutral`, `sad`, `surprise`, `angry`, `disgust`, `fear`).
  * Minh họa pipeline MaskTheFace: Ảnh gốc aligned $\to$ nhận diện landmark mũi/cằm $\to$ áp khẩu trang 3D $\to$ giữ nguyên nhãn và ID ảnh nguồn.
* **Speaker 1 nói:**
  > *"Để giải quyết rào cản dữ liệu, nhóm chọn **RAF-DB (Real-world Affective Faces Database)** — bộ dữ liệu chuẩn mực gồm 15,339 ảnh khuôn mặt in-the-wild được gán nhãn bởi 40 chuyên gia độc lập. Trong đó: 12,271 ảnh Train và 3,068 ảnh Test chính thức.
  > 
  > Vì trong tự nhiên chưa có tập dữ liệu cảm xúc đeo khẩu trang thật nào đủ lớn và có nhãn chuẩn xác, nhóm đã xây dựng pipeline sinh khẩu trang tự động bằng công cụ **MaskTheFace**:
  > - Dựa trên cấu trúc hình học của từng khuôn mặt, hệ thống tự động đo góc nghiêng, vị trí sống mũi và quai hàm để gắn các loại khẩu trang y tế tổng hợp với tỷ lệ che phủ sát với thực tế nhất.
  > - Tập huấn luyện được bổ sung **7,830 ảnh train_masked** và tập kiểm thử độc lập có **1,975 ảnh test_masked**.
  > - Điểm mấu chốt: Mỗi ảnh khẩu trang sinh ra đều giữ nguyên tiền tố tên ảnh gốc (ví dụ `raf_train_00012_surgical.jpg`) để phục vụ việc kiểm soát rò rỉ dữ liệu mà em sẽ trình bày ngay sau đây."*

---

#### Slide 4: Khủng hoảng rò rỉ dữ liệu & Kiểm toán độc lập (Leak-Free Audit)
* **Visuals:** 
  * Sơ đồ đỏ: Rò rỉ dữ liệu của pipeline cũ (Random Split 85/15 trộn cả gốc lẫn mask $\to$ cùng một người xuất hiện ở cả train và val $\to$ F1 ảo 0.826).
  * Sơ đồ xanh: Subject-Aware Split của nhóm (Grouped Split theo source_group $\to$ 0% leakage).
  * Bảng kết quả audit từ script `scripts/check_leakage.py` (`output/leakage_report.txt`).
* **Speaker 1 nói:**
  > *"Đây là đóng góp phương pháp luận quan trọng hàng đầu của nhóm. Khi khảo sát các mã nguồn FER kết hợp khẩu trang hiện nay, nhóm phát hiện một lỗ hổng nghiêm trọng: **Rò rỉ dữ liệu (Data Leakage)**.
  > - Nếu gộp ảnh gốc và ảnh khẩu trang lại rồi chia ngẫu nhiên 85/15, thì phiên bản không khẩu trang của người A nằm ở tập Train, còn bản có khẩu trang của chính người A lại lọt sang tập Validation! Hậu quả là mô hình 'học vẹt' khuôn mặt của người đó, đẩy validation F1 lên ảo tới 0.826, nhưng khi test trên người lạ thì sụp đổ.
  > - Nhóm đã giải quyết triệt để bằng **Subject-Aware Grouped Split**: Toàn bộ biến thể của một người phải nằm 100% ở Train hoặc 100% ở Val, phân tầng theo nhãn và cố định bằng JSON (`val_groups_r15_s42.json`).
  > - Chưa dừng lại ở đó, nhóm viết riêng công cụ **`scripts/check_leakage.py`** dùng giải thuật băm nhận thức **dHash** và đo tương quan điểm ảnh nửa trên khuôn mặt:
  >   - Kết quả xác nhận: **0 ảnh test_masked bị lọt từ train**, và **0 nhóm chung giữa Train và Val**.
  >   - Nghiên cứu của nhóm hoàn toàn minh bạch và nói không với số liệu ảo."*

---

#### Slide 5: Phát hiện đa khuôn mặt & Căn chỉnh hình học 5 Landmark (YuNet Alignment)
* **Visuals:**
  * So sánh: Crop hình chữ nhật thông thường (Haar Cascade / Bounding box) vs Căn chỉnh hình học tương đồng (Similarity Transform).
  * Template tọa độ 5 điểm chuẩn của RAF-DB: 2 mắt, mũi, 2 khóe miệng.
  * Tốc độ và độ chính xác của OpenCV YuNet.
* **Speaker 1 nói:**
  > *"Để đưa mô hình vào ứng dụng thực tế, bài toán phát hiện khuôn mặt là tiền đề quyết định:
  > - Thay vì dùng Haar Cascade cũ kỹ hay MediaPipe nặng nề, nhóm tích hợp **OpenCV YuNet** — mạng nơ-ron tích chập siêu nhẹ đạt tốc độ hơn 100 FPS và phát hiện chính xác ngay cả khi mặt đeo khẩu trang hoặc nghiêng góc lớn.
  > - Đặc biệt, nhóm áp dụng **Face Alignment (Căn chỉnh khuôn mặt)**: YuNet trích xuất 5 điểm mốc (2 mắt, đỉnh mũi, 2 khóe miệng). Nhóm dùng phép biến đổi tương đồng (Similarity Transform) để xoay, co dãn và dịch chuyển 5 điểm này khớp hoàn toàn vào **Template tọa độ chuẩn của RAF-DB** (được nhóm đo thực nghiệm trên 1,000 ảnh test với độ lệch chuẩn cực nhỏ ~0.03).
  > - Nhờ vậy, ảnh crop từ webcam thời gian thực có cùng phân phối không gian với ảnh lúc huấn luyện, loại bỏ hoàn toàn độ lệch phân phối hình học.
  > 
  > Tiếp theo, xin mời bạn [Tên Speaker 2] trình bày về kiến trúc mô hình học sâu, chiến lược huấn luyện và kết quả thực nghiệm đột phá của nhóm."*

---

### 🎙️ PHẦN 2: SPEAKER 2 — DEEP LEARNING, OPTIMIZATION & BENCHMARK (05:00 – 10:30)

#### Slide 6: Kiến trúc mô hình & Backbone Tiền huấn luyện trên Khuôn mặt (Model Architecture)
* **Visuals:**
  * Sơ đồ mạng: EfficientNet-B0 (MBConv blocks + Squeeze-and-Excitation).
  * So sánh Pretrained Domain: ImageNet (nhận diện chó, mèo, ô tô) vs HSEmotion (VGGFace2 nhận diện danh tính người $\to$ AffectNet nhận diện cảm xúc).
  * Classifier Head: 7 output classes.
* **Speaker 2 nói:**
  > *"Em xin cảm ơn bạn [Tên Speaker 1]. Kính thưa hội đồng, khi bước vào bài toán Deep Learning, câu hỏi đầu tiên nhóm đặt ra là: **Nên chọn Backbone nào?**
  > - Các nghiên cứu thông thường tải pre-trained weights từ ImageNet. Nhưng ImageNet được huấn luyện để phân biệt ô tô, chó, mèo — hoàn toàn không có nhận thức về các đơn vị vận động cơ mặt (Action Units).
  > - Nhóm quyết định lựa chọn **HSEmotion EfficientNet-B0**: Một mô hình đã trải qua 2 tầng tiền huấn luyện chuyên sâu trong miền khuôn mặt: đầu tiên trên **VGGFace2** (hàng triệu khuôn mặt người) và sau đó fine-tune trên **AffectNet** (bộ dữ liệu cảm xúc lớn nhất thế giới).
  > - EfficientNet-B0 chỉ có xấp xỉ 4 triệu tham số nhưng tận dụng kiến trúc Depthwise Separable Convolution và khối Squeeze-and-Excitation, cho khả năng trích xuất đặc trưng vùng mắt, chân mày và cơ trán vượt trội hơn hẳn các backbone nặng như ResNet-50 hay MobileNetV3."*

---

#### Slide 7: Chiến lược huấn luyện 2 giai đoạn & Kỹ thuật tối ưu hóa (2-Stage Training)
* **Visuals:**
  * Quy trình 2 Phase: 
    * Phase 1: Train RAF-DB gốc (Warmup 1 epoch head $\to$ Fine-tune 25 epochs toàn mạng).
    * Phase 2: Khởi tạo từ Phase 1 $\to$ Fine-tune 20 epochs trên Combined Dataset (Gốc + Masked).
  * Regularization: Mixup ($\alpha=0.2$), CutMix ($\alpha=1.0$), LowerFaceOcclusion ($p=0.3$), Label Smoothing 0.1, AdamW, EMA decay 0.998, PyTorch AMP.
* **Speaker 2 nói:**
  > *"Để mô hình vừa nhận diện xuất sắc mặt thường, vừa không bị suy giảm khi gặp khẩu trang, nhóm thiết kế **Chiến lược huấn luyện 2 giai đoạn (Two-Stage Transfer Learning)**:
  > - **Giai đoạn 1 (RAF-DB Baseline):** Huấn luyện trên 12,271 ảnh gốc. 1 epoch đầu đóng băng backbone để ổn định classifier head với LR $10^{-3}$, sau đó mở toàn mạng fine-tune với LR $10^{-4}$, Cosine Annealing scheduler.
  > - **Giai đoạn 2 (Mask-Aware Adaptation):** Khởi tạo trực tiếp từ trọng số tốt nhất của Giai đoạn 1, fine-tune trên tập dữ liệu tổng hợp (Combined: gốc + khẩu trang).
  > - Trong quá trình train, nhóm áp dụng tổ hợp tăng cường dữ liệu mạnh mẽ: **LowerFaceOcclusion** (chủ động che ngẫu nhiên nửa dưới khuôn mặt với xác suất 30% để ép mạng học vùng mắt), kết hợp **Mixup, CutMix, Random Erasing** và **Exponential Moving Average (EMA)** với hệ số suy giảm 0.998 để làm mượt trọng số, giúp mô hình hội tụ cực kỳ ổn định và chống overfitting tuyệt đối."*

---

#### Slide 8: Kỹ thuật Hiệu chỉnh Prior-Bias hậu huấn luyện (Post-Training Prior-Bias Calibration)
* **Visuals:**
  * Công thức Logit-Adjusted Loss: $\mathcal{L} = -\log \frac{e^{f_y(x) + \tau \log \pi_y}}{\sum_c e^{f_c(x) + \tau \log \pi_c}}$.
  * Đồ thị quét tham số $t$ từ $-0.5$ đến $+1.5$ trên tập Validation $\to$ điểm cực đại tại $t = +0.85$.
  * Minh họa hiệu ứng: Trước hiệu chỉnh (đoán thừa Fear/Disgust do recall cao nhưng precision thấp) $\to$ Sau hiệu chỉnh (cân bằng hoàn hảo).
* **Speaker 2 nói:**
  > *"Một bài toán hóc búa của dữ liệu thực tế là **Sự mất cân bằng lớp cực lớn**: Trong RAF-DB, lớp `happy` có tới gần 5,000 ảnh nhưng `fear` chỉ có 281 ảnh.
  > - Nhóm sử dụng **Logit-Adjusted Loss** trong lúc train để ép mô hình mở rộng biên quyết định cho các lớp thiểu số.
  > - Tuy nhiên, cơ chế này khiến logits thô dự đoán như thể 7 lớp có xác suất đồng đều, dẫn đến hiện tượng: trên tập test thực tế (vốn nhiều người cười và bình thản), mô hình bị đoán thừa `fear` và `disgust` — recall cao nhưng precision lại thấp.
  > - Để giải quyết, nhóm xây dựng công cụ **`src/calibrate.py`**:
  >   - Quét hệ số hiệu chỉnh $t$ trong khoảng $[-0.5, 1.5]$ để tìm nghiệm: $\arg\max (\text{logits} + t \log \pi)$ đạt Macro F1 cao nhất **chỉ tính trên tập Validation**.
  >   - Thuật toán tự động tìm ra điểm tối ưu tại **$t = +0.85$** cho mô hình Mask-Aware.
  >   - Kết quả: F1 trên tập Val tăng từ 0.7109 lên **0.7721**, Accuracy tăng từ 79.55% lên **84.64%**. Vector bias này được ghi trực tiếp vào checkpoint để áp dụng lúc suy luận mà không cần train lại một giây nào!"*

---

#### Slide 9: Kết quả thực nghiệm định lượng toàn diện (Benchmark Results)
* **Visuals:** Bảng vàng so sánh hiệu năng cốt lõi (Highlight con số **+22.68%** trên ảnh khẩu trang).

| Mô hình | Mặt thường (Unmasked) | Mặt có khẩu trang (Masked) | Toàn bộ tập test (Combined) |
| :--- | :---: | :---: | :---: |
| **MobileNetV3 (Pipeline cũ)** | 79.86% / F1 0.718 | 73.27% / F1 0.628 | 77.28% / F1 0.684 |
| **HSEmotion-B0 (Chỉ RAF-DB)** | **89.15%** / F1 0.833 | 56.61% / F1 0.534 | 76.40% / F1 0.725 |
| **HSEmotion-B0 (Mask-Aware ⭐)** | **88.43%** / F1 0.822 | **79.29%** / F1 0.708 | **84.85%** / F1 0.779 |
| **Mức chênh lệch cốt lõi** | *(Giữ nguyên, chênh <0.7%)* | **+22.68% Acc, +0.174 F1 🚀** | **+8.45% Acc, +0.054 F1 🚀** |

* **Speaker 2 nói:**
  > *"Và đây là slide quan trọng nhất chứng minh tính hiệu quả vượt bậc của đề tài. Mọi con số đều đo ở chế độ fp32 trên toàn bộ 5,043 ảnh test độc lập có TTA và Prior-bias:
  > 
  > 1. **Trên mặt thường (Unmasked 3,068 ảnh):** Mô hình Mask-Aware đạt **88.43% Accuracy và 0.822 Macro F1** — tương đương với mô hình chuyên biệt chỉ học mặt thường (89.15%). Điều này chứng minh mô hình **không hề bị suy giảm khả năng nhận diện gốc** khi học thêm khẩu trang!
  > 2. **Trên mặt đeo khẩu trang (Masked 1,975 ảnh):** Đây là bước nhảy vọt! Nếu dùng mô hình thông thường mang ra dự đoán mặt đeo khẩu trang, accuracy chỉ đạt **56.61%**. Nhưng với mô hình Mask-Aware của nhóm em, độ chính xác vọt lên **79.29%** — tức là **tăng tới 22.68 điểm phần trăm**, Macro F1 tăng mạnh từ 0.534 lên **0.708**!
  > 3. **Trên toàn bộ 5,043 ảnh kiểm thử (Combined):** Độ chính xác chung đạt **84.85%** và Macro F1 đạt **0.779**. So với baseline MobileNetV3 cũ của nhóm ở giai đoạn đầu đề tài (77.28%), hệ thống mới vượt trội hơn tới 7.5% trên mọi phương diện."*

---

#### Slide 10: Phân tích Ma trận nhầm lẫn & Từng cảm xúc (Confusion Matrix Analysis)
* **Visuals:** 
  * 3 Ma trận nhầm lẫn trực quan: Unmasked, Masked và Combined.
  * Bảng F1 chi tiết 7 cảm xúc trên tập đeo khẩu trang (`happy` 0.89, `surprise` 0.80, `sad` 0.79, `angry` 0.75, `neutral` 0.72, `fear` 0.53, `disgust` 0.47).
* **Speaker 2 nói:**
  > *"Đi sâu vào báo cáo phân loại của 1,975 ảnh có khẩu trang:
  > - **Happy đạt F1 0.8901 và Precision 94.1%**: Cho thấy dù bị che mất nụ cười ở miệng, mô hình vẫn nhận biết tuyệt đối thông qua ánh mắt cười (Duchenne marker).
  > - **Surprise và Sad đạt F1 xấp xỉ 0.80**: Mắt mở to hoặc đuôi mày chùng xuống được mô hình bắt trọn vẹn.
  > - **Angry đạt F1 0.7500**: Nhận diện rất tốt vùng cơ trán nhăn và chân mày hạ thấp.
  > - **Thử thách nằm ở Disgust (F1 0.47) và Fear (F1 0.53)**: Đây là điều hoàn toàn hợp lý về mặt sinh lý học thần kinh, bởi vì biểu cảm Ghê tởm biểu hiện mạnh nhất ở cơ nhăn cánh mũi và khóe miệng — vốn bị khẩu trang che mất. Tuy nhiên, mức F1 0.47 đã là bước tiến vượt bậc so với con số 0.33 của mô hình không có Mask-Aware."*

---

#### Slide 11: Bóc tách giá trị cải tiến (Ablation Study)
* **Visuals:** Biểu đồ thác nước (Waterfall chart) hoặc bảng tích lũy từng tính năng đóng góp vào sự tăng trưởng accuracy từ 79.8% lên 89.15% (trên RAF-DB).
* **Speaker 2 nói:**
  > *"Để hội đồng thấy rõ giá trị của từng quyết định thiết kế kỹ thuật, nhóm thực hiện nghiên cứu bóc tách (Ablation Study):
  > - Xuất phát từ MobileNetV3 cũ: Accuracy chỉ đạt 79.86%.
  > - Thay bằng Face-Pretrained HSEmotion EfficientNet-B0: Accuracy nhảy vọt lên 84.22% (+4.36%).
  > - Bổ sung YuNet Căn chỉnh 5 landmark và TTA lật ngang: Tăng thêm +2.1%.
  > - Áp dụng Logit-Adjusted Loss và Calibrate Prior-bias $t=0.70$: Đưa accuracy chạm mốc kỷ lục **89.15%** (+2.83% nữa).
  > 
  > Từng cải tiến đều có cơ sở khoa học và đóng góp định lượng rõ rệt. Tiếp theo, em xin nhường phần trình bày cho bạn [Tên Speaker 3] để giải mã vùng chú ý của AI và trình diễn ứng dụng thực tế."*

---

### 🎙️ PHẦN 3: SPEAKER 3 — XAI GRAD-CAM, REAL-TIME SYSTEM & LIVE DEMO (10:30 – 15:30)

#### Slide 12: Trực quan hóa giải thích AI & Hiện tượng Dịch chuyển chú ý (XAI & Attention Shift)
* **Visuals:** 
  * Hình ảnh Grad-CAM side-by-side: Cùng một người khi không đeo khẩu trang vs khi đeo khẩu trang.
  * Heatmap trên mặt thường: Điểm đỏ tập trung ở khóe miệng cười (`happy`) hoặc trán (`angry`).
  * Heatmap trên mặt đeo khẩu trang: Vùng đỏ **dịch chuyển rõ rệt** từ miệng lên vùng hốc mắt và vòm chân mày!
* **Speaker 3 nói:**
  > *"Em xin cảm ơn bạn [Tên Speaker 2]. Kính thưa hội đồng, một mô hình đạt 88% hay 79% sẽ vô nghĩa nếu chúng ta không biết nó dựa vào đâu để ra quyết định.
  > - Nhóm tích hợp kỹ thuật **Grad-CAM (Gradient-weighted Class Activation Mapping)**, can thiệp vào tầng tích chập cuối cùng `bn2` của EfficientNet-B0 trước khi vào lớp Pooling. Bằng cách tính đạo hàm riêng của score dự đoán đối với các feature maps, hệ thống tạo ra bản đồ nhiệt kích hoạt phản ánh chính xác vùng giải phẫu mà AI đang tập trung.
  > - **Khám phá thị giác quan trọng nhất của đề tài:**
  >   - Trên mặt thường: Khi cười (`happy`), Grad-CAM tập trung rực lửa vào 2 khóe miệng cười.
  >   - Nhưng khi đối tượng đeo khẩu trang: Mô hình thông thường vẫn cố nhìn vào vùng khẩu trang và đoán bừa thành `neutral`. Ngược lại, mô hình Mask-Aware của nhóm tạo ra một **Sự chuyển dịch vùng chú ý (Attention Shift)** kỳ diệu: Tâm điểm kích hoạt tự động dịch chuyển 100% lên vùng đuôi mắt (vết chân chim Duchenne smile) và nếp nhăn trán để đưa ra nhãn `happy` chính xác tuyệt đối.
  > - AI của chúng em không đoán mò, mà thực sự 'nhìn thấy' cảm xúc qua đôi mắt!"*

---

#### Slide 13: Thiết kế hệ thống thời gian thực & Ổn định chuỗi thời gian (Real-Time Tracking)
* **Visuals:**
  * Minh họa bài toán Video/Webcam: Hiện tượng nhảy nhãn (label flickering) giữa các frame liên tiếp.
  * Giải pháp của nhóm: **IoU Face Tracker + Exponential Moving Average (EMA) Smoothing** trên vector xác suất 7 lớp qua thời gian.
* **Speaker 3 nói:**
  > *"Trong môi trường webcam và video thực tế, một thách thức rất lớn là **Hiện tượng nhảy nhãn (Label Flickering)**: Do nhiễu ánh sáng hoặc micro-expression, frame trước vừa đoán Happy thì frame sau lại giật sang Neutral khiến người dùng khó chịu.
  > - Nhóm đã thiết kế module **`FaceTracker`** kết hợp giữa thuật toán liên kết IoU (Intersection over Union) và bộ lọc **Làm mượt xác suất theo thời gian (Temporal EMA Smoothing)**:
  >   - Mỗi khuôn mặt trong khung hình được gán một Track ID duy nhất và ổn định.
  >   - Xác suất dự đoán được cập nhật theo công thức: $P_{\text{smooth}} = (1 - \alpha) P_{\text{cũ}} + \alpha P_{\text{mới}}$ (với $\alpha = 0.4$).
  >   - Nhờ đó, nhãn cảm xúc hiển thị mượt mà, triệt tiêu 100% hiện tượng rung giật giữa các khung hình liên tiếp."*

---

#### Slide 14: Tối ưu hóa hiệu năng & Benchmark FPS (Hardware Benchmark & Optimization)
* **Visuals:**
  * Bảng số liệu FPS thực tế đo từ script `scripts/benchmark_fps.py` trên Apple M1 (MPS GPU) / NVIDIA GTX 1660 Ti:

| Số lượng khuôn mặt | FPS (Không bật Grad-CAM) | FPS (Có Grad-CAM cache 10 frame) | Thời gian chi tiết |
| :---: | :---: | :---: | :---: |
| **1 khuôn mặt** | **55.0 FPS** ⚡ | **37.4 FPS** ⚡ | Detect ~5.5ms \| Predict ~12.7ms |
| **2 khuôn mặt** | **37.4 FPS** | **25.9 FPS** | Predict ~21.1ms |
| **3 khuôn mặt** | **28.4 FPS** | **20.7 FPS** | Predict ~29.7ms |
| **5 khuôn mặt** | **17.4 FPS** | **15.1 FPS** | Quét đám đông mượt mà |

* **Speaker 3 nói:**
  > *"Để hệ thống chạy mượt trên các thiết bị phổ thông không cần siêu máy tính, nhóm đã thực hiện 2 tối ưu kỹ thuật lớn:
  > 1. **Chế độ suy luận linh hoạt:** Lúc đánh giá offline, ta bật TTA lật ngang để tối đa hóa độ chính xác. Nhưng khi chạy webcam trực tiếp, nhóm đo được việc tắt TTA chỉ làm giảm dưới 0.1% accuracy nhưng tăng tốc độ xử lý gấp đôi!
  > 2. **Kỹ thuật Grad-CAM Temporal Cache:** Một lần lan truyền ngược (backward pass) để vẽ Grad-CAM mất tới 90–100ms (kéo tụt FPS xuống còn 7.7 FPS). Nhóm tối ưu bằng cách: **chỉ tính toán lại Grad-CAM mỗi 10 frame** và tái sử dụng heatmap cho các frame đệm. Kết quả: Tốc độ tăng vọt từ **7.7 FPS lên 37.4 FPS** — hoàn toàn thời gian thực!
  > - Ngay cả khi có 2 đến 3 người cùng xuất hiện trong khung hình 720p, hệ thống vẫn duy trì ổn định từ 26 đến 37 FPS."*

---

#### Slide 15: Hệ sinh thái ứng dụng: Streamlit Web Dashboard & Desktop App
* **Visuals:**
  * Giao diện Streamlit Dark Theme sang trọng (`app.py`): Tab Single Face, Multi-Face Crowd, XAI Inspector, Head-to-Head Comparison.
  * Giao diện Desktop OpenCV Webcam (`webcam_app.py`): Nhấn phím `1`/`2`/`M` để đổi model, nhấn `G` để bật Grad-CAM real-time.
* **Speaker 3 nói:**
  > *"Toàn bộ nghiên cứu được đóng gói thành một hệ sinh thái sản phẩm hoàn chỉnh:
  > 1. **Web Dashboard Streamlit (`app.py`):** Giao diện Dark theme hiện đại hỗ trợ tải ảnh đơn, phân tích video trích xuất từng frame, phân tích đám đông nhiều người và đặc biệt là tab **Head-to-Head Comparison** cho phép đặt mô hình Baseline và Mask-Aware cạnh nhau để đối chiếu trực tiếp.
  > 2. **Desktop Real-Time App (`webcam_app.py`):** Ứng dụng chạy trực tiếp bằng OpenCV dành cho camera giám sát, nạp sẵn cả 2 mô hình vào VRAM, cho phép người dùng bấm phím tắt chuyển đổi tức thì giữa mô hình Baseline và Mask-Aware để chứng kiến sự khác biệt."*

---

#### Slide 16: Trực tiếp Live Demo (Live Demonstration - 3 Phút)
* **Hành động của Speaker 3:** Chuyển sang màn hình demo chạy lệnh `streamlit run app.py` và `webcam_app.py`.
* **Kịch bản Demo thực chiến từng bước:**
  1. **Demo 1 (So sánh Head-to-Head trên ảnh khẩu trang):** 
     - *Thao tác:* Vào tab so sánh, nạp ảnh người đeo khẩu trang cười (`happy`).
     - *Speaker 3 thuyết minh:* *"Hội đồng có thể thấy: Mô hình Baseline bên trái dự đoán sai là `Neutral` (xác suất 78%). Nhưng mô hình Mask-Aware bên phải nhận diện chuẩn xác `Happy` (xác suất 91%), và Grad-CAM chỉ rõ vùng kích hoạt tập trung vào đôi mắt cười."*
  2. **Demo 2 (Phân tích ảnh nhóm nhiều người):**
     - *Thao tác:* Nạp ảnh đám đông 4–5 người trong `data/multi_face_samples/`.
     - *Speaker 3 thuyết minh:* *"YuNet bắt trọn 5 khuôn mặt cùng lúc, căn chỉnh hình học và gán nhãn biểu cảm riêng biệt cho từng người không hề bị sót."*
  3. **Demo 3 (Live Webcam thời gian thực):**
     - *Thao tác:* Bật camera máy tính, thực hiện biểu cảm mặt thường $\to$ sau đó đeo khẩu trang lên mặt.
     - *Speaker 3 thuyết minh:* *"Khi em đeo khẩu trang che kín mũi và miệng, hệ thống vẫn phản hồi tức thì với biểu cảm cười hoặc ngạc nhiên của em với tốc độ hơn 35 FPS, nhãn không hề bị nhảy giật."*

---

#### Slide 17: Tổng kết đóng góp & Lời cảm ơn (Conclusion & Closing)
* **Visuals:**
  * Tóm tắt 4 đóng góp chính: (1) Mask-Aware +22.7%, (2) Quy trình đánh giá 0% leakage, (3) YuNet 5-point alignment, (4) Hệ thống real-time 37+ FPS kèm XAI.
  * Hạn chế & Hướng phát triển: Thu thập thêm dữ liệu khẩu trang thật; mở rộng sang nhận diện Vi-biểu cảm (Micro-expressions).
* **Speaker 3 nói:**
  > *"Để tổng kết lại, dự án của nhóm em đã chứng minh rằng: **Bằng việc kết hợp đúng đắn giữa Backbone tiền huấn luyện trong miền khuôn mặt, quy trình chia dữ liệu không rò rỉ, căn chỉnh hình học chuẩn mực và kỹ thuật hiệu chỉnh prior-bias, chúng ta hoàn toàn có thể giải quyết bài toán nhận diện cảm xúc bị che khuất với độ chính xác cao và tốc độ thời gian thực.**
  > 
  > Nhóm xin chân thành cảm ơn quý thầy cô trong hội đồng đã chú ý lắng nghe. Sau đây, nhóm chúng em rất mong nhận được những câu hỏi và đóng góp quý báu từ thầy cô!"*

---

## 🛡️ 4. BỘ CÂU HỎI PHẢN BIỆN BẢO VỆ ĐỒ ÁN (Q&A DEFENSE MASTER GUIDE)

Dưới đây là 10 câu hỏi hóc búa nhất mà hội đồng/giảng viên thường đặt ra, kèm phân vai người trả lời và câu trả lời mẫu chuẩn xác về mặt khoa học:

---

### ❓ Câu 1 (Dành cho Speaker 1): Tại sao nhóm lại dùng ảnh khẩu trang tổng hợp (MaskTheFace) mà không thu thập ảnh người thật đeo khẩu trang thật để huấn luyện?
* **Speaker 1 trả lời:**
  > *"Dạ thưa thầy/cô, trong nghiên cứu Khoa học Dữ liệu, việc dùng ảnh tổng hợp có kiểm soát (Synthetic Occlusion) mang lại 2 lợi thế quyết định:
  > 1. **Bảo toàn nhãn chân lý (Ground-Truth Preservation):** Nếu thu thập ảnh người thật đeo khẩu trang ngoài đời, việc gán nhãn cảm xúc rất chủ quan vì người gán nhãn cũng bị khẩu trang che mất biểu cảm. Trong khi đó, ảnh RAF-DB đã được 40 chuyên gia gán nhãn tỉ mỉ lúc mặt chưa che. Khi gắn khẩu trang nhân tạo lên, nhãn cảm xúc nguyên bản được bảo toàn 100%.
  > 2. **Ghép cặp hoàn hảo (Paired Counterfactual Data):** Chúng ta có chính xác ảnh người đó lúc chưa đeo khẩu trang và lúc đeo khẩu trang. Điều này cho phép nhóm kiểm soát được biến số can thiệp và kiểm tra rò rỉ dữ liệu một cách tuyệt đối.
  > Ngoài ra, nhóm vẫn thu thập một tập nhỏ ảnh người thật đeo khẩu trang thật (`data/real_masked_faces/`) để phục vụ demo định tính thực tế trên ứng dụng ạ."*

---

### ❓ Câu 2 (Dành cho Speaker 1): Phép căn chỉnh khuôn mặt bằng 5 điểm YuNet khác gì việc crop hình chữ nhật bounding box thông thường? Nó đóng góp bao nhiêu vào kết quả?
* **Speaker 1 trả lời:**
  > *"Dạ thưa thầy/cô, crop bounding box thông thường chỉ cắt một vùng chữ nhật quanh mặt. Khi người dùng nghiêng đầu, ngửa mặt hoặc góc camera bị chéo, vị trí của mắt và miệng trong ảnh crop sẽ bị lệch tọa độ hoàn toàn so với dữ liệu huấn luyện.
  > - Phép căn chỉnh của nhóm sử dụng **Similarity Transform (Phép biến đổi tương đồng 2D)**: Dựa trên 5 điểm mốc (2 mắt, đỉnh mũi, 2 khóe miệng), thuật toán xoay ảnh để trục 2 mắt luôn nằm ngang, đồng thời co dãn tỷ lệ để khoảng cách giữa 2 mắt và từ mắt tới miệng khớp chính xác vào **Template chuẩn của RAF-DB** (với độ lệch chuẩn đo được $\sim 0.03$).
  > - Nhờ căn chỉnh, ảnh crop từ webcam đưa vào mô hình có cùng phân phối không gian với dữ liệu lúc huấn luyện. Thực nghiệm chứng minh bước căn chỉnh này giúp tăng độ chính xác thêm từ **1.5% đến 2.0%** trên toàn pipeline ạ."*

---

### ❓ Câu 3 (Dành cho Speaker 2): Bản chất của Logit-Adjusted Loss là gì? Tại sao sau khi train lại cần bước Calibrate Prior-Bias trên tập Validation?
* **Speaker 2 trả lời:**
  > *"Dạ thưa thầy/cô:
  > 1. **Trong lúc huấn luyện:** Do dữ liệu cực kỳ mất cân bằng (Happy gấp 17 lần Fear), Logit-Adjusted Loss cộng thêm một lượng bù trừ $\tau \log \pi_c$ vào logit của lớp $c$ theo tần suất xuất hiện. Việc này ép mô hình phải tạo ra khoảng cách biên phân loại (margin) lớn hơn cho các lớp thiểu số, giúp mạng không bỏ quên các lớp ít mẫu.
  > 2. **Tại sao cần Calibrate:** Khi train với Logit Adjustment, mô hình học cách dự đoán như thể 7 lớp có xác suất ngang nhau. Nhưng trên tập kiểm thử thực tế, phân phối vẫn bị lệch (nhiều người bình thản và vui vẻ). Do đó, mô hình bị hiện tượng đoán thừa `fear` và `disgust` (recall cao nhưng precision bị kéo tụt).
  > - Bước hiệu chỉnh `src/calibrate.py` quét tìm tham số tối ưu $t$ sao cho khi cộng $t \log \pi$ vào logit lúc suy luận, Macro F1 trên **tập Validation** đạt cực đại. Vì $t = 0.85$ được chọn hoàn toàn trên tập Val, không hề chạm vào tập Test, nên đây là quy trình hiệu chỉnh hoàn toàn hợp lệ và giúp cân bằng tối ưu giữa Precision và Recall trên tập Test ạ."*

---

### ❓ Câu 4 (Dành cho Speaker 2): Tại sao nhóm lại chọn EfficientNet-B0 mà không dùng các kiến trúc Vision Transformer (ViT) hiện đại nhất?
* **Speaker 2 trả lời:**
  > *"Dạ thưa thầy/cô, nhóm đã cân nhắc rất kỹ giữa độ chính xác và tính khả thi trong môi trường thực tế (Real-Time Edge Deployment):
  > 1. **Hiệu năng phần cứng:** Vision Transformer (như Swin-T hay ViT-Base) đòi hỏi bộ nhớ lớn và cơ chế Self-Attention có độ phức tạp tính toán bậc hai $\mathcal{O}(N^2)$. Khi chạy đa khuôn mặt trên webcam ở độ phân giải HD, ViT khó đạt được mức 30 FPS trên CPU hoặc GPU phổ thông.
  > 2. **Lợi thế Pre-trained Domain:** HSEmotion EfficientNet-B0 đã được tiền huấn luyện trên VGGFace2 và AffectNet — đây là trọng số chuyên biệt cho khuôn mặt người. Một mạng ViT dù mạnh nhưng chỉ pre-train trên ImageNet tổng quát vẫn sẽ thua EfficientNet chuyên biệt về mặt Action Units nếu không có lượng dữ liệu khổng lồ hàng triệu ảnh để fine-tune.
  > - Với chỉ ~4 triệu tham số, EfficientNet-B0 đạt tới **88.4% accuracy** và chạy mượt mà **55 FPS**, hoàn toàn vượt trội về tỷ lệ hiệu năng trên chi phí tính toán ạ."*

---

### ❓ Câu 5 (Dành cho Speaker 2): Tại sao khi đánh giá Benchmark, nhóm bắt buộc phải chạy bằng fp32 mà không dùng fp16 (AMP) để cho nhanh?
* **Speaker 2 trả lời:**
  > *"Dạ thưa thầy/cô, nhóm phát hiện ra điều này qua thực nghiệm đo đạc nghiêm ngặt:
  > - Trong lúc huấn luyện, ta dùng PyTorch AMP (Automatic Mixed Precision - fp16) để tận dụng Tensor Cores giúp tăng tốc 2.5 lần và tiết kiệm VRAM.
  > - Tuy nhiên, khi đánh giá (evaluation), các hàm tính softmax và sự chênh lệch nhỏ giữa các logit của các lớp thiểu số (như Fear và Disgust) rất nhạy cảm với sai số làm tròn số học (numerical precision).
  > - Thực nghiệm đối đầu trên cùng một checkpoint: chạy dưới fp16 autocast làm accuracy bị giảm mất **gần 2 điểm phần trăm** (82.37% so với 84.22% fp32 thô trên RAF-DB test). Do đó, để đảm bảo tính chuẩn xác và công bằng tuyệt đối cho báo cáo khoa học, mọi kết quả benchmark của nhóm đều được đo đạc ở chuẩn dấu phẩy động 32-bit (fp32) ạ."*

---

### ❓ Câu 6 (Dành cho Speaker 1 & 2): Trong báo cáo `leakage_report.txt`, có 49 ảnh test trùng lặp với train. Liệu điều này có làm sai lệch kết quả đánh giá không?
* **Speaker 1 & 2 trả lời:**
  > *"Dạ thưa thầy/cô, nhóm xin khẳng định: **Điều này không làm sai lệch kết quả và là đặc tính cố hữu của bản thân bộ dữ liệu RAF-DB**:
  > 1. RAF-DB được các tác giả gốc thu thập từ mạng xã hội Flickr năm 2017. Trong split chính thức của tác giả, họ chia ngẫu nhiên mà không tách danh tính người (Identity-independent), dẫn tới việc có khoảng 1.6% ảnh chụp cùng một người ở cùng một buổi chụp xuất hiện ở cả train và test gốc.
  > 2. Công cụ `scripts/check_leakage.py` của nhóm được thiết kế cực kỳ nhạy nên đã phát hiện ra chính xác 49 cặp ảnh này.
  > 3. Nhóm đã thực hiện thí nghiệm cô lập: Loại bỏ hoàn toàn 49 ảnh này ra khỏi tập Test và đánh giá lại mô hình HSEmotion-B0. Kết quả: Accuracy chỉ thay đổi trong khoảng từ **0.05% đến 0.19%**, hoàn toàn không ảnh hưởng tới kết luận tổng thể của đề tài. Việc nhóm báo cáo rõ ràng 49 ảnh này thể hiện tính trung thực khoa học cao nhất của nhóm ạ."*

---

### ❓ Câu 7 (Dành cho Speaker 3): Grad-CAM hoạt động như thế nào về mặt toán học? Tại sao lại chọn can thiệp vào tầng `bn2` của EfficientNet?
* **Speaker 3 trả lời:**
  > *"Dạ thưa thầy/cô:
  > 1. Về mặt toán học, Grad-CAM tính đạo hàm riêng của điểm số dự đoán lớp $c$ ($y^c$) đối với từng điểm ảnh của feature map $A^k$ tại tầng tích chập cuối cùng: $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$. Các trọng số $\alpha_k^c$ này thể hiện mức độ quan trọng của kênh đặc trưng $k$ đối với việc quyết định lớp $c$. Sau đó, ta tính tổ hợp tuyến tính các feature maps có trọng số và áp hàm $\text{ReLU}$ để chỉ giữ lại các đặc trưng có đóng góp tích cực: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
  > 2. Nhóm chọn tầng `bn2` (Batch Normalization ngay sau khối tích chập 1x1 cuối cùng của EfficientNet-B0) vì đây là tầng chứa feature map ở mức ngữ nghĩa cao nhất (high-level semantic features) trước khi bị nén thành một vector phẳng 1D bởi Global Average Pooling. Nhờ đó, bản đồ nhiệt giữ được độ phân giải không gian tốt nhất để phân biệt giữa mắt, mũi và miệng ạ."*

---

### ❓ Câu 8 (Dành cho Speaker 3): Face Tracker xử lý thế nào khi một khuôn mặt bị che khuất tạm thời hoặc quay mặt đi rồi quay lại?
* **Speaker 3 trả lời:**
  > *"Dạ thưa thầy/cô, module `FaceTracker` của nhóm hoạt động theo cơ chế quản lý trạng thái vết (Track State Management):
  > - Khi một khuôn mặt biến mất khỏi khung hình (do quay mặt đi hoặc vật cản che), tracker không xóa ngay ID mà gán biến đếm `missed += 1`.
  > - Nhóm thiết lập ngưỡng chịu đựng `max_missed = 10` frames. Nếu trong vòng 10 frames (~0.3 giây) khuôn mặt đó xuất hiện lại ở vị trí lân cận với IoU $\ge 0.3$, tracker sẽ ghép nối lại và duy trì tiếp ID cũ cùng bộ nhớ xác suất đã làm mượt.
  > - Chỉ khi khuôn mặt biến mất quá 10 frames liên tiếp, ID đó mới được giải phóng để tránh chiếm dụng bộ nhớ. Điều này giúp hệ thống duy trì tính liền mạch rất cao trong thực tế ạ."*

---

### ❓ Câu 9 (Dành cho Speaker 3): Làm thế nào nhóm đạt được 37.4 FPS khi vừa nhận diện vừa vẽ Grad-CAM, trong khi một lượt backward pass mất tới gần 100ms?
* **Speaker 3 trả lời:**
  > *"Dạ thưa thầy/cô, đây chính là giải pháp kỹ thuật tối ưu mang tên **Grad-CAM Temporal Caching**:
  > - Trong luồng video 30 FPS, cơ mặt của con người không thể thay đổi hoàn toàn cấu trúc giải phẫu chỉ trong vòng 30 phần nghìn giây. Vùng chú ý của mạng (ví dụ vào đôi mắt) có tính quán tính cục bộ rất cao qua các khung hình liên tiếp.
  > - Thay vì thực hiện lan truyền ngược (backward pass) trên từng frame đơn lẻ (khiến hệ thống bị nghẽn ở mức 7.7 FPS), hệ thống của nhóm **chỉ kích hoạt backward pass một lần sau mỗi 10 frames** (tức khoảng 0.3 giây một lần) để cập nhật ma trận trọng số nhiệt mới.
  > - Ở 9 frames nằm giữa, hệ thống áp dụng ma trận nhiệt đã tính lên tọa độ bbox hiện tại của khuôn mặt.
  > - Nhờ cơ chế này, chi phí tính toán của Grad-CAM giảm đi 10 lần, đưa FPS từ **7.7 vọt lên 37.4 FPS** mà trải nghiệm thị giác của người dùng vẫn mượt mà và trực quan tuyệt đối ạ."*

---

### ❓ Câu 10 (Dành cho Cả nhóm): Nếu phải nêu ra 1 điểm đột phá duy nhất của đề tài mà nhóm tự hào nhất, đó là điều gì?
* **Cả nhóm trả lời:**
  > *"Dạ thưa hội đồng, điểm đột phá lớn nhất của dự án là **Tính toàn vẹn thực chứng (Empirical Integrity) và Khả năng ứng dụng thực tế**:
  > - Nhóm không dừng lại ở việc tạo ra một mô hình chạy trên notebook với các con số ảo bị rò rỉ.
  > - Nhóm đã chứng minh một mô hình học sâu nhỏ gọn (EfficientNet-B0) có thể tăng tới **+22.7 điểm phần trăm độ chính xác khi đeo khẩu trang**, giữ vững **88.4% trên mặt thường**, chạy thời gian thực **37–55 FPS** trên camera thật, và có khả năng giải thích minh bạch từng quyết định bằng thị giác.
  > - Mọi dòng code, quy trình chia dữ liệu và số liệu đánh giá đều được đóng gói sạch sẽ, kiểm toán rò rỉ độc lập và sẵn sàng triển khai thành ứng dụng thực tiễn phục vụ cộng đồng ạ!"*

---

## 💡 5. CHECKLIST TRƯỚC GIỜ LÊN SÂN KHẤU (PRE-FLIGHT CHECKLIST)

1. [ ] **Môi trường demo:** Chạy sẵn lệnh `uv run streamlit run app.py` trong terminal 1 và kiểm tra camera trên trình duyệt.
2. [ ] **Dự phòng (Offline Backup):** Chuẩn bị sẵn một video quay màn hình chất lượng cao phòng trường hợp camera hội trường bị lỗi kết nối.
3. [ ] **Chuyển giao micro:** Tập dượt 2 câu chuyển tiếp giữa Speaker 1 $\to$ Speaker 2 $\to$ Speaker 3 thật nhuần nhuyễn.
4. [ ] **Thời gian:** Speaker 1 kết thúc ở phút thứ 5, Speaker 2 kết thúc ở phút 10:30, Speaker 3 kết thúc ở phút 15:30.
5. [ ] **Tâm thế:** Tự tin, nói to rõ ràng, tự hào về số liệu thực chứng 88.4% / 79.3% của nhóm!
