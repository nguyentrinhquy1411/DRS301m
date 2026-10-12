# 📕 CẨM NANG THUYẾT TRÌNH DÀNH CHO SPEAKER 2
## Vai trò: Deep Learning & Evaluation Engineer
### Phụ trách: Kiến trúc Mô hình (HSEmotion EfficientNet-B0), Huấn luyện 2 giai đoạn, Logit Adjustment, Hiệu chỉnh Prior-Bias (`calibrate.py`), Benchmark Định lượng (+22.7% Masked Acc) & Ablation Study

---

## ⏱️ 1. THÔNG TIN PHÂN BỔ THỜI GIAN & NHIỆM VỤ

* **Thời lượng trên sân khấu:** **5.5 phút** (Từ phút **05:00** đến phút **10:30**).
* **Số lượng slide phụ trách:** **Slide 6 đến Slide 11** (6 slides - Trọng tâm kỹ thuật & Số liệu của toàn bộ đề tài).
* **Mục tiêu tối thượng của Speaker 2:** 
  1. Thuyết phục hội đồng về tính khoa học trong việc chọn **Face-Pretrained Backbone** thay vì ImageNet thông thường.
  2. Trình bày được bản chất toán học của **Logit-Adjusted Loss** và giải pháp **Hiệu chỉnh Prior-Bias hậu huấn luyện** (Điểm sáng kỹ thuật rất được giảng viên đánh giá cao).
  3. Bùng nổ với **Bảng số liệu Benchmark vàng**: Tăng vọt **+22.68%** trên ảnh có khẩu trang mà không làm suy giảm độ chính xác trên mặt thường.
  4. Phân tích ma trận nhầm lẫn và bảo vệ kết quả bằng Ablation Study mạch lạc.
* **Phong thái trình bày:** Chắc chắn, gãy gọn, làm chủ hoàn toàn các con số và công thức toán học, nói chậm và dứt khoát ở các bảng so sánh.

---

## 🎬 2. KỊCH BẢN NÓI CHI TIẾT TỪNG SLIDE (VERBATIM SCRIPT)

---

### 🟢 SLIDE 6: KIẾN TRÚC MÔ HÌNH & FACE-PRETRAINED BACKBONE (05:00 – 06:00)
* **Nội dung slide:**
  * Sơ đồ cấu trúc mạng: EfficientNet-B0 (MBConv blocks + Depthwise Separable Conv + Squeeze-and-Excitation).
  * So sánh Pretrained Domain:
    * ImageNet: 1,000 lớp đồ vật tổng quát (chó, mèo, ô tô) $\to$ không hiểu Action Units.
    * **HSEmotion:** 2 tầng tiền huấn luyện chuyên sâu trong miền khuôn mặt: **VGGFace2** (nhận diện danh tính khuôn mặt) $\to$ **AffectNet** (fine-tune nhận diện cảm xúc khuôn mặt).
  * Kích thước mô hình: ~4 triệu tham số, kích thước ảnh đầu vào chuẩn 224x224.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Em xin cảm ơn bạn [Tên Speaker 1]! Kính thưa quý thầy cô trong hội đồng, sau khi dữ liệu đã được làm sạch và căn chỉnh hình học, câu hỏi kỹ thuật then chốt đặt ra cho nhóm là: **Nên lựa chọn kiến trúc mạng và trọng số tiền huấn luyện nào?**
  > 
  > *Phần lớn các nghiên cứu hiện nay thường tải thẳng pre-trained weights từ ImageNet. Tuy nhiên, ImageNet được sinh ra để phân biệt 1,000 loài động vật và đồ vật tổng quát — hoàn toàn không có nhận thức không gian về các đơn vị vận động cơ mặt (Facial Action Units).*
  > 
  > *Nhóm đã tạo ra bước đột phá khi lựa chọn **HSEmotion EfficientNet-B0**:*
  > * *Đây là mô hình đã trải qua 2 tầng tiền huấn luyện chuyên sâu trong miền khuôn mặt người: Tầng 1 trên bộ dữ liệu **VGGFace2** với hơn 3 triệu ảnh mặt để học cấu trúc giải phẫu người, và Tầng 2 được fine-tune trên **AffectNet** — bộ dữ liệu cảm xúc lớn nhất thế giới.*
  > * *Với cấu trúc MBConv, Depthwise Separable Convolution kết hợp cơ chế chú ý kênh Squeeze-and-Excitation, EfficientNet-B0 chỉ có khoảng 4 triệu tham số nhưng nắm bắt các vi biểu cảm ở khóe mắt và vòm chân mày nhạy bén hơn hẳn các mạng nặng nề như ResNet-50 hay MobileNetV3.*
  > * *Classifier head được nhóm tái cấu trúc với 7 nơ-ron đầu ra tương ứng 7 lớp cảm xúc cơ bản của RAF-DB."*
* **Speaker Note:** Nhấn mạnh cụm từ **"Face-Pretrained Backbone"** để hội đồng thấy rõ nhóm hiểu sâu về Domain Adaptation chứ không chỉ copy-paste code mẫu.

---

### 🟢 SLIDE 7: CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN & REGULARIZATION (06:00 – 07:00)
* **Nội dung slide:**
  * Sơ đồ 2 Phase huấn luyện:
    * **Phase 1 (RAF-DB Baseline):** Huấn luyện trên 12,271 ảnh gốc $\to$ 1 epoch warmup head ($LR=10^{-3}$) $\to$ Fine-tune toàn mạng 25 epochs ($LR=10^{-4}$), Cosine Annealing, AdamW.
    * **Phase 2 (Mask-Aware Fine-Tuning):** Khởi tạo từ checkpoint Phase 1 $\to$ Fine-tune 20 epochs trên tập Combined (Gốc + Khẩu trang).
  * Tổ hợp Regularization:
    * **LowerFaceOcclusion ($p=0.3$):** Chủ động che nửa dưới khuôn mặt ngẫu nhiên khi train.
    * **Mixup ($\alpha=0.2$) & CutMix ($\alpha=1.0$):** Chống over-confidence.
    * **EMA (Exponential Moving Average):** Hệ số suy giảm 0.998 giúp trọng số hội tụ mượt mà.
    * **PyTorch AMP (fp16):** Tăng tốc 2.5 lần lúc train trên GPU.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Để mô hình vừa nhận diện xuất sắc mặt mộc, vừa không bị sụp đổ khi gặp khẩu trang, nhóm thiết kế **Chiến lược huấn luyện 2 giai đoạn (Two-Stage Transfer Learning)**:*
  > 
  > * **Giai đoạn 1 (RAF-DB Baseline):** Huấn luyện trên ảnh gốc RAF-DB. Ở epoch đầu tiên, ta đóng băng toàn bộ backbone, chỉ cho classifier head học với learning rate $10^{-3}$ để tránh phá vỡ trọng số pre-trained. Sau đó, mở toàn mạng fine-tune với learning rate vi sai $10^{-4}$, lịch trình giảm tốc Cosine Annealing.*
  > * **Giai đoạn 2 (Mask-Aware Adaptation):** Khởi tạo trực tiếp từ checkpoint tốt nhất của Giai đoạn 1, fine-tune 20 epochs trên tập dữ liệu tổng hợp (Combined: gốc + khẩu trang).*
  > 
  > *Trong suốt quá trình huấn luyện, nhóm tích hợp một tổ hợp điều hòa (Regularization) mạnh mẽ:*
  > * *Đặc biệt là kỹ thuật **LowerFaceOcclusion**: Chủ động cắt che ngẫu nhiên nửa dưới mặt với xác suất 30% để ép mạng nơ-ron phải chuyển hướng tìm kiếm đặc trưng lên nửa trên khuôn mặt.*
  > * *Kết hợp cùng **Mixup, CutMix, Random Erasing** và **Exponential Moving Average (EMA)** với hệ số suy giảm 0.998. EMA giúp làm mượt trọng số qua các epoch, triệt tiêu dao động ngẫu nhiên và giúp mô hình đạt độ tổng quát hóa tối ưu."*
* **Speaker Note:** Giải thích rõ: Giai đoạn 1 dạy mạng học cảm xúc con người; Giai đoạn 2 dạy mạng cách 'đọc vị' cảm xúc khi bị che khuất.

---

### 🟢 SLIDE 8: HIỆU CHỈNH PRIOR-BIAS HẬU HUẤN LUYỆN (CALIBRATE.PY) (07:00 – 08:00)
* **Nội dung slide:**
  * Thách thức: Dữ liệu mất cân bằng nghiêm trọng (Happy 4,772 ảnh vs Fear 281 ảnh).
  * Công thức **Logit-Adjusted Loss**: $\mathcal{L} = -\log \frac{e^{f_y(x) + \tau \log \pi_y}}{\sum_c e^{f_c(x) + \tau \log \pi_c}}$ (mở rộng biên phân loại cho lớp hiếm).
  * Tác dụng phụ: Logits thô dự đoán như thể 7 lớp cân bằng $\to$ trên tập test thực tế bị đoán thừa Fear/Disgust (Recall cao nhưng Precision thấp).
  * Giải pháp **`src/calibrate.py`**: Quét hệ số $t \in [-0.5, 1.5]$ trên **tập Validation** để tìm $\arg\max (\text{logits} + t \log \pi)$ đạt Macro F1 cao nhất.
  * Kết quả: Điểm cực đại tại **$t = +0.85$** $\to$ F1 trên Val tăng từ 0.7109 lên **0.7721**, Acc tăng từ 79.55% lên **84.64%**. Lưu trực tiếp vào checkpoint!
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Một bài toán hóc búa bậc nhất của dữ liệu cảm xúc thực tế là **Sự mất cân bằng lớp cực lớn**: Trong RAF-DB, số lượng ảnh Vui vẻ (`happy`) gấp tới 17 lần ảnh Sợ hãi (`fear`). Nếu train bằng hàm mất mát Cross-Entropy thông thường, mô hình sẽ bị thiên vị nặng nề về các lớp chiếm đa số.*
  > 
  > *Nhóm đã áp dụng **Logit-Adjusted Loss** trong lúc train để cộng thêm một lượng bù trừ tỷ lệ nghịch với tần suất xuất hiện, ép mô hình phải nới rộng biên quyết định (margin) cho các lớp thiểu số.*
  > 
  > *Tuy nhiên, cơ chế này nảy sinh một tác dụng phụ: Nó khiến logits thô dự đoán như thể 7 lớp có xác suất ngang bằng nhau. Nhưng ngoài đời thực và trên tập test, số lượng người vui vẻ và bình thản vẫn chiếm đa số. Hậu quả là mô hình bị đoán thừa `fear` và `disgust` — recall rất cao nhưng precision lại bị kéo tụt xuống.*
  > 
  > *Để giải quyết triệt để, nhóm phát triển thuật toán **`src/calibrate.py`**:*
  > * *Hệ thống quét hệ số hiệu chỉnh $t$ trong khoảng $[-0.5, 1.5]$ để tìm nghiệm: $\arg\max(\text{logits} + t \log \pi)$ đạt Macro F1 cao nhất **chỉ tính trên tập Validation**.*
  > * *Thuật toán tìm ra điểm tối ưu tại **$t = +0.85$** cho mô hình Mask-Aware.*
  > * *Nhờ đó, F1 trên tập Val tăng vọt từ 0.7109 lên **0.7721**, Accuracy tăng từ 79.55% lên **84.64%**.*
  > * *Vector logit_bias này được ghi trực tiếp vào checkpoint để áp dụng lúc suy luận, mang lại hiệu quả vượt trội mà không cần train lại mô hình."*
* **Speaker Note:** Nhấn mạnh rằng: $t$ được chọn 100% trên tập Val, **tuyệt đối không chạm vào tập Test** $\to$ Tuân thủ chuẩn mực học máy khắt khe nhất.

---

### 🟢 SLIDE 9: KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG TOÀN DIỆN (08:00 – 09:15)
* **Nội dung slide:** Bảng vàng kết quả kiểm thử chính thức (fp32, TTA + Prior-bias trên toàn bộ 5,043 ảnh test độc lập):

| Mô hình thử nghiệm | Mặt thường (Unmasked - 3,068 ảnh) | Mặt có khẩu trang (Masked - 1,975 ảnh) | Toàn bộ tập test (Combined - 5,043 ảnh) |
| :--- | :---: | :---: | :---: |
| **MobileNetV3 (Pipeline cũ của nhóm)** | 79.86% / F1 0.718 | 73.27% / F1 0.628 | 77.28% / F1 0.684 |
| **HSEmotion-B0 (Chỉ train RAF-DB)** | **89.15%** / F1 0.833 | 56.61% / F1 0.534 | 76.40% / F1 0.725 |
| **HSEmotion-B0 (Mask-Aware ⭐ Đề xuất)** | **88.43%** / F1 0.822 | **79.29%** / F1 0.708 | **84.85%** / F1 0.779 |
| **Mức tăng trưởng đột phá** | *(Chênh lệch không đáng kể <0.7%)* | **+22.68% Acc, +0.174 F1 🚀** | **+8.45% Acc, +0.054 F1 🚀** |

* **Lời thoại trực tiếp (Speaking Script):**
  > *"Và sau đây là slide quan trọng nhất chứng minh giá trị thực nghiệm của đề tài. Mọi số liệu trong bảng này đều được đo độc lập ở chuẩn dấu phẩy động 32-bit (fp32) có bật TTA lật ngang và Prior-bias trên 5,043 ảnh test chính thức:
  > 
  > * **Thứ nhất, trên mặt không đeo khẩu trang (Unmasked 3,068 ảnh):** Mô hình Mask-Aware của nhóm đạt **88.43% Accuracy và 0.822 Macro F1** — gần như tương đương với mô hình chuyên biệt chỉ học mặt thường (89.15%). Điều này chứng minh: Mô hình hoàn toàn **không bị hiện tượng quên thảm họa (Catastrophic Forgetting)**, giữ nguyên năng lực phân loại xuất sắc trên mặt mộc.*
  > 
  > * **Thứ hai, trên mặt đeo khẩu trang (Masked 1,975 ảnh) — Đây là bước nhảy vọt của đề tài:** Nếu lấy mô hình chỉ train trên RAF-DB thông thường mang ra dự đoán mặt khẩu trang, độ chính xác chỉ đạt vỏn vẹn **56.61%**. Nhưng với mô hình Mask-Aware của nhóm, độ chính xác vọt lên tới **79.29%** — tức là **tăng tới 22.68 điểm phần trăm**, Macro F1 tăng mạnh từ 0.534 lên **0.708**!*
  > 
  > * **Thứ ba, trên toàn bộ 5,043 ảnh kiểm thử (Combined):** Độ chính xác chung đạt **84.85%** và Macro F1 đạt **0.779**.*
  > 
  > *So với baseline MobileNetV3 cũ ở giai đoạn đầu của nhóm (77.28%), hệ thống mới vượt trội hơn tới **7.57% Accuracy** và giải quyết triệt để bài toán nhận diện khi bị che khuất."*
* **Speaker Note:** Nói chậm lại, dừng 2 giây ở con số **+22.68%** để hội đồng kịp nhìn bảng số liệu. Đây là điểm số quyết định điểm A của bài báo cáo.

---

### 🟢 SLIDE 10: MA TRẬN NHẦM LẪN & BÁO CÁO TỪNG CẢM XÚC (09:15 – 09:55)
* **Nội dung slide:**
  * Hình ảnh 3 Ma trận nhầm lẫn (Confusion Matrices) chuẩn hóa theo phần trăm.
  * Bảng F1-Score chi tiết của 7 cảm xúc trên tập đeo khẩu trang (1,975 ảnh):
    * `Happy`: F1 **0.8901** (Precision 94.1%)
    * `Surprise`: F1 **0.8019** (Recall 80.6%)
    * `Sad`: F1 **0.7896** (Recall 84.9%)
    * `Angry`: F1 **0.7500** (Precision 80.0%)
    * `Neutral`: F1 **0.7243** (Recall 77.9%)
    * `Fear`: F1 **0.5333** (Precision 64.0%)
    * `Disgust`: F1 **0.4670** (Recall 45.1%)
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Đi sâu vào phân tích ma trận nhầm lẫn của 1,975 ảnh đeo khẩu trang:
  > * *Cảm xúc Vui vẻ (`happy`) đạt F1 cực cao **0.8901** với Precision lên tới **94.1%**. Dù miệng bị che kín, mô hình vẫn nhận diện chuẩn xác thông qua cơ vòng mi mắt.*
  > * *Ngạc nhiên (`surprise`) và Buồn bã (`sad`) đạt F1 xấp xỉ **0.80**, bắt trọn độ mở to của mí mắt và đầu lông mày chùng xuống.*
  > * *Tức giận (`angry`) đạt F1 **0.7500**, nhận diện xuất sắc cơ cau mày và nếp nhăn trán.*
  > * *Hai lớp gặp nhiều thách thức nhất là `fear` (F1 0.53) và `disgust` (F1 0.47). Về mặt sinh lý học thần kinh, cảm xúc Ghê tởm biểu hiện chủ yếu ở việc nhăn mũi và nhếch mép — vốn là 2 vùng bị khẩu trang che mất hoàn toàn. Tuy nhiên, mức F1 0.47 đã là một bước tiến vượt bậc so với con số 0.33 của mô hình không có Mask-Aware."*
* **Speaker Note:** Thẳng thắn chỉ ra lý do tự nhiên tại sao Disgust/Fear khó nhất $\to$ Thể hiện sự hiểu biết sâu sắc về bài toán chứ không chỉ khoe số đẹp.

---

### 🟢 SLIDE 11: BÓC TÁCH GIÁ TRỊ CẢI TIẾN (ABLATION STUDY) (09:55 – 10:30)
* **Nội dung slide:**
  * Bảng Ablation Study bóc tách từng module cải tiến trên tập RAF-DB Unmasked test:

| Bước cải tiến kỹ thuật | Accuracy | Macro F1 | Đóng góp kỹ thuật |
| :--- | :---: | :---: | :--- |
| 1. MobileNetV3 (Pipeline cũ) | 79.86% | 0.718 | Baseline ban đầu |
| 2. + HSEmotion EfficientNet-B0 (fp32) | 84.22% | 0.764 | **+4.36%** nhờ Face-Pretrained weights |
| 3. + YuNet Căn chỉnh 5 landmark & TTA | 86.32% | 0.785 | **+2.10%** nhờ triệt tiêu độ lệch hình học |
| 4. + Logit Adjustment & Calibrate Prior-bias | **89.15%** | **0.833** | **+2.83%** nhờ cân bằng biên phân loại |

* **Lời thoại trực tiếp (Speaking Script):**
  > *"Để hội đồng thấy rõ giá trị thực sự của từng thành phần kỹ thuật, nhóm đã thực hiện nghiên cứu bóc tách (Ablation Study):
  > * *Nếu chỉ dùng MobileNetV3 thông thường: Accuracy dừng lại ở 79.86%.*
  > * *Khi chuyển sang HSEmotion EfficientNet-B0: Độ chính xác tăng vọt lên 84.22% (+4.36%), chứng minh tầm quan trọng của backbone tiền huấn luyện trên khuôn mặt.*
  > * *Bổ sung bước YuNet căn chỉnh hình học và TTA lật ngang: Tăng thêm +2.10%.*
  > * *Và cuối cùng, khi kết hợp Logit-Adjusted Loss với bước Calibrate Prior-bias: Accuracy chạm mốc kỷ lục **89.15%** và Macro F1 đạt **0.833**.*
  > 
  > *Mọi bước đi của nhóm đều có bằng chứng định lượng rõ ràng. Vậy mô hình đã học được gì và làm sao vận hành thời gian thực trên camera? Em xin nhường lời cho bạn [Tên Speaker 3] trình bày về Explainable AI và Hệ thống Real-Time!"*
* **Speaker Note:** Chuyển giao micro mượt mà sang Speaker 3.

---

## 🛡️ 3. BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU DÀNH CHO SPEAKER 2

Dưới đây là 4 câu hỏi học thuật hóc búa nhất mà hội đồng sẽ nhắm vào Speaker 2:

### ❓ Q1: Cơ chế toán học của Logit-Adjusted Loss là gì? Nó khác gì Weighted Cross-Entropy thông thường?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, sự khác biệt mang tính bản chất nằm ở cơ chế phân tách biên (Margin Separation):*
  > * *Weighted Cross-Entropy chỉ nhân thêm một trọng số phạt lớn hơn vào hàm mất mát khi mạng dự đoán sai lớp thiểu số. Điều này dễ dẫn đến việc làm méo mó độ lớn gradient và khiến mô hình bị nhạy cảm quá mức với nhiễu.*
  > * *Logit-Adjusted Loss (theo nghiên cứu của Menon et al., ICLR 2021) dựa trên lý thuyết quyết định Bayes: Nó can thiệp trực tiếp vào logits trước khi tính Softmax: $f_c^*(x) = f_c(x) + \tau \log \pi_c$, trong đó $\pi_c$ là tần suất tiên nghiệm của lớp $c$.*
  > * *Về mặt hình học, điều này ép mạng nơ-ron phải dịch chuyển ranh giới quyết định (Decision Boundary) ra xa các lớp thiểu số, tạo ra một lề phân loại (margin) tỷ lệ thuận với $\log(1/\pi_c)$. Nhờ đó, mạng tối đa hóa khoảng cách phân loại một cách tự nhiên và ổn định hơn Weighted CE rất nhiều ạ."*

---

### ❓ Q2: Tại sao phải chọn $t$ trong `calibrate.py` trên tập Validation mà không chỉnh trực tiếp trên tập Test?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, việc chọn $t$ trên tập Test là hoàn toàn vi phạm nguyên tắc cơ bản của Học máy (Test Set Contamination / Data Snooping):*
  > * *Tập Test là tập dữ liệu ẩn dùng để kiểm thử độc lập độ tổng quát hóa của mô hình đối với dữ liệu chưa từng thấy.*
  > * *Nếu ta dò tìm $t$ trên tập Test để chọn ra con số F1 cao nhất, ta đang vô tình tối ưu hóa siêu tham số trên chính tập kiểm thử, dẫn đến việc kết quả báo cáo bị thổi phồng ảo.*
  > * *Trong quy trình của nhóm, thuật toán quét $t$ trên tập **Validation** (3,002 ảnh chia theo nhóm không rò rỉ). Sau khi thuật toán xác định điểm tối ưu là $t = 0.85$, giá trị $t$ này được đóng băng và áp dụng thẳng vào tập Test độc lập. Kết quả trên tập Test tăng vọt từ 73.2% lên 79.29% chứng minh rằng vector hiệu chỉnh tìm được từ Validation có tính tổng quát hóa thực sự ạ."*

---

### ❓ Q3: Tại sao nhóm không dùng Vision Transformer (ViT) như Swin Transformer mà lại chọn EfficientNet-B0?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, nhóm đã cân nhắc bài toán tối ưu đa mục tiêu giữa Độ chính xác (Accuracy), Tài nguyên tính toán và Tốc độ thời gian thực:*
  > * *1. **Độ phức tạp tính toán:** Vision Transformer dựa trên cơ chế Self-Attention có chi phí tính toán $\mathcal{O}(N^2)$ theo số lượng patches. Khi triển khai bài toán quét nhiều khuôn mặt cùng lúc (Multi-Face) trên camera, ViT sẽ ngốn rất nhiều VRAM và khó đạt được tốc độ 30+ FPS trên các thiết bị biên hoặc laptop không có GPU khủng.*
  > * *2. **Hiệu quả của Face-Domain Pretraining:** EfficientNet-B0 chỉ có ~4 triệu tham số nhưng đã được huấn luyện chuyên biệt trên VGGFace2 và AffectNet. Trong bài toán FER, một mạng CNN nhỏ gọn nhưng mang trọng số miền khuôn mặt luôn hội tụ tốt hơn và ít bị overfit hơn một mạng ViT hàng chục triệu tham số pre-train trên đồ vật thông thường.*
  > * *Với EfficientNet-B0, nhóm đạt tới 88.4% accuracy trên mặt thường, 79.3% trên mặt khẩu trang và vận hành mượt mà 55 FPS, hoàn toàn đáp ứng mục tiêu đề tài ạ."*

---

### ❓ Q4: Tại sao lúc huấn luyện nhóm dùng fp16 (AMP) nhưng lúc đánh giá benchmark bắt buộc phải dùng fp32?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, đây là một phát hiện thực nghiệm rất thú vị của nhóm:*
  > * *Lúc huấn luyện, PyTorch AMP (Automatic Mixed Precision - fp16) giúp mô hình tính toán ma trận nhanh hơn gấp 2.5 lần và giảm 50% dung lượng VRAM mà không ảnh hưởng tới xu hướng cập nhật gradient.*
  > * *Tuy nhiên, khi đánh giá (evaluation), các phép tính Softmax và việc so sánh xác suất giữa các lớp thiểu số rất nhạy cảm với sai số làm tròn số học (numerical precision).*
  > * *Thực nghiệm trên cùng một checkpoint: chạy dưới fp16 autocast làm accuracy bị giảm mất **gần 2 điểm phần trăm** (82.37% so với 84.22% fp32 trên RAF-DB test). Do đó, để đảm bảo tính chuẩn xác và công bằng tuyệt đối cho báo cáo khoa học, mọi kết quả benchmark của nhóm đều được đo đạc ở chuẩn dấu phẩy động 32-bit (fp32) ạ."*

---

## 📋 4. CHECKLIST DÀNH RIÊNG CHO SPEAKER 2 TRƯỚC GIỜ LÊN SÀN
* [ ] Nhớ chính xác 3 con số vàng: **88.43%** (Unmasked), **79.29%** (Masked - tăng **+22.68%**), **84.85%** (Combined).
* [ ] Thuộc lòng giá trị tối ưu: **$t = +0.85$** (cho Mask-Aware) và **$t = +0.70$** (cho model RAF-DB).
* [ ] Nắm chắc công thức và ý nghĩa của **Logit-Adjusted Loss**, **EMA decay 0.998**, **Cosine Annealing**.
* [ ] Canh đúng 5.5 phút: Nhìn đồng hồ bấm giờ ở phút 10:15 để chuẩn bị câu chuyển tiếp sang Speaker 3.
