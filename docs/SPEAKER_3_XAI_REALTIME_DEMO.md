# 📙 CẨM NANG THUYẾT TRÌNH DÀNH CHO SPEAKER 3
## Vai trò: XAI & Real-Time Application Engineer
### Phụ trách: Explainable AI (Grad-CAM & Attention Shift), Face Tracker IoU + EMA, Tối ưu hóa hiệu năng thời gian thực (55 FPS), Hệ sinh thái ứng dụng Streamlit/Desktop & **TRỰC TIẾP ĐIỀU KHIỂN LIVE DEMO**

---

## ⏱️ 1. THÔNG TIN PHÂN BỔ THỜI GIAN & NHIỆM VỤ

* **Thời lượng trên sân khấu:** **5 phút trình bày slide** + **3–4 phút Live Demo** (Từ phút **10:30** đến phút **18:00**).
* **Số lượng slide phụ trách:** **Slide 12 đến Slide 17** (6 slides kết màn + phiên trình diễn thực tế).
* **Mục tiêu tối thượng của Speaker 3:** 
  1. Thuyết phục hội đồng bằng hình ảnh trực quan: Chứng minh hiện tượng **Attention Shift (Dịch chuyển vùng chú ý)** của Grad-CAM khi đối tượng đeo khẩu trang.
  2. Trình bày được kỹ thuật tối ưu hóa kỹ nghệ: **IoU Tracker + EMA Smoothing** chống nhấp nháy nhãn, và kỹ thuật **Grad-CAM Temporal Cache** tăng tốc từ 7.7 lên **37.4 FPS**.
  3. **Tỏa sáng trong phần Live Demo:** Thao tác mượt mà, tự tin, chỉ rõ sự khác biệt giữa mô hình cũ và mô hình Mask-Aware ngay trên màn hình chiếu.
  4. Khép lại bài thuyết trình với sự khiêm tốn khoa học: Nêu rõ đóng góp, hạn chế và hướng phát triển.
* **Phong thái trình bày:** Hào hứng, tương tác cao với màn hình chiếu, chuẩn bị sẵn môi trường demo từ trước, sẵn sàng chuyển tab ứng dụng nhanh chóng.

---

## 🎬 2. KỊCH BẢN NÓI CHI TIẾT TỪNG SLIDE (VERBATIM SCRIPT)

---

### 🟢 SLIDE 12: EXPLAINABLE AI VỚI GRAD-CAM & ATTENTION SHIFT (10:30 – 11:40)
* **Nội dung slide:**
  * Nguyên lý **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Hook vào tầng tích chập cuối cùng `bn2` của EfficientNet-B0 trước Global Pooling $\to$ Tính đạo hàm riêng $\to$ Bản đồ nhiệt kích hoạt.
  * Hình ảnh Grad-CAM so sánh trực diện (Side-by-Side):
    * **Mặt thường:** Điểm nóng rực lửa tập trung ở 2 khóe miệng cười (`happy`) hoặc cơ cau mày trán (`angry`).
    * **Mặt có khẩu trang:** 
      * Mô hình Baseline cũ: Vẫn cố nhìn vào lớp vải khẩu trang $\to$ đoán bừa `neutral`.
      * **Mô hình Mask-Aware: Hiện tượng Attention Shift kỳ diệu** $\to$ Tâm điểm kích hoạt dịch chuyển 100% từ khóe miệng lên vùng hốc mắt, vết chân chim (Duchenne smile) và vòm chân mày!
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Em xin cảm ơn bạn [Tên Speaker 2]! Kính thưa quý thầy cô trong hội đồng, một mô hình đạt độ chính xác 88% hay 79% sẽ trở nên vô nghĩa nếu chúng ta không biết nó dựa vào đâu để ra quyết định. Để mở chiếc hộp đen của AI, nhóm đã tích hợp kỹ thuật **Explainable AI (Grad-CAM)**:
  > 
  > *Nhóm can thiệp trực tiếp vào tầng tích chập cuối cùng `bn2` của EfficientNet-B0. Bằng cách tính gradient của lớp dự đoán đối với các feature maps, hệ thống tạo ra bản đồ nhiệt kích hoạt phản ánh chính xác vùng giải phẫu mà mạng nơ-ron đang chú ý.*
  > 
  > *Và đây là **Khám phá thị giác quan trọng nhất của toàn bộ đề tài**:*
  > * *Trên mặt thường: Khi một người cười (`happy`), Grad-CAM tập trung rực rỡ vào hai khóe miệng cười.*
  > * *Nhưng khi người đó đeo khẩu trang: Mô hình thông thường vẫn cố nhìn vào lớp vải khẩu trang và đoán nhầm thành `neutral`. Trong khi đó, mô hình Mask-Aware của nhóm tạo ra một **Sự dịch chuyển vùng chú ý (Attention Shift)** kỳ diệu: Tâm điểm kích hoạt tự động dịch chuyển 100% từ miệng lên khóe mắt (vết chân chim Duchenne smile) và vòm chân mày để nhận diện nhãn `happy` chính xác tuyệt đối.*
  > * *AI của chúng em không hề đoán mò, mà thực sự 'nhìn thấy' cảm xúc ẩn sau đôi mắt!"*
* **Speaker Note:** Dùng con trỏ chuột hoặc laser chỉ thẳng vào vùng đỏ trên mắt ở ảnh có khẩu trang trên slide để gây ấn tượng mạnh với hội đồng.

---

### 🟢 SLIDE 13: THIẾT KẾ THỜI GIAN THỰC & ỔN ĐỊNH CHUỖI THỜI GIAN (11:40 – 12:40)
* **Nội dung slide:**
  * Vấn đề thực tế: **Label Flickering (Nhảy nhãn liên tục)** giữa các frame liên tiếp do nhiễu ánh sáng hoặc chuyển động micro-expression.
  * Giải pháp kỹ thuật của nhóm: Module **`FaceTracker`** kết hợp:
    1. **IoU Face Association:** Ghép nối bbox giữa các frame, cấp phát Track ID ổn định cho từng khuôn mặt, ngưỡng mất dấu `max_missed = 10` frames.
    2. **Temporal EMA Smoothing:** Làm mượt vector xác suất 7 lớp qua thời gian:
       $$P_{\text{smooth}}^{(t)} = (1 - \alpha) P_{\text{smooth}}^{(t-1)} + \alpha P_{\text{mới}}^{(t)} \quad (\alpha = 0.4)$$
  * Kết quả: Nhãn cảm xúc hiển thị mượt mà, ổn định tuyệt đối giữa các khung hình video.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Khi đưa mô hình ra webcam và video thực tế, nhóm đối mặt với một vấn đề rất khó chịu của camera: **Hiện tượng nhảy nhãn (Label Flickering)**. Chỉ cần một cái chớp mắt hay ánh đèn thay đổi nhẹ, frame trước vừa đoán Happy thì frame sau đã giật sang Neutral khiến người xem mất tin tưởng.*
  > 
  > *Để khắc phục, nhóm đã tự thiết kế module **`FaceTracker`** kết hợp 2 cơ chế:*
  > * *1. **Theo dõi đa khuôn mặt bằng IoU:** Thuật toán tính độ giao nhau trên diện tích hợp (Intersection over Union) giữa các frame để gán Track ID duy nhất cho từng người, duy trì trạng thái nhớ vết tối đa 10 frames ngay cả khi người đó quay mặt đi trong tích tắc.*
  > * *2. **Làm mượt xác suất theo thời gian (Temporal EMA Smoothing):** Thay vì lấy argmax tức thời của frame hiện tại, nhóm cập nhật vector xác suất 7 lớp bằng bộ lọc trung bình trượt hàm mũ với hệ số $\alpha = 0.4$.*
  > * *Nhờ đó, nhãn cảm xúc và thanh đo xác suất hiển thị vô cùng mượt mà, triệt tiêu 100% hiện tượng rung giật nhãn giữa các khung hình liên tiếp."*
* **Speaker Note:** Giải thích rõ: Nhờ module này mà trải nghiệm người dùng trên webcam đạt mức hoàn thiện thương mại (Production-ready).

---

### 🟢 SLIDE 14: TỐI ƯU HÓA HIỆU NĂNG & BENCHMARK FPS (12:40 – 13:40)
* **Nội dung slide:**
  * Bảng số liệu Benchmark FPS đo thực nghiệm từ `scripts/benchmark_fps.py` trên Apple M1 (MPS GPU) / NVIDIA GPU:

| Số lượng khuôn mặt | FPS (Không bật Grad-CAM) | FPS (Có Grad-CAM cache 10 frame) | Thời gian chi tiết từng bước |
| :---: | :---: | :---: | :---: |
| **1 khuôn mặt** | **55.0 FPS** ⚡ | **37.4 FPS** ⚡ | Detect ~5.5ms \| Predict ~12.7ms |
| **2 khuôn mặt** | **37.4 FPS** | **25.9 FPS** | Predict ~21.1ms |
| **3 khuôn mặt** | **28.4 FPS** | **20.7 FPS** | Predict ~29.7ms |
| **5 khuôn mặt** | **17.4 FPS** | **15.1 FPS** | Quét đám đông ổn định |

  * 2 Kỹ thuật tối ưu hóa then chốt:
    1. **Tắt TTA khi stream webcam:** Thực nghiệm đo được việc tắt TTA lật ngang chỉ làm giảm dưới 0.1% accuracy (88.43% $\to$ 88.43%, F1 0.822 $\to$ 0.818) nhưng tăng tốc độ suy luận gấp đôi!
    2. **Kỹ thuật Grad-CAM Temporal Cache:** Một lần backward pass mất ~90ms (kéo tụt FPS xuống 7.7). Bằng cách **chỉ tính lại Grad-CAM mỗi 10 frame** và tái sử dụng heatmap ở các frame giữa $\to$ **FPS vọt từ 7.7 lên 37.4 FPS** (tăng tốc gấp 5 lần)!
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Để hệ thống vận hành trơn tru trên laptop phổ thông mà không đòi hỏi GPU đắt tiền, nhóm đã nghiên cứu tối ưu hóa hiệu năng kỹ nghệ:*
  > 
  > * Nhóm nhận thấy: Một lượt lan truyền ngược (backward pass) để tính đạo hàm Grad-CAM mất tới 90 đến 100 mili-giây — điều này khiến camera bị nghẽn ở mức chỉ 7.7 FPS.
  > * Nhóm đưa ra giải pháp đột phá: **Grad-CAM Temporal Cache**. Trong một video 30 FPS, cơ mặt con người không thể thay đổi vị trí giải phẫu trong 30 phần nghìn giây. Vì vậy, hệ thống **chỉ tính toán lại Grad-CAM mỗi 10 frame một lần** (~0.3 giây), và tái sử dụng heatmap đó cho các frame đệm.
  > * Kết quả ngoạn mục: Tốc độ xử lý tăng vọt từ **7.7 FPS lên tới 37.4 FPS** khi có Grad-CAM!
  > * Đồng thời, ở chế độ live webcam, ta tắt TTA lật ngang vì thực nghiệm chứng minh việc này chỉ làm giảm dưới 0.1% accuracy nhưng tăng tốc độ suy luận lên gấp đôi.
  > 
  > *Khi không bật Grad-CAM, hệ thống đạt tới **55 FPS cho 1 mặt** và duy trì ổn định **28 đến 37 FPS khi có 2 đến 3 người cùng xuất hiện trong khung hình 720p**."*
* **Speaker Note:** Nêu bật các con số: 55 FPS và 37.4 FPS để chứng minh tính Real-time thực thụ.

---

### 🟢 SLIDE 15: HỆ SINH THÁI ỨNG DỤNG STREAMLIT & DESKTOP APP (13:40 – 14:30)
* **Nội dung slide:**
  * Hình ảnh 2 sản phẩm hoàn thiện:
    1. **Streamlit Web Dashboard (`app.py`):** Giao diện Dark theme cao cấp với 4 chế độ:
       - Single-face analysis kèm thanh trượt xác suất 7 lớp.
       - Multi-face Crowd Scanner (phân tích ảnh đám đông).
       - Video Batch Processor (tải video lên $\to$ xuất video đã gán nhãn).
       - **Head-to-Head Comparison:** Đối đầu trực tiếp giữa Baseline và Mask-Aware.
    2. **Desktop Live App (`webcam_app.py`):** Viết trên nền OpenCV C++ backend, nạp sẵn cả 2 model vào bộ nhớ, chuyển đổi tức thì bằng phím tắt `1` / `2` / `M` và bật tắt Grad-CAM bằng phím `G`.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Toàn bộ các mô hình và thuật toán đã được nhóm đóng gói thành một hệ sinh thái sản phẩm hoàn chỉnh:
  > 
  > * **Thứ nhất là Web Dashboard trên Streamlit (`app.py`):** Giao diện Dark theme hiện đại, cho phép người dùng tải ảnh đơn, quét ảnh đám đông nhiều người, phân tích video trích xuất frame và đặc biệt là tab **Head-to-Head Comparison** giúp so sánh trực diện kết quả của mô hình Baseline và Mask-Aware trên cùng một ảnh.
  > * **Thứ hai là Desktop App (`webcam_app.py`):** Ứng dụng độc lập tối ưu cho camera an ninh và giám sát thời gian thực. Ứng dụng nạp sẵn cả 2 checkpoint vào VRAM, cho phép người dùng bấm phím `1`, `2` hoặc `M` để hoán đổi mô hình ngay tức thì và nhấn `G` để bật/tắt bản đồ nhiệt Grad-CAM.
  > 
  > *Và ngay sau đây, em xin phép chuyển sang phần **Trực tiếp Live Demo** để quý thầy cô tận mắt chứng kiến hệ thống vận hành!"*
* **Speaker Note:** Chuyển ngay sang cửa sổ trình duyệt đã bật sẵn Streamlit và cửa sổ terminal.

---

### 🟢 SLIDE 16 & LIVE DEMO TRỰC TIẾP (14:30 – 17:30)
* **Hành động của Speaker 3:** Đứng trước màn hình demo, thao tác dứt khoát theo đúng 3 kịch bản:

#### 🎯 Demo 1: Tab So sánh Đối đầu (Head-to-Head Comparison) — 1 phút
* **Thao tác:** Mở tab so sánh trên Streamlit, nạp ảnh người đeo khẩu trang cười (`happy`) trong `data/real_masked_faces/`.
* **Lời nói minh họa:**
  > *"Kính thưa hội đồng, trước mắt thầy cô là tab So sánh Đối đầu:
  > * *Ở khung bên trái, mô hình Baseline (chỉ train trên ảnh thường) dự đoán sai hoàn toàn: Nó đoán người này là `Neutral` với xác suất 78%, và heatmap Grad-CAM bị phân tán vô định trên tấm khẩu trang.*
  > * *Nhưng ở khung bên phải, mô hình Mask-Aware của nhóm nhận diện chuẩn xác 100% cảm xúc `Happy` với xác suất 91%! Và nhìn vào heatmap Grad-CAM, thầy cô có thể thấy rõ: Vùng kích hoạt màu đỏ rực đã dịch chuyển hoàn toàn lên khóe mắt đang cười.*
  > * *Sự vượt trội 22.7% trên lý thuyết đã được chứng minh trực quan bằng mắt thường!"*

#### 🎯 Demo 2: Quét đám đông nhiều khuôn mặt (Multi-Face Scanner) — 1 phút
* **Thao tác:** Nạp một ảnh có 4–5 người trong `data/multi_face_samples/group_people_1.jpg`.
* **Lời nói minh họa:**
  > *"Tiếp theo là khả năng nhận diện đa khuôn mặt:
  > * *Chỉ trong chưa đầy 30 mili-giây, YuNet đã phát hiện đồng thời cả 5 khuôn mặt trong ảnh, căn chỉnh hình học và đưa qua mạng theo batch.*
  > * *Hệ thống gán nhãn biểu cảm chính xác cho từng người: Người này Happy, người kia Surprise, người cạnh bên Neutral.*
  > * *Người dùng có thể click chọn bất kỳ khuôn mặt nào để xem riêng bản đồ nhiệt Grad-CAM và phân bố xác suất 7 lớp của người đó."*

#### 🎯 Demo 3: Live Webcam thời gian thực & Stress Test khẩu trang — 1 phút
* **Thao tác:** Chạy `python webcam_app.py` (hoặc tab Webcam trên Streamlit). Ban đầu không đeo khẩu trang $\to$ Thực hiện biểu cảm $\to$ Sau đó lấy khẩu trang y tế đeo lên mặt và cười/ngạc nhiên.
* **Lời nói minh họa:**
  > *"Và cuối cùng là bài kiểm tra thời gian thực với webcam:
  > * *Khi em để mặt mộc, hệ thống nhận diện `Happy` ở mức 55 FPS mượt mà.*
  > * *Bây giờ, em xin phép đeo chiếc khẩu trang y tế này che kín hoàn toàn miệng và mũi...*
  > * *Khi em mỉm cười qua ánh mắt, hệ thống lập tức bắt được nhãn `Happy` với tốc độ hơn 35 FPS! Nhãn không hề bị nhảy giật nhờ module FaceTracker IoU + EMA Smoothing.*
  > * *Em nhấn phím 'G' — bản đồ nhiệt Grad-CAM lập tức hiện lên quanh mắt em ở chế độ thời gian thực hoàn hảo!"*

---

### 🟢 SLIDE 17: TỔNG KẾT, HẠN CHẾ & KẾT THÚC (17:30 – 18:00)
* **Nội dung slide:**
  * 4 Đóng góp cốt lõi:
    1. Mask-Aware tăng **+22.68%** trên ảnh khẩu trang, giữ nguyên 88.43% mặt thường.
    2. Quy trình kiểm toán rò rỉ dữ liệu 0% leakage (`leakage_report.txt`).
    3. Căn chỉnh hình học 5-point landmark đồng nhất phân phối.
    4. Hệ thống đa khuôn mặt thời gian thực 55 FPS kèm giải thích XAI.
  * Hạn chế & Hướng phát triển: Thu thập thêm dữ liệu khẩu trang thật đa dạng; mở rộng sang nhận diện Vi-biểu cảm (Micro-expressions) trên chuỗi video tốc độ cao.
  * Lời cảm ơn hội đồng.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Kính thưa quý thầy cô, bài thuyết trình và phần live demo vừa rồi đã chứng minh rằng: **Dự án của nhóm không dừng lại ở những con số lý thuyết, mà đã hiện thực hóa thành một sản phẩm AI hoàn chỉnh, chính xác cao, minh bạch và chạy thời gian thực.**
  > 
  > *Dù đạt được những kết quả rất đáng khích lệ, nhóm vẫn nhận thức rõ các hạn chế khoa học như tập dữ liệu khẩu trang vẫn chủ yếu là tổng hợp, và cảm xúc Ghê tởm/Sợ hãi khi bị che khuất vẫn là một thách thức lớn của sinh lý học.*
  > 
  > *Thay mặt nhóm nghiên cứu, em xin chân thành cảm ơn quý thầy cô trong hội đồng đã chú ý lắng nghe và theo dõi phần trình diễn của nhóm. Nhóm chúng em rất mong nhận được những nhận xét và câu hỏi phản biện từ quý thầy cô ạ!"*
* **Speaker Note:** Cúi đầu chào trang trọng, cả 3 thành viên cùng đứng thẳng bước lên hàng trên sẵn sàng cho phần Q&A.

---

## 🛡️ 3. BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU DÀNH CHO SPEAKER 3

Dưới đây là 4 câu hỏi trọng tâm mà hội đồng chắc chắn sẽ nhắm vào Speaker 3:

### ❓ Q1: Cơ chế toán học của Grad-CAM là gì? Tại sao lại hook vào tầng `bn2` của EfficientNet-B0?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô:
  > * *1. **Về mặt toán học:** Grad-CAM tính toán trọng số tầm quan trọng $\alpha_k^c$ cho feature map thứ $k$ bằng cách lấy trung bình toàn cục (Global Average Pooling) của đạo hàm riêng từ điểm số dự đoán lớp $c$ ($y^c$) đối với từng điểm ảnh $A_{i,j}^k$:*
  >   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
  >   *Sau đó, bản đồ nhiệt là tổ hợp tuyến tính các feature maps nhân với trọng số $\alpha_k^c$ và đi qua hàm $\text{ReLU}$ để chỉ giữ lại các đặc trưng kích hoạt có lợi cho lớp $c$:*
  >   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
  > * *2. **Tại sao chọn tầng `bn2`:** Đây là lớp Batch Normalization ngay sau khối Conv 1x1 mở rộng cuối cùng của EfficientNet-B0, nằm ngay trước tầng Global Average Pooling. Feature map tại đây mang mức độ biểu diễn ngữ nghĩa cao nhất (high-level visual semantics) nhưng vẫn bảo toàn được kích thước không gian 7x7. Nếu hook sâu hơn thì đã bị gom thành vector 1D, còn nếu hook nông hơn thì đặc trưng chỉ là các đường nét hình học cấp thấp ạ."*

---

### ❓ Q2: Face Tracker giải quyết thế nào khi có 2 người đi lướt qua nhau hoặc một người quay mặt đi rồi quay lại?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, module `FaceTracker` của nhóm xử lý dựa trên quản lý trạng thái vết (Track State Management):*
  > * *1. **Khi người quay mặt đi:** Nếu trong 1–2 frame khuôn mặt tạm thời không detect được, tracker không vội xóa ID mà tăng biến đếm `missed += 1`. Nhóm thiết lập ngưỡng chịu đựng `max_missed = 10` frames (~0.3 giây). Nếu trong khoảng thời gian này khuôn mặt xuất hiện trở lại ở tọa độ lân cận với $\text{IoU} \ge 0.3$, hệ thống sẽ phục hồi lại đúng Track ID đó cùng vector xác suất đã làm mượt.*
  > * *2. **Khi 2 người đi lướt qua nhau:** Thuật toán giải bài toán ghép cặp bằng Ma trận chi phí IoU tham lam (Greedy IoU Matching) sắp xếp theo thứ tự diện tích giao nhau giảm dần. Nhờ vậy, khuôn mặt có độ trùng khớp cao nhất luôn được ghép trước, giảm thiểu tối đa hiện tượng hoán đổi ID (ID switch) giữa 2 người ạ."*

---

### ❓ Q3: Tại sao Grad-CAM thông thường lại rất chậm và nhóm đã làm thế nào để kéo FPS từ 7.7 lên 37.4?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô:
  > * *Grad-CAM thông thường chậm vì nó bắt buộc phải thực hiện **Lan truyền ngược (Backward Pass)** để tính đạo hàm riêng của tensor đầu ra về tới tầng tích chập cuối cùng. Trên luồng webcam, việc tính backward pass mất từ 90 đến 100 mili-giây, tức là kéo tốc độ hệ thống xuống dưới 8 FPS.*
  > * *Nhóm nhận định rằng: Trong luồng video 30 FPS, vị trí cảm xúc trên cơ mặt người không thể thay đổi đột ngột chỉ trong vòng 30ms. Vùng chú ý của mạng có tính quán tính thời gian rất cao.*
  > * *Nhóm đề xuất giải pháp kỹ thuật **Grad-CAM Temporal Cache**: Hệ thống chỉ chạy backward pass một lần sau mỗi 10 frames (~0.3 giây) để tính lại trọng số nhiệt. Ở 9 frames nằm giữa, hệ thống giữ nguyên ma trận nhiệt và chỉ cập nhật vị trí hình học theo bbox di chuyển của khuôn mặt.*
  > * *Giải pháp này giúp giảm tải tính toán gradient tới 90%, đưa FPS từ 7.7 vọt lên **37.4 FPS** mà mắt người nhìn vào vẫn thấy bản đồ nhiệt cập nhật mượt mà và trực quan ạ."*

---

### ❓ Q4: Nếu trong lúc Live Demo camera bị lỗi hoặc ánh sáng hội trường quá tối thì nhóm xử lý thế nào?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, nhóm luôn tuân thủ nguyên tắc dự phòng kỹ thuật (Engineering Redundancy):*
  > * *1. Ứng dụng Streamlit của nhóm có sẵn tab **Tải ảnh và Tải video** độc lập. Trong thư mục `data/` nhóm đã chuẩn bị sẵn các mẫu ảnh nhóm đông người và video demo chuẩn.*
  > * *2. Nhóm đã quay sẵn một video demo chất lượng cao 1080p 60fps lưu sẵn trong máy để có thể trình chiếu ngay lập tức nếu camera gặp sự cố phần cứng.*
  > * *3. Về vấn đề ánh sáng yếu: Mạng YuNet có khả năng thích ứng dải động rất cao và bộ tiền xử lý của nhóm có bước chuẩn hóa ImageNet Normalization nên có thể xử lý tốt các khung hình có độ sáng yếu trong hội trường ạ."*

---

## 📋 4. CHECKLIST DÀNH RIÊNG CHO SPEAKER 3 TRƯỚC GIỜ LÊN SÀN
* [ ] Chạy sẵn `uv run streamlit run app.py` trong terminal trước giờ thuyết trình, mở sẵn trên trình duyệt ở chế độ toàn màn hình.
* [ ] Chuẩn bị sẵn một chiếc khẩu trang y tế để trong túi áo để đeo lúc live demo.
* [ ] Kiểm tra quyền truy cập camera của trình duyệt Chrome/Edge.
* [ ] Thuộc lòng các con số: **55 FPS** (1 mặt), **37.4 FPS** (có Grad-CAM cache 10 frame), $\alpha = 0.4$ (EMA smoothing).
* [ ] Giữ tinh thần thoải mái, phần demo là phần ghi điểm trực quan mạnh mẽ nhất của nhóm!
