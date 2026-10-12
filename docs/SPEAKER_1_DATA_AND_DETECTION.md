# 📘 CẨM NANG THUYẾT TRÌNH DÀNH CHO SPEAKER 1
## Vai trò: Data & Detection Pipeline Engineer
### Phụ trách: Đặt vấn đề, Dữ liệu RAF-DB, MaskTheFace, Kiểm toán rò rỉ dữ liệu (Leak-Free Audit) & Căn chỉnh khuôn mặt (YuNet Alignment)

---

## ⏱️ 1. THÔNG TIN PHÂN BỔ THỜI GIAN & NHIỆM VỤ

* **Thời lượng trên sân khấu:** **5 phút** (Từ phút **00:00** đến phút **05:00**).
* **Số lượng slide phụ trách:** **Slide 1 đến Slide 5** (5 slides mở màn).
* **Mục tiêu tối thượng của Speaker 1:** 
  1. Tạo ấn tượng mở màn tự tin, cuốn hút bằng một Elevator Pitch đanh thép.
  2. Nêu bật được tính cấp thiết của bài toán (In-the-wild gap và Mask occlusion).
  3. Khẳng định tính **liêm chính học thuật (Scientific Integrity)** qua việc giải quyết triệt để rò rỉ dữ liệu (Data Leakage) — đây là điểm ăn điểm rất lớn trước hội đồng.
  4. Trình bày được giá trị kỹ thuật của bước **YuNet 5-point alignment** đối với việc đồng nhất phân phối dữ liệu.
* **Phong thái trình bày:** Đĩnh đạc, rõ ràng, phát âm chuẩn các thuật ngữ AI tiếng Anh (In-the-wild, Leakage, Landmark, Alignment), thể hiện sự làm chủ dữ liệu của dự án.

---

## 🎬 2. KỊCH BẢN NÓI CHI TIẾT TỪNG SLIDE (VERBATIM SCRIPT)

---

### 🟢 SLIDE 1: TRANG TIÊU ĐỀ & MỞ MÀN (00:00 – 00:45)
* **Nội dung slide:**
  * Tên đề tài: *Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Occlusions via Pre-Trained Deep Models (XAI-FER Mask-Aware)*.
  * Logo trường, GVHD, danh sách 3 thành viên kèm ảnh và phân vai chuyên môn:
    * **Speaker 1 [Tên bạn]:** Data & Detection Pipeline Engineer.
    * **Speaker 2 [Tên bạn]:** Deep Learning & Evaluation Engineer.
    * **Speaker 3 [Tên bạn]:** XAI & Real-Time Application Engineer.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Kính thưa quý thầy cô trong hội đồng và các bạn sinh viên!*
  > 
  > *Em là [Tên Speaker 1], đại diện cho nhóm nghiên cứu thực hiện đề tài: **Hệ thống Nhận diện Cảm xúc Đa khuôn mặt Thời gian thực Kết hợp Giải thích AI trong Điều kiện Che khuất (XAI-FER Mask-Aware)**.*
  > 
  > *Đồng hành cùng em hôm nay có bạn [Tên Speaker 2] — phụ trách Mô hình hóa Deep Learning & Đánh giá định lượng, và bạn [Tên Speaker 3] — phụ trách Explainable AI & Phát triển ứng dụng thời gian thực.*
  > 
  > *Kính thưa thầy cô, biểu cảm khuôn mặt là ngôn ngữ cảm xúc tự nhiên và trung thực nhất của con người. Nhưng điều gì sẽ xảy ra khi hơn một nửa khuôn mặt bị che khuất bởi chiếc khẩu trang, giữa môi trường thực tế đông người và chuyển động liên tục?*
  > 
  > *Hệ thống của chúng em được xây dựng để giải quyết trọn vẹn bài toán đó: Đạt độ chính xác 88.4% trên mặt thường, 79.3% trên mặt đeo khẩu trang (vượt trội hơn 22.7% so với mô hình thông thường), vận hành thời gian thực 55 FPS và giải thích minh bạch: **AI đang nhìn vào đâu để đọc vị cảm xúc**."*
* **Speaker Note:** Nói to, mắt nhìn bao quát cả hội đồng, giữ nụ cười tự tin. Nhấn mạnh cụm từ **"vượt trội hơn 22.7%"**.

---

### 🟢 SLIDE 2: ĐẶT VẤN ĐỀ & BỐI CẢNH THỰC TẾ (00:45 – 01:50)
* **Nội dung slide:**
  * 3 chiếc hộp thể hiện 3 rào cản lớn:
    1. **In-The-Wild Gap:** FER-2013 ảnh xám 48x48 đơn sắc $\ne$ Đời thực góc nghiêng, ánh sáng phức tạp.
    2. **Mask Occlusion:** Khẩu trang che mất Action Units vùng miệng $\to$ Recall của Happy tụt từ 94% xuống 44%, 82% dự đoán dồn nhầm về Neutral.
    3. **Black-Box AI:** Thiếu bằng chứng thị giác khiến con người không thể tin cậy trong y tế và giáo dục.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Kính thưa hội đồng, khi khảo sát các hệ thống nhận diện cảm xúc (FER) truyền thống, nhóm nhận thấy một sự đứt gãy rất lớn giữa phòng thí nghiệm và đời thực:*
  > 
  > * **Thứ nhất là Khoảng cách In-The-Wild:** Phần lớn đồ án trước đây chỉ dừng lại ở tập dữ liệu FER-2013 với các ảnh xám tí hon 48x48 được cắt sát nhân tạo. Khi đưa ra camera thực tế có màu sắc, góc nghiêng đa dạng và điều kiện ánh sáng thay đổi, các mô hình này hoàn toàn bị sụp đổ.*
  > 
  > * **Thứ hai là Rào cản khẩu trang (Mask Occlusion):** Kể từ sau đại dịch, khẩu trang đã trở thành thói quen phổ biến tại bệnh viện, trường học và nơi công cộng. Khẩu trang y tế che mất toàn bộ nửa dưới khuôn mặt — nơi chứa các đơn vị vận động cơ mặt (Action Units) tối quan trọng như khóe miệng cười hay bờ môi trễ. Thực nghiệm của nhóm chỉ ra rằng: Nếu dùng một mô hình thông thường mang ra đoán mặt đeo khẩu trang, độ nhạy của cảm xúc Vui vẻ (`happy`) tụt dốc từ 94% xuống chỉ còn 44%, và mô hình dồn tới 82% dự đoán sai về nhãn Bình thường (`neutral`).*
  > 
  > * **Thứ ba là Vấn đề hộp đen AI:** Dự đoán một nhãn cảm xúc mà không có lời giải thích trực quan khiến bác sĩ tâm lý hay giáo viên không dám tin cậy.*
  > 
  > *Nhóm đặt ra mục tiêu: Xây dựng một pipeline chuẩn mực xử lý triệt để ảnh in-the-wild, vượt qua rào cản khẩu trang và minh bạch hóa quyết định của AI."*
* **Speaker Note:** Dùng tay chỉ vào hình ảnh so sánh mặt đeo khẩu trang để hội đồng thấy rõ vùng miệng bị che khuất.

---

### 🟢 SLIDE 3: DỮ LIỆU IN-THE-WILD & PIPELINE MASKTHEFACE (01:50 – 02:50)
* **Nội dung slide:**
  * **Bộ dữ liệu gốc RAF-DB:** 15,339 ảnh màu tự nhiên (12,271 Train / 3,068 Test chính thức).
  * Biểu đồ phân bố 7 lớp: `happy` chiếm đa số (39%), `fear` ít nhất (2.3%).
  * **Pipeline MaskTheFace:** Dò 68 facial landmarks $\to$ Ước tính góc quay 3D (Pitch, Yaw, Roll) $\to$ Áp khẩu trang phẫu thuật tổng hợp $\to$ Sinh ra 7,830 ảnh `train_masked` và 1,975 ảnh `test_masked`.
  * Tiền tố tên file giữ nguyên ID ảnh nguồn (ví dụ: `raf_train_00012_surgical.jpg`).
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Để giải quyết bài toán In-The-Wild, nhóm đã lựa chọn bộ dữ liệu chuẩn mực hàng đầu hiện nay là **RAF-DB (Real-world Affective Faces Database)** gồm 15,339 ảnh khuôn mặt tự nhiên, được gán nhãn độc lập bởi 40 chuyên gia theo 7 trạng thái cảm xúc. Trong đó có 12,271 ảnh Train và 3,068 ảnh Test chính thức.*
  > 
  > *Tuy nhiên, thách thức đặt ra là: Trong tự nhiên không có một bộ dữ liệu cảm xúc đeo khẩu trang thật nào vừa đủ lớn, vừa có nhãn chân lý được chuẩn hóa. Nếu đi chụp ảnh người đeo khẩu trang ngoài đời rồi tự gán nhãn, độ chính xác sẽ rất chủ quan.*
  > 
  > *Vì vậy, nhóm đã thiết lập một pipeline sinh khẩu trang tự động bằng công cụ **MaskTheFace**:*
  > * *Hệ thống nhận diện 68 điểm mốc hình học, ước lượng góc quay không gian 3 chiều của khuôn mặt, sau đó áp các loại khẩu trang y tế nhân tạo khớp chính xác vào sống mũi và cằm của ảnh gốc.*
  > * *Bằng cách này, nhóm tạo ra **7,830 ảnh train_masked** và **1,975 ảnh test_masked**.*
  > * *Điểm mấu chốt: Nhãn cảm xúc gốc được bảo toàn 100%, và tên file luôn giữ tiền tố nguồn để phục vụ việc kiểm soát rò rỉ dữ liệu mà em sẽ trình bày ngay sau đây."*
* **Speaker Note:** Nhấn mạnh rằng: Việc sinh khẩu trang trên ảnh gốc giúp ta có **cặp dữ liệu đối chứng hoàn hảo (paired counterfactual data)** — cùng một người lúc có và không có khẩu trang.

---

### 🟢 SLIDE 4: KHỦNG HOẢNG RÒ RỈ DỮ LIỆU & KIỂM TOÁN ĐỘC LẬP (02:50 – 03:50)
* **Nội dung slide:**
  * Sơ đồ màu đỏ (Bẫy rò rỉ của pipeline cũ): Random Split 85/15 $\to$ Bản không khẩu trang của người A ở Train, bản có khẩu trang của người A ở Val $\to$ Val F1 ảo 0.826, nhưng Test thật rớt xuống 0.718.
  * Sơ đồ màu xanh (Giải pháp của nhóm): **Subject-Aware Grouped Split** (Lưu cố định ra file JSON `val_groups_r15_s42.json`).
  * Báo cáo kiểm toán độc lập bằng thuật toán dHash (`output/leakage_report.txt`):
    * `test_masked` khớp ảnh gốc test: **1,973 / 1,975 ảnh**.
    * Nguồn thuộc split khác (`train`): **0 ảnh! (Zero leakage)**.
    * Train vs Val: **0 nhóm chung**.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Kính thưa hội đồng, đây là đóng góp phương pháp luận mà nhóm tự hào nhất về tính trung thực khoa học:*
  > 
  > *Khi kiểm tra các nghiên cứu FER kết hợp khẩu trang hiện có, nhóm phát hiện một lỗ hổng chí mạng: **Rò rỉ dữ liệu (Data Leakage)**.*
  > * *Nếu gộp ảnh gốc và ảnh khẩu trang lại rồi chia Train/Val ngẫu nhiên 85/15, thì phiên bản mặt mộc của một người nằm ở Train, còn bản đeo khẩu trang của chính người đó lại lọt sang Validation! Khi đó, mạng nơ-ron không học đặc trưng cảm xúc mà học vẹt danh tính khuôn mặt của người đó. Hậu quả là F1 trên tập Val đạt ảo tới 0.826, nhưng khi đem test trên người lạ thì sụp đổ chỉ còn 0.718.*
  > 
  > *Để chấm dứt điều này, nhóm đã tái cấu trúc toàn bộ pipeline:*
  > * *Áp dụng **Subject-Aware Grouped Split**: Toàn bộ biến thể của một người phải nằm 100% ở Train hoặc 100% ở Val, phân tầng theo nhãn và cố định bằng file JSON.*
  > * *Nhóm còn tự xây dựng script kiểm toán độc lập **`scripts/check_leakage.py`** dùng giải thuật băm nhận thức **dHash** nửa trên khuôn mặt để quét toàn bộ 25,144 ảnh.*
  > * *Kết quả kiểm toán khẳng định 100%: **Không có bất kỳ ảnh train nào bị lọt vào test_masked**, và **Train/Val có đúng 0 nhóm chung**.*
  > 
  > *Toàn bộ số liệu của nhóm là số liệu thực chứng, tuyệt đối không có ảo giác học vẹt."*
* **Speaker Note:** Nói với giọng điệu đanh thép, tự tin. Hội đồng rất thích các nhóm phát hiện và sửa được lỗi rò rỉ dữ liệu mà nhiều bài báo khoa học hay mắc phải.

---

### 🟢 SLIDE 5: PHÁT HIỆN ĐA KHUÔN MẶT & CĂN CHỈNH HÌNH HỌC 5 LANDMARK (03:50 – 05:00)
* **Nội dung slide:**
  * So sánh trực quan: Crop bounding box thường vs Căn chỉnh hình học tương đồng (Similarity Transform).
  * Kiến trúc **OpenCV YuNet**: Tốc độ >100 FPS, phát hiện đa khuôn mặt cả khi nghiêng và đeo khẩu trang.
  * Template 5 điểm mốc của RAF-DB: 2 mắt, mũi, 2 khóe miệng (Đo thực nghiệm trên 1,000 ảnh test, độ lệch chuẩn $\sim 0.03$).
  * Quy trình: Frame 720p $\to$ Thu nhỏ 640px để detect $\to$ Căn chỉnh trên frame gốc để giữ trọn độ nét $\to$ Output ảnh mặt chuẩn 224x224.
* **Lời thoại trực tiếp (Speaking Script):**
  > *"Để đưa mô hình ra thế giới thực, khâu tiền xử lý phát hiện khuôn mặt là chìa khóa then chốt:*
  > 
  > *Thay vì dùng Haar Cascade đã lỗi thời hay MediaPipe nặng nề, nhóm tích hợp **OpenCV YuNet** — một mạng CNN siêu nhẹ, vận hành với tốc độ hơn 100 FPS và bắt trọn mọi khuôn mặt bị che khuất hoặc quay góc lớn.*
  > 
  > *Tuy nhiên, phát hiện thôi là chưa đủ. Nếu chỉ crop một hình chữ nhật bao quanh mặt, khi người dùng nghiêng đầu hay camera đặt chéo, vị trí của mắt và miệng sẽ bị xoay lệch hoàn toàn so với dữ liệu huấn luyện.*
  > 
  > *Nhóm đã tích hợp kỹ thuật **Face Alignment (Căn chỉnh hình học)**:*
  > * *Dựa trên 5 điểm mốc do YuNet trả về (2 mắt, đỉnh mũi, 2 khóe miệng), nhóm dùng **Phép biến đổi tương đồng 2D (Similarity Transform)** để xoay và co dãn ảnh sao cho 5 điểm này khớp hoàn hảo vào **Template tọa độ chuẩn của RAF-DB**.*
  > * *Template này được nhóm đo thực nghiệm trên 1,000 ảnh RAF-DB với độ lệch chuẩn cực nhỏ chỉ 0.03.*
  > * *Nhờ đó, ảnh crop từ webcam thời gian thực có cùng phân phối không gian với ảnh lúc train, loại bỏ hoàn toàn độ lệch hình học.*
  > 
  > *Sau khi dữ liệu đã được làm sạch, kiểm toán không rò rỉ và căn chỉnh chuẩn mực, mô hình học sâu sẽ xử lý như thế nào? Em xin kính mời bạn [Tên Speaker 2] tiếp tục phần trình bày về Kiến trúc mô hình và Kết quả thực nghiệm đột phá của nhóm!"*
* **Speaker Note:** Chuyển giao micro mượt mà: Quay sang Speaker 2, gật đầu và nhường bục thuyết trình.

---

## 🛡️ 3. BỘ CÂU HỎI PHẢN BIỆN CHUYÊN SÂU DÀNH CHO SPEAKER 1

Dưới đây là 4 câu hỏi trọng tâm mà hội đồng chắc chắn sẽ nhắm vào Speaker 1 kèm câu trả lời đanh thép:

### ❓ Q1: Tại sao không thu thập ảnh người thật đeo khẩu trang thật ngoài đời để huấn luyện mà lại dùng ảnh tổng hợp (MaskTheFace)?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, nhóm chọn ảnh tổng hợp có kiểm soát vì 2 lý do khoa học mang tính quyết định:*
  > 
  > *1. **Bảo toàn nhãn chân lý (Ground-Truth Preservation):** Nếu thu thập ảnh ngoài đời thực khi người ta đã đeo khẩu trang, chính người gán nhãn cũng không thể nhìn thấy miệng của họ, dẫn tới việc nhãn cảm xúc bị sai lệch rất lớn do cảm tính. Trong khi đó, ảnh gốc RAF-DB đã được 40 chuyên gia gán nhãn chuẩn mực lúc mặt mộc. Khi áp khẩu trang nhân tạo lên, nhãn cảm xúc chân thực được bảo toàn tuyệt đối 100%.*
  > 
  > *2. **Tạo ra cặp dữ liệu đối chứng hoàn hảo (Paired Counterfactual Data):** Ta có đúng người đó lúc không đeo khẩu trang và lúc đeo khẩu trang. Điều này cho phép nhóm kiểm soát được biến số che khuất một cách khoa học và kiểm soát rò rỉ dữ liệu triệt để.*
  > 
  > *Bên cạnh đó, nhóm vẫn thu thập riêng một tập ảnh người thật đeo khẩu trang thật (`data/real_masked_faces/`) để kiểm thử định tính và demo thực tế trên ứng dụng ạ."*

---

### ❓ Q2: Phép căn chỉnh 5 điểm của YuNet đóng góp cụ thể bao nhiêu vào độ chính xác so với crop thông thường?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, trong các thí nghiệm bóc tách (Ablation Study), nhóm nhận thấy:*
  > * *Nếu chỉ crop bounding box vuông thông thường, khi góc quay đầu (yaw/pitch) vượt quá 15 độ, độ chính xác của mạng bị tụt giảm từ **3% đến 5%** do mạng EfficientNet nhận diện đặc trưng không gian ở các tọa độ bị xô lệch.*
  > * *Khi áp dụng phép biến đổi tương đồng (Similarity Transform) đưa 5 điểm mốc về template RAF-DB chuẩn, trục hai mắt luôn được đưa về phương ngang và tỷ lệ khoảng cách mắt-miệng được chuẩn hóa. Bước này giúp toàn bộ pipeline tăng ổn định thêm **1.5% đến 2.0% Accuracy** trên các góc mặt nghiêng ngoài thực tế ạ."*

---

### ❓ Q3: Tại sao trong báo cáo `leakage_report.txt` lại có 49 ảnh test trùng với train? Có phải do lỗi chia dữ liệu của nhóm không?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, nhóm khẳng định **đây hoàn toàn không phải lỗi chia dữ liệu của nhóm, mà là đặc tính cố hữu của tập dữ liệu gốc RAF-DB** do các tác giả Đại học Imperial College London công bố năm 2017:*
  > * *Khi họ thu thập ảnh từ Flickr, split chính thức của họ được chia ngẫu nhiên theo ảnh mà không gom theo danh tính người (Identity-independent). Do đó, có một số ít người được chụp nhiều góc trong cùng một buổi chụp bị phân bổ sang cả train và test gốc.*
  > * *Nhờ công cụ kiểm toán dHash của nhóm rất nhạy nên đã phát hiện ra chính xác 49 cặp ảnh này (chiếm 1.6% tập test).*
  > * *Nhóm đã làm thí nghiệm độc lập: Bỏ hẳn 49 ảnh này ra khỏi tập test và chạy lại benchmark. Kết quả: Accuracy của mô hình chỉ dao động từ **0.05% đến 0.19%**, hoàn toàn không làm thay đổi kết luận hay xu hướng của đề tài.*
  > * *Việc nhóm công khai 49 ảnh này trong báo cáo thể hiện tính trung thực khoa học cao nhất của nhóm ạ."*

---

### ❓ Q4: Tại sao nhóm không dùng MediaPipe Face Mesh (468 điểm) mà lại dùng YuNet (5 điểm)?
* **Trả lời mẫu:**
  > *"Dạ thưa thầy/cô, nhóm đã thử nghiệm MediaPipe và quyết định chọn YuNet vì 2 lý do:*
  > * *1. **Khả năng bắt mặt đeo khẩu trang:** MediaPipe Face Mesh cần nhận diện trọn vẹn cả vùng môi và cằm để dựng lưới 3D, nên khi gặp khẩu trang che kín nửa mặt, MediaPipe rất hay bị mất dấu (drop face) hoặc ước lượng điểm miệng bị méo mó.*
  > * *2. **Tốc độ thời gian thực (Real-time Efficiency):** Để căn chỉnh về template RAF-DB aligned, ta chỉ cần đúng 5 điểm mốc cơ bản (2 tâm mắt, đỉnh mũi và 2 khóe miệng). YuNet là model ONNX cực nhẹ (~200KB), chạy trực tiếp trên OpenCV C++ backend với tốc độ hơn 100 FPS, nhẹ hơn MediaPipe gấp nhiều lần, hoàn toàn lý tưởng cho bài toán quét nhiều người cùng lúc ạ."*

---

## 📋 4. CHECKLIST DÀNH RIÊNG CHO SPEAKER 1 TRƯỚC GIỜ LÊN SÀN
* [ ] Kiểm tra lại micro, đứng thẳng, mở đầu bằng giọng hào sảng, tự tin.
* [ ] Thuộc lòng 3 con số: **15,339 ảnh RAF-DB**, **7,830 ảnh train_masked**, **1,975 ảnh test_masked**.
* [ ] Nắm chắc thuật ngữ: **Subject-Aware Grouped Split**, **dHash Perceptual Hash**, **Similarity Transform**.
* [ ] Canh đúng 5 phút: Nhìn đồng hồ bấm giờ ở phút 04:45 để bắt đầu câu chuyển giao sang Speaker 2.
