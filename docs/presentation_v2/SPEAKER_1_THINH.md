# Speaker 1: Hồ Trường Thịnh (SE200698)

**Phần:** Data & Preprocessing · **Slide:** 1–8 · **Thời lượng:** ~6.5 phút

File slide: `docs/XAI-FER_Slides_v2.pptx`. Kịch bản bên dưới trùng với speaker notes trong file slide.

## Kịch bản theo slide

### Slide 1: XAI-FER Mask-Aware (title)

[THỊNH · ~20s] Kính chào thầy cô và các bạn. Nhóm em trình bày đề tài XAI-FER Mask-Aware: nhận diện cảm xúc khuôn mặt thời gian thực, nhiều khuôn mặt cùng lúc, có giải thích, và vẫn hoạt động khi người dùng đeo khẩu trang. Em là Thịnh, phụ trách dữ liệu và tiền xử lý. Bạn Quý trình bày mô hình, huấn luyện và đánh giá. Bạn Sang trình bày hệ thống thời gian thực, phần giải thích Grad-CAM và demo.

### Slide 2: Problem: 3 barriers

[THỊNH · ~50s] Bài toán có ba rào cản. Thứ nhất, khoảng cách giữa phòng lab và đời thực: mô hình học trên FER-2013, ảnh xám 48×48, hỏng khi gặp ảnh màu, mặt nghiêng, ánh sáng phức tạp; vì vậy nhóm dùng RAF-DB, ảnh màu chụp ngoài đời. Thứ hai, khẩu trang che nửa dưới khuôn mặt. Nhóm đo trực tiếp: mô hình chỉ học RAF-DB đạt 89.2% trên mặt thường nhưng tụt còn 56.6% khi đeo khẩu trang; 48% ảnh vui bị đoán thành trung tính vì mất nụ cười. Thứ ba, mô hình hộp đen chỉ trả về một nhãn, khó tin trong y tế, giáo dục; nhóm kèm heatmap Grad-CAM cho mỗi dự đoán. Mục tiêu: in-the-wild, mask-aware, real-time và explainable.

### Slide 3: Roadmap: one pipeline, three presenters

[THỊNH · ~25s] Đây là lộ trình. Cả hệ thống là một pipeline, chia thành ba đoạn liền nhau. Em trình bày đoạn đầu: dữ liệu đi vào, được sinh khẩu trang, chia tập không rò rỉ và tiền xử lý thành tensor. Quý nhận tensor đó để huấn luyện, hiệu chỉnh và đánh giá mô hình. Sang đưa mô hình vào webcam: dò mặt, căn chỉnh, theo dõi, giải thích và chạy demo. Mỗi slide có ghi tên người trình bày ở góc dưới.

### Slide 4: RAF-DB: what the model learns from

[THỊNH · ~55s] Dữ liệu gốc là RAF-DB: 15,339 ảnh, khoảng 40 người gán nhãn cho mỗi ảnh, 12,271 ảnh train và 3,068 ảnh test chính thức. Nhóm dùng bản aligned: ảnh 100×100 màu, hai mắt đã nằm ở vị trí cố định. Biểu đồ cho thấy vấn đề đầu tiên của dữ liệu: lệch lớp rất mạnh. Happy có 4,772 ảnh, fear chỉ 281, chênh khoảng 17 lần. Nếu không xử lý, mô hình sẽ thiên về happy và neutral. Cột cam là ảnh khẩu trang sinh ra: chỉ 64% ảnh có bản khẩu trang, vì công cụ dò landmark bỏ sót mặt nhỏ hoặc nghiêng. Ở tập test khẩu trang, fear chỉ còn 35 ảnh, nên điểm F1 của fear dao động lớn; bạn Quý sẽ nhắc lại điểm này.

### Slide 5: RAF-DB data & mask generation pipeline

[THỊNH · ~50s] Vì chưa có bộ dữ liệu cảm xúc đeo khẩu trang thật nào đủ lớn và có nhãn, nhóm tự sinh bằng MaskTheFace. Quy trình: lấy ảnh RAF-DB gốc, dùng dlib tìm 68 điểm landmark ở sống mũi, má, cằm, rồi uốn ảnh mẫu khẩu trang 2D cho khớp khuôn mặt. Kết quả: 7,830 ảnh train_masked và 1,975 ảnh test_masked. Có hai quy tắc quan trọng: giữ nguyên nhãn của ảnh gốc, vì cảm xúc không đổi khi đeo khẩu trang; và giữ ID ảnh nguồn trong tên file, ví dụ raf_train_00012_surgical.jpg. Nhờ có cặp trước/sau, nhóm đo được đúng tác động của khẩu trang. Hạn chế: chỉ một kiểu khẩu trang; slide 8 sẽ cho thấy cách nhóm bù lại.

### Slide 6: Data leakage & independent audit

[THỊNH · ~70s] Cái tên file giữ ID ảnh nguồn được dùng ngay ở đây, để chặn rò rỉ dữ liệu. Pipeline cũ gộp ảnh gốc và ảnh khẩu trang rồi chia ngẫu nhiên 85/15. Khi đó ảnh gốc của một người nằm ở train, bản đeo khẩu trang của chính người đó nằm ở validation: mô hình nhớ khuôn mặt chứ không học cảm xúc. Val F1 đạt 0.826 nhưng test chỉ 0.718. Tệ hơn, early stopping và hiệu chỉnh đều đọc val F1, nên mọi lựa chọn sau đều lệch. Nhóm chia theo nhóm ảnh nguồn: mọi biến thể của một ảnh luôn cùng phía, phân tầng theo nhãn, 15% làm validation, lưu cố định trong file JSON và dùng chung cho cả hai giai đoạn huấn luyện. Code có assert không nhóm nào nằm ở cả hai phía. Nhóm cũng kiểm toán độc lập bằng perceptual hash và tương quan pixel: 0 ảnh test_masked lấy từ train, 0 ảnh nguồn chung giữa train và val. Có 49 ảnh test trùng người với train, đó là đặc điểm của split chính thức RAF-DB; bỏ chúng đi thì độ chính xác chỉ đổi 0.1 đến 0.2 điểm.

### Slide 7: Preprocessing: from JPEG file to model input

[THỊNH · ~55s] Sau khi chia tập, mỗi ảnh đi qua cùng một lõi tiền xử lý bốn bước. Một: đọc JPEG, chuyển sang RGB, nhãn lấy theo tên thư mục với thứ tự 7 lớp cố định. Hai: phóng 100×100 lên 224×224, đúng kích thước đầu vào của EfficientNet-B0. Ba: chuyển thành tensor, giá trị pixel về khoảng 0 đến 1. Bốn: chuẩn hoá theo mean và std của ImageNet, vì backbone được huấn luyện trước với đúng thống kê này; dùng thống kê khác sẽ làm lệch toàn bộ đặc trưng. Lõi này dùng chung cho ba nơi: train có thêm augmentation ngẫu nhiên; validation và test chỉ resize và chuẩn hoá, để kết quả lặp lại được; webcam thì phải căn chỉnh khuôn mặt trước cho giống ảnh RAF-DB aligned, phần đó bạn Sang trình bày ở slide 16. Code cũng tự kiểm tra: thư mục lớp lạ thì dừng, ảnh gốc và bản khẩu trang khác nhãn thì dừng.

### Slide 8: Augmentation: simulate the wild and the mask

[THỊNH · ~55s] Augmentation chỉ áp dụng cho tập train, ngẫu nhiên mỗi epoch, nên mỗi epoch mô hình thấy một phiên bản mới của từng ảnh. Hình phía trên là chính các phép biến đổi trong code áp lên một ảnh RAF-DB. Nhóm thứ nhất mô phỏng đời thực: cắt 80 đến 100% khuôn mặt, lật ngang, xoay ±10 độ, đổi độ sáng và tương phản ±30%. Nhóm thứ hai là LowerFaceOcclusion do nhóm tự viết: che từ khoảng 50–62% chiều cao trở xuống, mép trên hơi võng như dây khẩu trang, 6 màu hoặc nhiễu, xác suất 0.3, chỉ bật ở giai đoạn 2. Mục đích là bù cho việc MaskTheFace chỉ có một kiểu khẩu trang. Cuối cùng RandomErasing xoá một vùng nhỏ để mô hình chịu được tay, kính, tóc che mặt. Vậy dữ liệu đã sạch, không rò rỉ, nhưng còn lệch lớp 17 lần. Mời bạn Quý trình bày cách mô hình học trên dữ liệu này và xử lý lệch lớp.

## Giải thích để hiểu sâu

**Vì sao phải tiền xử lý đúng cách.** Mô hình chỉ học được từ những gì nó thấy. Nếu ảnh đầu vào khác kích thước, khác thống kê màu hay khác bố cục so với lúc backbone được huấn luyện trước, đặc trưng bị lệch ngay từ lớp đầu.

**Từng bước, kèm vị trí trong code (`src/dataset.py`):**
- **Đọc ảnh và nhãn** (`_image_folder_samples`): mỗi thư mục là một lớp, thứ tự lớp cố định `angry, disgust, fear, happy, neutral, sad, surprise`. Nếu có thư mục lạ, code dừng lại để không gán nhầm nhãn.
- **Chia tập theo nhóm** (`source_group`, `get_val_groups`): khoá nhóm là phần `raf_train_00012` của tên file. Code phân tầng theo nhãn, tách 15% làm val với seed 42 và lưu vào `data/splits/val_groups_r15_s42.json`. Cuối hàm có `assert` không nhóm nào nằm ở cả hai phía. Số liệu: RAF-only 10,429 train / 1,842 val; gộp với ảnh khẩu trang 17,099 / 3,002.
- **Resize 224×224:** EfficientNet-B0 nhận ảnh 224. Ảnh gốc 100×100 được phóng lên, không có thêm thông tin, nhưng khớp với kích thước receptive field mà backbone đã học.
- **ToTensor và Normalize:** dùng mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]` của ImageNet, vì HSEmotion được huấn luyện với chuẩn hoá này.
- **Augmentation** (`get_transforms`): RandomResizedCrop(scale 0.8–1.0, ratio 0.9–1.1), HorizontalFlip 0.5, RandomAffine(±10°, dịch 4%), ColorJitter(0.3, 0.3, 0.2), RandomGrayscale 0.1, LowerFaceOcclusion, rồi RandomErasing(p 0.25, 2–20% diện tích).
- **LowerFaceOcclusion** (lớp tự viết): vẽ một đa giác từ 50–62% chiều cao xuống cằm, mép trên võng tối đa 8% như dây khẩu trang. 75% trường hợp tô màu đặc (6 màu), 25% tô nhiễu. p = 0.3 ở chế độ `combined` (giai đoạn 2), p = 0 ở chế độ `original` (giai đoạn 1), để baseline RAF-DB giữ "sạch".
- **Val/test:** chỉ Resize + ToTensor + Normalize, không ngẫu nhiên, nên chạy lại luôn ra cùng kết quả.
- **Cache RAM:** ảnh 100×100 khoảng 30 KB, cả tập gộp khoảng 0.7 GB, nên giải mã một lần để epoch sau không phải đọc JPEG lại.

**Câu chuyện cần giữ khi nói:** dữ liệu lệch lớp 17× → sinh khẩu trang (có cặp trước/sau) → chia nhóm để không rò rỉ → lõi tiền xử lý chung → augmentation chỉ ở train. Kết thúc bằng câu: "dữ liệu sạch nhưng còn lệch lớp", rồi chuyển cho Quý.

## Câu hỏi hội đồng có thể hỏi

**H: Vì sao không dùng khẩu trang thật?**

Đ: Chưa có bộ dữ liệu cảm xúc đeo khẩu trang thật đủ lớn và có nhãn. Ảnh tổng hợp giữ được nhãn gốc (khoảng 40 người gán) và tạo được cặp trước/sau, nên đo được đúng tác động của khẩu trang. Hạn chế này nằm trong phần Future work.

**H: Phóng ảnh 100 lên 224 có làm mất chất lượng không?**

Đ: Không thêm thông tin, nhưng không mất thông tin. Backbone được huấn luyện trước ở 224, nên đầu vào 224 giữ đúng tỉ lệ đặc trưng mà nó đã học. Đầu vào 100×100 sẽ làm feature map cuối chỉ còn khoảng 3×3.

**H: Tại sao chuẩn hoá theo ImageNet mà không theo thống kê của RAF-DB?**

Đ: Vì backbone HSEmotion được huấn luyện trước với chuẩn hoá ImageNet. Fine-tuning giữ phần lớn trọng số, nên đầu vào phải cùng thang đo thì đặc trưng mới khớp.

**H: Lật ngang có làm sai cảm xúc không?**

Đ: Không, vì cảm xúc gần như đối xứng trái phải. Các phép không giữ nhãn, như xoay lớn hay cắt mất mắt, không được dùng: xoay chỉ ±10°, crop giữ ít nhất 80%.

**H: Chia nhóm theo ảnh nguồn đã đủ chống rò rỉ theo người chưa?**

Đ: RAF-DB không có ID người. Nhóm dùng ID ảnh nguồn, cộng thêm kiểm toán bằng perceptual hash. Còn 49 ảnh test (1.6%) trùng người với train, đây là đặc điểm của split chính thức; bỏ chúng đi thì accuracy chỉ đổi 0.1–0.2 điểm.

**H: Tại sao occlusion chỉ bật ở giai đoạn 2?**

Đ: Để giai đoạn 1 là baseline RAF-DB thuần, dùng làm mốc so sánh công bằng: mô hình RAF-only trong bảng kết quả không hề thấy khẩu trang.
