# Speaker 3: Trịnh Quốc Sang (SE194517)

**Phần:** Real-Time System, XAI & Demo · **Slide:** 16–21 · **Thời lượng:** ~4.5 + demo 2.5 phút

File slide: `docs/XAI-FER_Slides_v2.pptx`. Kịch bản bên dưới trùng với speaker notes trong file slide.

## Kịch bản theo slide

### Slide 16: Multi-face detection & 5-landmark alignment

[SANG · ~50s] Cảm ơn Quý. Đây là bước tiền xử lý lúc suy luận, tương ứng với thẻ Webcam mà Thịnh đã giới thiệu. OpenCV YuNet dò mọi khuôn mặt trong khung hình kèm 5 điểm mốc: hai mắt, mũi, hai khóe miệng, kể cả khi đeo khẩu trang hay nghiêng đầu. Sau đó một phép biến đổi tương tự, gồm xoay, phóng đều và tịnh tiến, đưa 5 điểm này về đúng template của RAF-DB. Nhờ vậy ảnh cắt từ webcam có cùng bố cục với ảnh mô hình đã học. Phép ước lượng dùng LMedS nên chịu được một điểm sai, ví dụ khóe miệng bị khẩu trang che. Dò trên khung 640 pixel cho nhanh, căn chỉnh trên khung gốc để giữ chi tiết. Tổng cộng khoảng 5 đến 7 ms mỗi khung.

### Slide 17: XAI: Grad-CAM and the attention shift

[SANG · ~55s] Để mở hộp đen, nhóm dùng Grad-CAM tại tầng bn2, bản đồ đặc trưng 1280×7×7, tính cho nhãn thật trên ảnh test thật. Khi không đeo khẩu trang, cả hai mô hình đều nhìn vào miệng. Khi đeo khẩu trang, Mask-Aware chuyển sự chú ý lên mắt và lông mày. Nhóm đo định lượng trên 1,975 cặp ảnh: tỉ lệ năng lượng heatmap phía trên đường khẩu trang tăng từ 47% lên 58%. Mô hình chỉ RAF-DB cũng dịch lên, từ 42 lên 55%, nhưng dàn trải và vẫn lùi về neutral. Hệ quả thấy được: happy bị đoán thành neutral giảm từ 48% xuống 9%. Grad-CAM là giải thích thô 7×7, mang tính định tính.

### Slide 18: Real-time: IoU tracker + EMA smoothing

[SANG · ~45s] Với video nhiều người, mỗi khung hình dự đoán độc lập sẽ làm nhãn nhảy liên tục. Nhóm giữ ID cho từng khuôn mặt bằng IoU tracker: ghép khuôn mặt giữa hai khung nếu IoU từ 0.3 trở lên, giữ ID tối đa 10 khung, khoảng 0.3 giây, khi mặt bị mất tạm. Xác suất được làm mượt bằng EMA với alpha 0.4, tức nhớ khoảng 2 đến 3 khung gần nhất, nên nhãn ổn định nhưng vẫn phản ứng trong khoảng 0.1 giây. Ghép tham lam O(M·N) đủ cho tối đa 5 khuôn mặt, không cần Kalman filter.

### Slide 19: Real-time performance (Apple M1)

[SANG · ~40s] Hiệu năng đo trên Apple M1 dùng GPU MPS, khung tổng hợp 1280×720 có mặt RAF-DB. Một khuôn mặt: 55 FPS khi tắt Grad-CAM, 37.4 FPS khi bật, vì Grad-CAM chỉ tính lại mỗi 10 khung, trung bình thêm khoảng 8 ms mỗi khung. Năm khuôn mặt vẫn trên 15 FPS. Tắt TTA không làm giảm accuracy, vẫn 88.43%. Hạn chế: chưa đo bằng webcam thật và chưa đo trên GPU NVIDIA.

### Slide 20: R deployment: TorchScript + Shiny

[SANG · ~40s] Nhóm cũng triển khai bằng R. Huấn luyện ở Python, sau đó xuất TorchScript, đã gộp sẵn bias hiệu chỉnh vào mô hình, kèm file meta gồm lớp, mean, std. R nạp bằng torch::jit_load chạy trên LibTorch, không cần Python lúc suy luận. Kiểm tra parity: sai số tối đa 3e-7, argmax khớp 10/10. Ứng dụng Shiny đã có dò mặt, căn chỉnh Umeyama, tracker và phần toán Grad-CAM; tab dự đoán trực tiếp và hook bn2 cho Grad-CAM đang hoàn thiện.

### Slide 21: Conclusion & contributions

[SANG · ~40s + demo] Tổng kết: Mask-Aware tăng 22.7 điểm trên ảnh khẩu trang, đạt 79.29%; không có rò rỉ giữa train và val; 84.85% trên toàn bộ 5,043 ảnh test; 37.4 FPS có Grad-CAM. Hạn chế: khẩu trang tổng hợp một kiểu, chưa có tập khẩu trang thật có nhãn; một seed; Grad-CAM định tính; disgust và fear vẫn khó. Hướng tiếp: gán nhãn khẩu trang thật, ablation nhiều seed, đo webcam thật và GPU NVIDIA, hoàn thiện app R. Bây giờ em xin demo trực tiếp. [DEMO ~2.5 phút: 1) không khẩu trang, cười → happy, heatmap ở miệng; 2) đeo khẩu trang, cười bằng mắt → vẫn happy, heatmap dời lên mắt; 3) hai người cùng khung → hai ID ổn định; 4) bật/tắt Grad-CAM để thấy FPS thay đổi.] Cảm ơn thầy cô, nhóm em xin nhận câu hỏi.

## Giải thích để hiểu sâu

**Tiền xử lý lúc suy luận (`src/face_detector.py`).** YuNet trả về bbox và 5 điểm mốc (2 mắt, mũi, 2 khóe miệng). Phép biến đổi tương tự có 4 bậc tự do: xoay, phóng đều, tịnh tiến x và y, nên không làm méo mặt. Phép này đưa 5 điểm về template RAF-DB rồi cắt ảnh 224×224. Sau đó ảnh đi qua đúng lõi eval mà Thịnh trình bày: ToTensor + Normalize ImageNet. LMedS chọn nghiệm theo trung vị sai số, nên khi một điểm (khóe miệng dưới khẩu trang) bị đoán sai, nghiệm vẫn đúng.

**Grad-CAM (`src/gradcam.py`).** Lấy feature map ở tầng bn2 (1280 kênh × 7×7). Trọng số của mỗi kênh là trung bình gradient của điểm lớp đích theo kênh đó. Cộng có trọng số các kênh, qua ReLU, rồi phóng lên kích thước ảnh. Độ phân giải 7×7 nên chỉ chỉ ra vùng, không chỉ ra chi tiết. Chỉ số "năng lượng phía trên đường khẩu trang" là tổng heatmap ở phần trên chia cho tổng toàn ảnh, trung bình trên 1,975 cặp.

**Tracker và EMA (`src/inference.py`).** Ghép tham lam theo IoU giảm dần, ngưỡng 0.3; track mất quá 10 khung thì xoá. `P = 0.6·P_cũ + 0.4·P_mới`, trọng số của khung cách đây k bước là 0.4·0.6^k, nên 2–3 khung gần nhất chiếm phần lớn.

**FPS.** Grad-CAM cần một lượt backward (~80 ms), nên chỉ tính lại mỗi 10 khung, các khung giữa dùng lại heatmap cũ. Vì vậy chi phí trung bình chỉ khoảng 8 ms mỗi khung.

**Demo (chuẩn bị trước):** mở sẵn app, kiểm tra camera, ánh sáng phía trước mặt, có sẵn khẩu trang và một bạn đứng cùng khung. Nếu camera lỗi, chiếu `output/demo_video.mp4` thay thế.

## Câu hỏi hội đồng có thể hỏi

**H: Grad-CAM có đáng tin không?**

Đ: Đây là giải thích thô (7×7) và định tính. Nhóm bổ sung một chỉ số định lượng trên 1,975 cặp ảnh (47% → 58%), và chỉ số này khớp với thay đổi trong ma trận nhầm lẫn (happy → neutral từ 48% xuống 9%).

**H: Sao không dùng Kalman filter hay DeepSORT?**

Đ: Với tối đa khoảng 5 khuôn mặt, ghép IoU tham lam là đủ và gần như không tốn thời gian. DeepSORT cần thêm mạng ReID, làm giảm FPS.

**H: FPS đo trên frame tổng hợp có đại diện không?**

Đ: Đo được chi phí tính toán, nhưng chưa tính độ trễ của camera thật. Đây là hạn chế đã ghi; bước tiếp theo là đo webcam thật và GPU NVIDIA.

**H: Căn chỉnh ở webcam có ảnh hưởng số liệu test không?**

Đ: Không. Ảnh test RAF-DB đã căn chỉnh sẵn. Căn chỉnh chỉ để ảnh webcam giống phân phối đó; độ lệch so với template khoảng σ ≈ 0.03 trên khoảng 1,000 ảnh test.

**H: Vì sao làm thêm bản R?**

Đ: Để chứng minh mô hình độc lập với Python lúc suy luận. Parity test cho sai số tối đa 3e-7 và argmax khớp 100%.
