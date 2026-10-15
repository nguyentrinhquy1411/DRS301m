# Speaker 2: Nguyễn Trinh Quý (SE203691)

**Phần:** Model, Training & Evaluation · **Slide:** 9–15 · **Thời lượng:** ~6 phút

File slide: `docs/XAI-FER_Slides_v2.pptx`. Kịch bản bên dưới trùng với speaker notes trong file slide.

## Kịch bản theo slide

### Slide 9: Face-pretrained backbone

[QUÝ · ~45s] Cảm ơn Thịnh. Điểm xuất phát là backbone. Lựa chọn thông thường là ImageNet, nhưng ImageNet học chó, mèo, xe, không biết gì về cơ mặt. Nhóm chọn EfficientNet-B0 của HSEmotion: học trước trên VGGFace2 để nhận dạng khuôn mặt, rồi trên AffectNet để nhận diện cảm xúc. Đầu ra 7 lớp được khởi tạo từ classifier AffectNet, bỏ lớp contempt vì RAF-DB không có. Mô hình chỉ khoảng 4 triệu tham số, đủ nhẹ cho thời gian thực, và đạt 89.15% trên test RAF-DB, F1 0.833.

### Slide 10: Two-stage transfer learning

[QUÝ · ~55s] Huấn luyện gồm hai giai đoạn, dùng chung một phép chia validation mà Thịnh vừa trình bày. Giai đoạn 1 trên RAF-DB gốc: 1 epoch chỉ train phần head để không phá trọng số đã học trước, rồi 25 epoch fine-tune toàn mạng. Giai đoạn 2 khởi tạo từ giai đoạn 1, train 20 epoch trên ảnh gốc cộng ảnh khẩu trang, bật LowerFaceOcclusion. Để xử lý lệch lớp 17 lần, nhóm dùng Logit-Adjusted loss: cộng log tần suất lớp vào logits khi tính loss, để lớp hiếm như fear có biên rộng hơn. Các kỹ thuật chống overfit: label smoothing 0.1, Mixup và CutMix, trung bình trọng số EMA 0.998, AdamW với warmup và cosine, learning rate backbone 1e-4 nhỏ hơn head 1e-3.

### Slide 11: Post-training prior-bias calibration

[QUÝ · ~55s] Logit-Adjusted loss có tác dụng phụ: mô hình đoán như thể các lớp cân bằng, nên đoán thừa fear và disgust. Ví dụ fear có recall 0.78 nhưng precision chỉ 0.32. Cách sửa là hiệu chỉnh sau huấn luyện: cộng t nhân log prior vào logits rồi lấy argmax. Nhóm quét t trên tập validation và chọn t tối đa macro F1: 0.85 cho Mask-Aware, 0.70 cho RAF-DB. Tập test không được dùng ở bước này, và t được lưu trong checkpoint. Trên validation, accuracy tăng 5.09 điểm và macro F1 tăng 0.061. Để kiểm tra không overfit validation: t tốt nhất trên test khẩu trang là 1.0 với F1 0.713, rất gần mức 0.708 tại t 0.85.

### Slide 12: Results: +22.68 points on masked faces

[QUÝ · ~55s] Đây là kết quả chính, cùng một giao thức: fp32, TTA lật ngang, hiệu chỉnh chọn trên validation, split test chính thức. Trên ảnh khẩu trang, Mask-Aware đạt 79.29%, cao hơn mô hình chỉ RAF-DB 22.68 điểm, macro F1 tăng 0.174. Cái giá rất nhỏ: trên mặt thường chỉ giảm 0.72 điểm, từ 89.15 xuống 88.43%. Trên toàn bộ 5,043 ảnh test đạt 84.85%. Dòng MobileNetV3 là pipeline cũ có validation bị rò rỉ, nên chỉ để tham khảo, không so sánh trực tiếp.

### Slide 13: Per-emotion analysis on masked faces

[QUÝ · ~45s] Nhìn theo từng cảm xúc trên ảnh khẩu trang: happy đạt F1 0.89, vì nụ cười thật làm má nâng lên và mắt híp lại, tức Action Unit 6, vẫn nhìn thấy trên khẩu trang. Khó nhất là disgust 0.47 và fear 0.53, vì tín hiệu của chúng nằm ở mũi và miệng, như nhăn mũi, nhếch môi, kéo môi ngang, đều bị che. Như Thịnh đã nói, fear chỉ có 35 ảnh test khẩu trang, nên con số này dao động lớn.

### Slide 14: Where the errors go on masked faces

[QUÝ · ~45s] Ma trận nhầm lẫn chuẩn hoá theo hàng, trên 1,975 ảnh test khẩu trang. Mô hình chỉ RAF-DB dồn 48% ảnh happy về neutral, vì nó tìm nụ cười mà không thấy. Mask-Aware giảm còn 9%. Các lỗi còn lại có lý do giải phẫu: fear bị nhầm sang surprise 29%, vì cả hai đều nhướng mày và mở to mắt, khác nhau chủ yếu ở miệng, chính là vùng bị che.

### Slide 15: Ablation: what post-processing adds

[QUÝ · ~45s] Ablation trên cùng checkpoint Mask-Aware cho biết bước hậu xử lý nào đáng giá. Hiệu chỉnh prior-bias đóng góp nhiều nhất: cộng 4.7 điểm trên mặt thường và 6.5 điểm trên mặt khẩu trang. TTA lật ngang gần như không đổi, nên khi chạy webcam nhóm tắt TTA để tiết kiệm thời gian. Đóng góp lớn nhất vẫn là dữ liệu khẩu trang cộng occlusion, 22.68 điểm ở slide 12. Lưu ý: toàn bộ số liệu này đo trên ảnh test đã căn chỉnh sẵn. Với webcam, khuôn mặt chưa được căn chỉnh. Mời bạn Sang trình bày cách hệ thống xử lý điều đó trong thời gian thực.

## Giải thích để hiểu sâu

**Logit-Adjusted loss (Menon et al., ICLR 2021).** Cross-entropy được tính trên `logits + τ·log(π)`, trong đó π là tần suất lớp trong train và τ = 1. Lớp hiếm có log π rất âm, nên mô hình phải đẩy logit của lớp đó cao hơn mới giảm được loss, tức là biên cho lớp hiếm rộng hơn. Lúc dự đoán thì không cộng, nên mô hình dự đoán như thể các lớp cân bằng.

**Prior-bias calibration (`src/calibrate.py`).** Vì dữ liệu thật vẫn lệch (test cũng nhiều happy), nên cộng lại một phần prior: `argmax(logits + t·log π)`. t = 0 là không chỉnh, t = 1 là trả lại đủ prior. Quét t từ −0.5 đến 1.5 trên VAL, chọn t có macro F1 cao nhất (0.85 cho Mask-Aware, 0.70 cho RAF-DB) và lưu vào checkpoint. Ở bản TorchScript, phần cộng này đã gộp sẵn vào mô hình.

**Hai giai đoạn.** Giai đoạn 1 tạo baseline RAF-DB sạch (không occlusion). Giai đoạn 2 khởi tạo từ đó và thêm ảnh khẩu trang + LowerFaceOcclusion. Cả hai dùng cùng file split, nên ảnh val của giai đoạn 2 không bao giờ được giai đoạn 1 nhìn thấy.

**Các kỹ thuật regularization, mỗi cái một câu:**
- Label smoothing 0.1: nhãn đích 0.9 thay vì 1, vì nhãn RAF-DB vốn có nhiễu.
- Mixup/CutMix: trộn hai ảnh và trộn nhãn theo tỉ lệ, giúp mô hình bớt tự tin thái quá.
- EMA 0.998: giữ trung bình trượt của trọng số và đánh giá bằng bản trung bình này, cho kết quả ổn định hơn.
- LR backbone 1e-4 nhỏ hơn head 1e-3: không phá đặc trưng khuôn mặt đã học trước.

**Đọc bảng kết quả.** Accuracy trên mặt khẩu trang là 79.29%, F1 0.708. Accuracy cao hơn F1 vì happy chiếm nhiều ảnh. Hãy nhấn mạnh macro F1 tăng 0.174, vì macro F1 coi mọi lớp như nhau.

**Cần nói rõ:** mọi số trên test đo trên ảnh RAF-DB đã căn chỉnh sẵn. Câu này là cầu nối sang phần của Sang.

## Câu hỏi hội đồng có thể hỏi

**H: Logit adjustment khác class weighting thế nào?**

Đ: Weighted CE nhân trọng số vào loss của từng mẫu, nên gradient của lớp hiếm bị phóng to và huấn luyện dễ dao động. LA dịch biên quyết định bằng cách cộng log prior vào logits, có cơ sở lý thuyết là tối ưu balanced error, và ổn định hơn.

**H: Chọn t trên val có phải là overfit val không?**

Đ: t là một tham số vô hướng, chọn trên val và không chạm vào test. Kiểm tra lại thì t tốt nhất trên test khẩu trang (1.0, F1 0.713) gần t* (0.85, F1 0.708).

**H: Tại sao accuracy mặt thường giảm 0.72 điểm?**

Đ: Mô hình phải chia khả năng cho cả vùng mắt. Đổi lại mặt khẩu trang tăng 22.68 điểm và toàn bộ test tăng 8.45 điểm.

**H: Có chạy nhiều seed không?**

Đ: Chưa, mới một seed. Đây là hạn chế đã ghi trên slide. Với fear chỉ có 35 ảnh test khẩu trang, sai số của lớp này có thể vài điểm.

**H: Vì sao không so sánh trực tiếp với MobileNetV3?**

Đ: Dòng đó dùng pipeline cũ có validation rò rỉ, nên early stopping và hiệu chỉnh đã bị lệch. Nhóm giữ nó để tham khảo, không dùng làm bằng chứng.
