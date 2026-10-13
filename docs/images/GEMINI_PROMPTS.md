# Prompt tạo hình minh hoạ khái niệm (dán vào Antigravity / Gemini)

Các hình `fig01`–`fig08` trong thư mục này được vẽ **từ số liệu thật** bằng script; **không** thay chúng bằng hình AI.
Hình AI chỉ dùng cho **sơ đồ khái niệm** (`concept_*.png`) bên dưới. Lý do: một heatmap Grad-CAM hay một biểu đồ
do AI "vẽ" là dữ liệu bịa — nếu giảng viên hỏi "hình này lấy từ đâu?" bạn phải trả lời được là "chạy từ code".

**Cách dùng:** mở Antigravity → chat với Gemini → dán khối prompt chung + prompt của từng hình → lưu ảnh
đúng tên file vào `docs/images/`. Tài liệu `docs/STUDY_GUIDE_QA.md` đã có sẵn dòng chèn ảnh cho từng tên file
(đang để dạng comment `<!-- ... -->`; tạo ảnh xong thì bỏ comment).

> Mẹo: AI tạo ảnh thường viết sai dấu tiếng Việt → yêu cầu chữ trong hình bằng **tiếng Anh**, ngắn.

---

## Khối prompt chung (dán trước mỗi prompt)

```
Style: clean flat technical infographic for a university computer-vision report.
White background (#fcfcfb), thin dark-gray lines, accent colors blue #2a78d6 and orange #eb6834 only,
plus light gray for secondary elements. Sans-serif labels in English, short, spelled exactly as given.
No 3D, no gradients, no stock-photo people, no logos, no watermark. Aspect ratio 16:9, high resolution.
```

---

### concept_01_facs_aus.png — Action Units trên khuôn mặt & vùng bị khẩu trang che
```
A neutral line-drawing of a frontal human face (simple, gender-neutral, no hair detail).
A horizontal dashed line across the nose bridge labeled "mask line".
Above the line, blue callouts pointing to: inner brow "AU1", outer brow "AU2", between brows "AU4",
upper eyelid "AU5", outer eye corner crow's feet "AU6 (Duchenne)", lower eyelid "AU7".
Below the line, the lower half is covered by a light-gray semi-transparent surgical mask; orange callouts
(drawn faded) point to: nose wrinkle "AU9", upper lip "AU10", lip corners "AU12", "AU15", "AU20", open mouth "AU25/26".
Title at top: "Upper face stays visible, lower-face Action Units are lost".
```

### concept_02_pipeline.png — Pipeline end-to-end
```
Left-to-right flowchart with 7 rounded boxes connected by arrows:
"Webcam frame 1280x720" -> "YuNet detector (bbox + 5 landmarks)" -> "Similarity transform to RAF-DB template (224x224)"
-> "EfficientNet-B0 (HSEmotion)" -> "Logits + prior bias, softmax" -> "IoU tracker + EMA smoothing" -> "Label + Grad-CAM overlay".
Under the EfficientNet box a small branch arrow down to "Grad-CAM every 10 frames (cached)".
Above the whole flow a thin bracket labeled "Real-time loop ~37 FPS (1 face, M1)".
```

### concept_03_leakage_split.png — Rò rỉ dữ liệu: random split vs group split
```
Two panels side by side.
Left panel title "Random split (leaky)": a pool of small face icons where each original face has a twin with a mask
(pairs share the same color outline). Arrows split them into "Train" and "Val" boxes; several twin pairs are cut in half
(one in Train, one in Val), highlighted with red-orange warning marks and the caption "same identity on both sides".
Right panel title "Group split (no leakage)": pairs are first wrapped together in a small group box
"group = raf_train_00012", then whole groups go to Train or Val; caption "0 shared groups".
```

### concept_04_alignment.png — Căn chỉnh 5 điểm
```
Left: a tilted face crop with 5 blue dots (two eyes, nose tip, two mouth corners).
Middle: an arrow labeled "estimateAffinePartial2D (rotation + uniform scale + translation, LMedS)".
Right: a 224x224 square with the same face upright; 5 orange target dots at fixed template positions,
with small coordinate labels "(0.27,0.32) (0.70,0.32) (0.48,0.57) (0.30,0.73) (0.69,0.73)".
```

### concept_05_mbconv_se.png — Khối MBConv + Squeeze-and-Excitation
```
Vertical block diagram of an inverted-residual block:
"Input C" -> "1x1 Conv expand (x6)" -> "BN + SiLU" -> "Depthwise 3x3 / 5x5" -> "BN + SiLU"
-> side module "Squeeze-and-Excitation: global avg pool -> FC -> SiLU -> FC -> sigmoid -> channel scale"
-> "1x1 Conv project to C_out" -> "BN" -> "+ skip connection" -> "Output".
Show the channel width visually: narrow -> wide -> narrow (inverted bottleneck).
```

### concept_06_logit_adjustment.png — Logit Adjustment & prior-bias
```
Two-part figure.
Part A "Training: logit-adjusted loss": a 2D scatter with many blue dots (class "happy", frequent) and few orange dots
(class "fear", rare). A decision boundary line pushed toward the blue cluster, with a wide margin around the orange
class, caption "rare class gets a larger margin: tau * log(pi_c / pi_y)".
Part B "Inference: add t * log(prior)": a slider from t=0 to t=1 with marker at "t* = 0.85 (chosen on validation)";
left end labeled "balanced: high recall on rare classes", right end labeled "matches real class frequency".
```

### concept_07_gradcam.png — Cách tính Grad-CAM
```
Flow diagram: face image -> CNN -> "last feature maps A^k (1280 x 7 x 7)" -> "class score y^c".
A dashed backward arrow from y^c to the feature maps labeled "gradients dy^c/dA^k".
Then "global average pool gradients -> weights alpha_k" -> "weighted sum of A^k" -> "ReLU" -> "7x7 heatmap"
-> "upsample to 224x224, overlay on face" (show a face with a warm blob around the eyes).
```

### concept_08_realtime_cache.png — Cache Grad-CAM theo thời gian
```
A horizontal timeline of 21 small frame tiles numbered 0..20.
Every frame tile has a small blue tick "predict (~13 ms)".
Frames 0, 10, 20 additionally have an orange block "Grad-CAM backward (~80 ms)".
Frames in between show a faded orange "reuse cached heatmap".
Caption: "Average Grad-CAM cost ~8 ms per frame -> ~37 FPS instead of single-digit FPS".
```

### concept_09_r_architecture.png — Kiến trúc nhánh R
```
Two columns.
Left column "Python (training & export)": "train.py" -> "best_model_mask_aware.pth" -> "export_jit.py (wrap + logit_bias buffer)"
-> "best_model_mask_aware_jit.pt + _meta.json". A check badge "parity test: max diff < 1e-4".
Right column "R (inference & UI)": "torch for R (LibTorch) jit_load" -> "R/inference.R, R/face_detector.R, R/gradcam.R" -> "Shiny app (app.R)".
An arrow from the .pt file across to jit_load.
```
