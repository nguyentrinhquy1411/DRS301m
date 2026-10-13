# 📐 MIGRATION PLAN: PORTING XAI-FER MASK-AWARE TO R
## Kế Hoạch Chuyển Đổi Hệ Thống Sang Ngôn Ngữ R (Nhánh `r-migration`)

---

## 📌 1. TỔNG QUAN CHIẾN LƯỢC (EXECUTIVE SUMMARY)

### 1.1. Mục Tiêu Dự Án
Chuyển đổi toàn bộ hoặc từng phần hệ thống **Real-Time Multi-Face Explainable Facial Expression Recognition (XAI-FER Mask-Aware)** từ hệ sinh thái Python (`PyTorch`, `OpenCV`, `Streamlit`) sang hệ sinh thái **ngôn ngữ R** (`R 4.3+`, `torch for R`, `OpenCV/magick`, `R Shiny`).

### 1.2. Tại Sao Lại Chuyển Đổi Sang R? (Why R?)
1. **Thống kê & Đánh giá Khoa học Vượt trội:** R là ngôn ngữ tiêu chuẩn vàng trong khoa học dữ liệu và thống kê y sinh học. Việc sử dụng `tidyverse`, `yardstick`, `caret` cho phép thực hiện các kiểm định thống kê chuyên sâu (ANOVA, paired t-test, Wilson score interval cho F1-score) mà Python thường phải cấu hình phức tạp.
2. **Hệ thống Báo cáo Tái lập (Reproducible Research):** R kết hợp với **Quarto (`.qmd`)** hoặc **R Markdown** cho phép tự động xuất báo cáo kiểm thử chuẩn IEEE/Springer tự động cập nhật số liệu khi mô hình thay đổi.
3. **Giao diện Trực quan Hóa Reactive Đẳng Cấp:** **R Shiny** (`shiny` + `bslib`) có mô hình lập trình phản ứng (reactive programming) chặt chẽ, tối ưu hơn Streamlit trong việc quản lý trạng thái phức tạp và phân tích đa luồng.
4. **Hệ sinh thái Deep Learning Thuần R Hiện Đại:** Kể từ năm 2021, Posit (RStudio) đã phát triển thư viện **`torch` for R** chạy trực tiếp trên nền tảng **LibTorch C++** (không cần cài đặt Python hay Conda, hỗ trợ đầy đủ GPU CUDA và Autograd).

---

## 🔄 2. BẢNG ĐỐI CHIẾU THÀNH PHẦN KỸ THUẬT (PYTHON $\to$ R EQUIVALENCE)

| Thành phần hệ thống | Công nghệ hiện tại (Python) | Công nghệ đích (R Ecosystem) | Đánh giá tính khả thi |
| :--- | :--- | :--- | :---: |
| **Quản lý Môi trường & Gói** | `uv`, `pyproject.toml` | `renv`, `renv.lock` | ✅ Hoàn toàn khả thi (100% chuẩn R) |
| **Deep Learning Framework** | `torch`, `torchvision`, `timm` | `torch` for R (LibTorch), `torchvision` R | ✅ Khả thi cao qua TorchScript/LibTorch |
| **Mô hình Backbone** | HSEmotion EfficientNet-B0 | TorchScript JIT export (`.pt`) hoặc R `torch::nn_module` | ✅ Khả thi, bảo toàn 100% trọng số |
| **Hàm Mất mát & Tối ưu hóa** | Logit-Adjusted CE, AdamW, EMA | R `torch::nn_module()`, `torch::optim_adamw()` | ✅ Tái lập toán học chính xác 100% |
| **Phát hiện Mặt & Landmark** | OpenCV YuNet ONNX, Haar Cascade | `opencv` R, `magick`, `imager`, hoặc `Rcpp` | ⚠️ Cần module C++ `Rcpp` hoặc LibTorch ONNX |
| **Căn chỉnh Hình học 5 điểm** | `cv2.estimateAffinePartial2D` | `imager::warp` hoặc giải ma trận Procrustes LMeDS | ✅ Khả thi toán học (đại số ma trận trong R) |
| **XAI (Grad-CAM)** | Tensor backward hooks | R `torch::autograd` backward hooks | ✅ Khả thi (R `torch` hỗ trợ hooks) |
| **Web Dashboard** | Streamlit (`app.py`) | **R Shiny (`app.R`)** + `bslib` Dark Theme | 🚀 **Vượt trội** về trải nghiệm reactive |
| **Ứng dụng Webcam** | OpenCV HighGUI (`webcam_app.py`) | `opencv::ocv_video()` hoặc Shiny Camera stream | ⚠️ Giới hạn FPS hơn so với C++ Python |
| **Đánh giá & Benchmark** | Scikit-learn (`classification_report`) | `yardstick`, `caret`, `tibble` | 🚀 **Vượt trội** về tính khoa học thống kê |
| **Trực quan hóa Đồ thị** | `matplotlib`, `seaborn` | **`ggplot2`**, `plotly`, `patchwork` | 🚀 **Vượt trội** về mặt thẩm mỹ xuất bản |

---

## 🏗️ 3. LỰA CHỌN MÔ HÌNH KIẾN TRÚC MIGRATION (ARCHITECTURAL STRATEGIES)

Có 2 hướng tiếp cận để chuyển đổi sang R:

```
HƯỚNG 1: PURE NATIVE R (KHUYẾN NGHỊ CHO NGHIÊN CỨU THUẦN R)
[Dữ liệu RAF-DB] ---> [R Data Loader] ---> [R torch (LibTorch)] ---> [Shiny Web App]
(100% R code, không phụ thuộc vào Python runtime, sử dụng LibTorch C++ backend)

HƯỚNG 2: HYBRID R-PYTHON (KHUYẾN NGHỊ TRIỂN KHAI NHANH - FAST PROTOTYPE)
[R Shiny Frontend] ---> [reticulate bridge] ---> [Python FERPredictor Engine]
(Giao diện R Shiny cao cấp gọi trực tiếp code Python sẵn có dưới nền)
```

> **Đề xuất chiến lược của nhóm:** Áp dụng **Phương pháp Chuyển đổi 4 Giai đoạn (Staged 4-Phase Migration)**:
> * Khởi đầu bằng việc trích xuất trọng số mô hình đã huấn luyện sang chuẩn trung gian **TorchScript JIT** (`.pt`), sau đó nạp trực tiếp vào **`torch` for R** để phục vụ suy luận và xây dựng ứng dụng **R Shiny**.
> * Giữ nguyên nhánh `main` (Python) làm đối chứng (Ground Truth) để kiểm tra sai số số học (Numerical Parity Validation) giữa kết quả R và Python với ngưỡng $\Delta < 10^{-5}$.

---

## 🗺️ 4. LỘ TRÌNH 4 GIAI ĐOẠN CHI TIẾT (STEP-BY-STEP ROADMAP)

```
        SPRINT 1                     SPRINT 2                     SPRINT 3                     SPRINT 4
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  MÔI TRƯỜNG & DỮ LIỆU│     │ INFERENCE & GRAD-CAM │     │  R SHINY DASHBOARD   │     │  THỐNG KÊ & TRAINING │
│  • Khởi tạo renv     │     │  • TorchScript export│     │  • app.R + bslib     │     │  • train.R (R torch) │
│  • torch for R CUDA  │ ──> │  • inference.R       │ ──> │  • Head-to-Head tab  │ ──> │  • calibrate.R       │
│  • dataset.R         │     │  • gradcam.R         │     │  • Multi-face scan   │     │  • Quarto report     │
│  • check_leakage.R   │     │  • align_face.R      │     │  • Live camera test  │     │  • Testing & Deploy  │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

### 🟢 GIAI ĐOẠN 1: MÔI TRƯỜNG & HẠ TẦNG DỮ LIỆU TRONG R (SPRINT 1)
**Mục tiêu:** Thiết lập môi trường R chuẩn mực, cô lập phụ thuộc bằng `renv`, nạp dữ liệu RAF-DB và chuyển đổi bộ kiểm toán rò rỉ dữ liệu sang R.

#### Công việc cụ thể:
1. **Khởi tạo môi trường R (`renv`):**
   * Cài đặt R 4.3+ trên hệ thống.
   * Khởi tạo dự án: `renv::init()`.
   * Cài đặt các gói cốt lõi:
     ```R
     install.packages(c("torch", "torchvision", "tidyverse", "jsonlite", 
                        "imager", "magick", "yardstick", "ggplot2", "shiny", "bslib"))
     ```
   * Kích hoạt backend GPU LibTorch cho R:
     ```R
     torch::install_torch(type = "cuda") # hoặc "cpu" nếu không có NVIDIA GPU
     ```
2. **Cấu trúc thư mục mới trên nhánh `r-migration`:**
   ```text
   ├── R/
   │   ├── dataset.R          # S3 Class nạp ảnh RAF-DB, Masked, Grouped Split
   │   ├── face_detector.R    # Phát hiện mặt & Căn chỉnh tương đồng 5 điểm
   │   ├── models.R           # Kiến trúc HSEmotion / Nạp TorchScript
   │   ├── gradcam.R          # Grad-CAM hook trên R torch
   │   ├── inference.R        # Predictor & IoU FaceTracker
   │   └── evaluate.R         # Đánh giá bằng yardstick & xuất ggplot2
   ├── scripts/
   │   ├── check_leakage.R    # Kiểm toán dHash bằng R
   │   └── export_jit.py      # Script Python xuất mô hình sang TorchScript (.pt)
   ├── app.R                  # Ứng dụng R Shiny Dashboard hoàn chỉnh
   ├── renv.lock              # Khóa phiên bản gói R
   └── docs/R_MIGRATION_PLAN.md
   ```
3. **Chuyển đổi `scripts/check_leakage.py` $\to$ `scripts/check_leakage.R`:**
   * Sử dụng gói `imager` hoặc `magick` để đọc ảnh, tính ma trận gradient và tính mã băm dHash 64-bit.
   * Xác thực lại trên 25,144 ảnh thực tế để đảm bảo kết quả đối khớp 100% với báo cáo Python.

---

### 🟢 GIAI ĐOẠN 2: CHUYỂN GIAO MÔ HÌNH, SUY LUẬN & GRAD-CAM (SPRINT 2)
**Mục tiêu:** Chuyển giao các checkpoint trọng số từ PyTorch sang R, triển khai suy luận và thuật toán Grad-CAM trên nền tảng `torch` for R.

#### Công việc cụ thể:
1. **Xuất mô hình sang TorchScript JIT (`scripts/export_jit.py`):**
   * Vì R không có thư viện `timm` nguyên bản, cách tiếp cận chuẩn mực của ngành AI là Trace/Script mô hình PyTorch thành file TorchScript độc lập:
     ```python
     # Chạy trên Python để xuất sang JIT
     import torch
     from models import load_checkpoint_model
     model, ckpt = load_checkpoint_model("output/best_model_mask_aware.pth")
     model.eval()
     dummy_input = torch.randn(1, 3, 224, 224)
     traced_model = torch.jit.trace(model, dummy_input)
     traced_model.save("output/best_model_mask_aware_jit.pt")
     ```
2. **Nạp mô hình nguyên bản trong R (`R/models.R`):**
   ```R
   library(torch)
   load_fer_model <- function(model_path, device = "cpu") {
     model <- torch::jit_load(model_path)
     model$to(device = torch_device(device))
     model$eval()
     return(model)
   }
   ```
3. **Căn chỉnh khuôn mặt trong R (`R/face_detector.R`):**
   * Lập trình phép biến đổi tương đồng 2D (Similarity Transform) bằng đại số ma trận thuần R (`solve()` hoặc SVD phân rã Procrustes).
   * Ánh xạ 5 landmark của YuNet vào Template RAF-DB chuẩn:
     $$\mathbf{P}_{\text{RAF}} = [[0.27, 0.32], [0.70, 0.32], [0.48, 0.57], [0.30, 0.73], [0.69, 0.73]]^\top$$
4. **Triển khai Grad-CAM trên R `torch` (`R/gradcam.R`):**
   * Đăng ký Tensor Hook trong R:
     ```R
     # Bắt gradient tầng tích chập cuối cùng trong R
     handle <- target_layer$register_backward_hook(function(module, grad_input, grad_output) {
       grad_storage$grad <<- grad_output[[1]]$clone()
     })
     ```
   * Tính trọng số kênh $\alpha_k^c$ và tạo bản đồ nhiệt bằng hàm `torch_relu()`.

---

### 🟢 GIAI ĐOẠN 3: XÂY DỰNG R SHINY WEB DASHBOARD (`app.R`) (SPRINT 3)
**Mục tiêu:** Thay thế hoàn toàn Streamlit bằng ứng dụng **R Shiny** đẳng cấp cao, giao diện Dark Theme hiện đại, hỗ trợ phân tích đa khuôn mặt và so sánh đối đầu.

#### Công việc cụ thể:
1. **Thiết kế Giao diện UI với `bslib`:**
   * Sử dụng chủ đề Dark Mode (`bs_theme(bootswatch = "darkly", primary = "#00bc8c")`).
   * Bố cục Sidebar + Tabset Panel trực quan:
     * **Tab 1: Single Face & XAI Inspector:** Xem ảnh, thanh xác suất 7 cảm xúc (`plotly`), bản đồ nhiệt Grad-CAM tương tác.
     * **Tab 2: Head-to-Head Comparison:** Đặt 2 mô hình (Baseline vs Mask-Aware) cạnh nhau, chứng minh trực tiếp sự vượt trội 22.7% trên ảnh khẩu trang.
     * **Tab 3: Multi-Face Crowd Scanner:** Tải ảnh nhóm đông người, hiển thị bounding box từng người kèm nhãn.
     * **Tab 4: Live Webcam Snapshot:** Chụp ảnh trực tiếp từ camera trình duyệt để phân tích cảm xúc tức thì.
2. **Quản lý Trạng thái Reactive & Tối ưu Bộ nhớ:**
   * Sử dụng `shiny::reactiveVal()` để lưu trữ frame ảnh và danh sách khuôn mặt đã detect.
   * Áp dụng `future` / `promises` nếu cần phân luồng bất đồng bộ để giao diện không bị đóng băng khi tính toán Grad-CAM.

---

### 🟢 GIAI ĐOẠN 4: HUẤN LUYỆN THUẦN R & BÁO CÁO KHOA HỌC (SPRINT 4)
**Mục tiêu:** Viết lại toàn bộ vòng lặp huấn luyện bằng R `torch` (cho mục đích nghiên cứu học thuật) và xây dựng hệ thống báo cáo Quarto tự động.

#### Công việc cụ thể:
1. **Lập trình Hàm Mất mát & Huấn luyện trong R (`R/train.R`):**
   * Viết lớp tùy biến `logit_adjusted_loss` kế thừa `nn_module` trong R:
     ```R
     logit_adjust_loss <- nn_module(
       initialize = function(class_prior, tau = 1.0) {
         self$tau <- tau
         self$log_prior <- torch_tensor(tau * log(pmax(class_prior, 1e-8)))
       },
       forward = function(y_pred, y_true) {
         adjusted_logits <- y_pred + self$log_prior$to(device = y_pred$device)
         nnf_cross_entropy(adjusted_logits, y_true)
       }
     )
     ```
   * Tái lập thuật toán quét tìm $t = +0.85$ trong `R/calibrate.R`.
2. **Hệ Thống Báo Cáo Tái Lập Quarto (`report.qmd`):**
   * Tự động đọc file kết quả `best_model_mask_aware_tta_eval.json`.
   * Dùng `ggplot2` vẽ ma trận nhầm lẫn chuẩn xuất bản quốc tế.
   * Biên dịch ra định dạng HTML Dashboard hoặc PDF bài báo khoa học chỉ bằng 1 dòng lệnh:
     ```bash
     quarto render report.qmd --to pdf
     ```

---

## ⚠️ 5. MA TRẬN ĐÁNH GIÁ RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro kỹ thuật tiềm ẩn | Mức độ | Khả năng xảy ra | Phương án giải quyết dự phòng (Mitigation Strategy) |
| :--- | :---: | :---: | :--- |
| **1. Cài đặt LibTorch CUDA trên Windows gặp lỗi DLL** | Cao | Trung bình | Tự động tải LibTorch qua `torch::install_torch(type="cuda")`. Nếu gặp lỗi driver, kích hoạt chế độ suy luận CPU (vẫn đạt 25–35 FPS trên ảnh đơn). |
| **2. R thiếu thư viện `timm` để load backbone** | Trung bình | Chắc chắn | **Giải quyết triệt để bằng TorchScript JIT**: Export model từ PyTorch sang file `.pt`. R nạp JIT không cần định nghĩa lại kiến trúc mạng từ đầu. |
| **3. Tốc độ đọc webcam liên tục trong R Shiny chậm hơn Python** | Trung bình | Cao | Trong môi trường Web, cho phép người dùng dùng chế độ **Webcam Snapshot (chụp ảnh frame)** thay vì stream liên tục 60 FPS, hoặc dùng JavaScript client-side capture. |
| **4. Sai số tính toán giữa PyTorch và R torch** | Thấp | Thấp | Thiết lập bài kiểm tra `tests/testthat/test_numerical_parity.R`: So sánh output logits giữa 2 ngôn ngữ trên 100 ảnh mẫu, yêu cầu sai số $\max \|y_{\text{python}} - y_{\text{R}}\| < 10^{-5}$. |

---

## 🎯 6. KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG

Kế hoạch chuyển đổi sang ngôn ngữ R trên nhánh **`r-migration`** là hoàn toàn khả thi về mặt kỹ thuật, mang lại giá trị gia tăng rất lớn về mặt:
1. **Uy tín học thuật:** Sử dụng R chứng minh năng lực đa ngôn ngữ và tận dụng thế mạnh thống kê của R.
2. **Chất lượng giao diện:** R Shiny đem lại trải nghiệm bảng điều khiển phân tích số liệu trực quan hơn nhiều so với Streamlit.
3. **Bảo toàn thành quả:** Nhờ chuẩn định dạng **TorchScript JIT**, ta không cần huấn luyện lại từ đầu (tiết kiệm hàng chục giờ GPU), mà kế thừa trực tiếp mô hình đỉnh cao **88.43% / 79.29%** đã đạt được.

Nhánh `r-migration` đã được khởi tạo sẵn sàng để bắt đầu triển khai Sprint 1 bất cứ lúc nào!
