# ==============================================================================
# R/models.R: Nạp và quản lý mô hình FER trong môi trường R
# Hỗ trợ:
# 1. Nạp trực tiếp mô hình TorchScript JIT (.pt) qua thư viện torch for R (LibTorch)
# 2. Suy luận logits, xác suất Softmax và kiểm tra metadata cấu hình
# ==============================================================================

#' Nạp mô hình TorchScript JIT trong R
#'
#' @param model_path Đường dẫn đến file .pt (Ví dụ: "output/best_model_mask_aware_jit.pt")
#' @param device Thiết bị tính toán ("cpu" hoặc "cuda")
#' @return Danh sách chứa mô hình torch và metadata
load_fer_jit_model <- function(model_path = "output/best_model_mask_aware_jit.pt", device = "cpu") {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Thư viện 'torch' chưa được cài đặt. Hãy chạy: install.packages('torch') và torch::install_torch()")
  }
  
  if (!file.exists(model_path)) {
    stop(paste("Không tìm thấy file mô hình TorchScript tại:", model_path))
  }
  
  message(paste("📦 Đang nạp mô hình TorchScript JIT:", basename(model_path), "..."))
  target_device <- torch::torch_device(device)
  
  # Nạp mô hình TorchScript JIT
  model <- torch::jit_load(model_path)
  model$to(device = target_device)
  model$eval()
  
  # Đọc metadata nếu có
  meta_file <- sub("\\.pt$", "_meta.json", model_path)
  metadata <- list(
    classes = c("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
    input_size = 224,
    mean = c(0.485, 0.456, 0.406),
    std = c(0.229, 0.224, 0.225)
  )
  
  if (file.exists(meta_file) && requireNamespace("jsonlite", quietly = TRUE)) {
    meta_json <- jsonlite::fromJSON(meta_file)
    metadata$classes <- meta_json$classes
    metadata$input_size <- meta_json$input_size
    metadata$mean <- meta_json$mean
    metadata$std <- meta_json$std
    metadata$model_name <- meta_json$model_name
    metadata$best_val_f1 <- meta_json$best_val_f1
  }
  
  message(paste("✅ Đã nạp thành công mô hình! Input size:", metadata$input_size, "px"))
  
  list(
    model = model,
    device = target_device,
    metadata = metadata
  )
}

#' Dự đoán xác suất từ tensor ảnh [B, 3, H, W]
#'
#' @param fer_model Đối tượng mô hình trả về từ load_fer_jit_model
#' @param image_tensor Tensor ảnh đã chuẩn hóa [B, 3, H, W]
#' @param tta Có áp dụng Test-Time Augmentation (lật ngang) hay không
#' @return Ma trận xác suất [B, 7]
predict_fer_tensor <- function(fer_model, image_tensor, tta = FALSE) {
  torch::with_no_grad({
    model <- fer_model$model
    x <- image_tensor$to(device = fer_model$device)
    
    logits <- model(x)
    
    if (tta) {
      # Lật ngang theo chiều width (dim = 4 trong R torch)
      x_flip <- torch::torch_flip(x, dims = c(4))
      logits_flip <- model(x_flip)
      logits <- 0.5 * (logits + logits_flip)
    }
    
    probs <- torch::nnf_softmax(logits, dim = 2)
    as.matrix(probs$cpu())
  })
}
