# ==============================================================================
# R/dataset.R: Nạp và xử lý tập dữ liệu RAF-DB và MaskTheFace trong R
# ==============================================================================

CLASSES <- c("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMAGENET_MEAN <- c(0.485, 0.456, 0.406)
IMAGENET_STD  <- c(0.229, 0.224, 0.225)

#' Lấy nhóm nguồn từ tên tệp để chống rò rỉ dữ liệu
#' Ví dụ: "raf_train_00012_surgical.jpg" -> "raf_train_00012"
get_source_group <- function(file_path) {
  fname <- basename(file_path)
  m <- regexpr("^raf_(train|test)_[0-9]+", fname)
  if (m[1] != -1) {
    regmatches(fname, m)
  } else {
    sub("\\.[^.]+$", "", fname)
  }
}

#' Nạp danh sách mẫu ảnh từ thư mục phân loại theo lớp
#'
#' @param root_dir Thư mục gốc chứa các thư mục con 7 lớp
#' @return Data frame gồm path, class, class_idx, group
scan_class_dir <- function(root_dir) {
  if (!dir.exists(root_dir)) return(data.frame())
  
  rows <- list()
  for (cls in CLASSES) {
    cls_dir <- file.path(root_dir, cls)
    if (dir.exists(cls_dir)) {
      files <- list.files(cls_dir, pattern = "\\.(jpg|jpeg|png)$", full.names = TRUE, ignore.case = TRUE)
      if (length(files) > 0) {
        idx <- which(CLASSES == cls) # 1-indexed trong R
        rows[[cls]] <- data.frame(
          path = files,
          class = cls,
          class_idx = idx,
          group = sapply(files, get_source_group),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  
  if (length(rows) == 0) return(data.frame())
  do.call(rbind, rows)
}

#' Tiền xử lý mảng ảnh RGB [H, W, 3] thành Tensor chuẩn hóa ImageNet [1, 3, 224, 224]
#'
#' @param img_array Mảng 3 chiều [H, W, 3] với giá trị [0, 255]
#' @param target_size Kích thước ảnh vuông đầu ra (mặc định 224)
#' @return Tensor R torch [1, 3, target_size, target_size]
preprocess_image_array <- function(img_array, target_size = 224) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Thư viện 'torch' chưa được cài đặt")
  }
  
  # Chuyển về float32 và chuẩn hóa về [0, 1]
  arr <- as.array(img_array) / 255.0
  
  # Chuyển layout từ [H, W, C] sang [C, H, W]
  t_img <- torch::torch_tensor(arr, dtype = torch::torch_float32())$permute(c(3, 1, 2))
  
  # Chuẩn hóa ImageNet Mean / Std
  mean_t <- torch::torch_tensor(IMAGENET_MEAN, dtype = torch::torch_float32())$view(c(3, 1, 1))
  std_t  <- torch::torch_tensor(IMAGENET_STD, dtype = torch::torch_float32())$view(c(3, 1, 1))
  norm_t <- (t_img - mean_t) / std_t
  
  # Thêm chiều batch [1, 3, H, W]
  norm_t$unsqueeze(1)
}
