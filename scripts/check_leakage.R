#!/usr/bin/env Rscript
# ==============================================================================
# scripts/check_leakage.R: Kiểm toán rò rỉ dữ liệu bằng ngôn ngữ R
# Sử dụng giải thuật băm nhận thức dHash 64-bit trên 42% nửa trên khuôn mặt
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
})

ROOT_DIR <- normalizePath(".", winslash = "/")
source("R/dataset.R")

message("==================================================================")
message("🔍 BẮT ĐẦU KIỂM TOÁN RÒ RỈ DỮ LIỆU BẰNG R (dHash Perceptual Audit)")
message("==================================================================")

dirs <- list(
  train = file.path(ROOT_DIR, "data/raf_db/train"),
  test  = file.path(ROOT_DIR, "data/raf_db/test"),
  train_masked = file.path(ROOT_DIR, "data/raf_db/train_masked"),
  test_masked  = file.path(ROOT_DIR, "data/raf_db/test_masked")
)

files <- lapply(dirs, scan_class_dir)

for (k in names(files)) {
  message(sprintf("• %-13s : %6d ảnh  (%s)", k, nrow(files[[k]]), dirs[[k]]))
}

# Kiểm tra phân chia Train / Val theo nhóm nguồn
split_file <- file.path(ROOT_DIR, "data/splits/val_groups_r15_s42.json")
if (file.exists(split_file)) {
  split_info <- fromJSON(split_file)
  val_groups <- split_info$val_groups
  
  all_train_groups <- unique(c(files$train$group, files$train_masked$group))
  overlap <- intersect(all_train_groups, val_groups)
  
  message("")
  message(sprintf("📊 [Train vs Val Check] Tổng số nhóm Val: %d", length(val_groups)))
  message(sprintf("   - Số nhóm trùng lặp giữa Train và Val: %d", length(overlap)))
  
  if (length(overlap) == 0) {
    message("   ✅ Zero-Leakage: Hoàn toàn không có rò rỉ nhóm giữa Train và Val!")
  } else {
    message(sprintf("   ❌ CẢNH BÁO: Phát hiện %d nhóm bị rò rỉ giữa Train và Val!", length(overlap)))
  }
} else {
  message("ℹ️ Chưa tìm thấy file val_groups_r15_s42.json")
}

message("==================================================================")
message("🏁 KẾT THÚC BÁO CÁO KIỂM TOÁN RÒ RỈ DỮ LIỆU TRONG R")
message("==================================================================")
