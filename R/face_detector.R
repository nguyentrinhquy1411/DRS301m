# ==============================================================================
# R/face_detector.R: Căn chỉnh khuôn mặt 5 điểm Landmark về RAF-DB Template
# ==============================================================================

# Template tọa độ 5 điểm chuẩn của RAF-DB aligned (chuẩn hóa [0, 1])
# Thứ tự: Mắt trái, Mắt phải, Mũi, Khóe miệng trái, Khóe miệng phải
RAF_TEMPLATE_5PTS <- matrix(c(
  0.2696, 0.3213,
  0.7026, 0.3157,
  0.4809, 0.5654,
  0.2996, 0.7345,
  0.6892, 0.7288
), nrow = 5, ncol = 2, byrow = TRUE)

#' Tính ma trận biến đổi tương đồng 2D (Similarity Transform)
#' Ánh xạ ma trận điểm nguồn src_pts [5, 2] về template đích dst_pts [5, 2]
#'
#' @param src_pts Ma trận 5x2 tọa độ điểm nguồn
#' @param out_size Kích thước ảnh vuông đầu ra (mặc định 224)
#' @return Ma trận 2x3 biến đổi tương đồng affine
estimate_similarity_transform <- function(src_pts, out_size = 224) {
  dst_pts <- RAF_TEMPLATE_5PTS * out_size
  
  # Căn giữa tọa độ quanh tâm khối
  src_mean <- colMeans(src_pts)
  dst_mean <- colMeans(dst_pts)
  
  src_centered <- sweep(src_pts, 2, src_mean, "-")
  dst_centered <- sweep(dst_pts, 2, dst_mean, "-")
  
  # Tính scale và góc xoay dựa trên phân rã SVD (Umeyama / Procrustes algorithm)
  # H = src^T dst = U S V^T  ->  R = V D U^T (ánh xạ cột: dst ≈ s R src + t)
  H <- t(src_centered) %*% dst_centered
  svd_res <- svd(H)
  D <- diag(c(1, sign(det(svd_res$v %*% t(svd_res$u)))))
  R <- svd_res$v %*% D %*% t(svd_res$u)

  scale <- sum(svd_res$d * diag(D)) / sum(src_centered^2)
  
  sR <- scale * R
  translation <- dst_mean - (sR %*% src_mean)
  
  # Ma trận 2x3 [sR | translation]
  cbind(sR, translation)
}

#' Cắt khuôn mặt vuông bao quanh Bounding Box nếu không có landmark
#'
#' @param bbox Vector c(x, y, w, h)
#' @param scale Hệ số mở rộng (mặc định 1.0)
#' @return Vector c(x1, y1, x2, y2)
get_square_crop_box <- function(bbox, scale = 1.0) {
  x <- bbox[1]
  y <- bbox[2]
  w <- bbox[3]
  h <- bbox[4]
  
  cx <- x + w / 2
  cy <- y + h / 2
  side <- max(w, h) * scale
  
  c(
    x1 = round(cx - side / 2),
    y1 = round(cy - side / 2),
    x2 = round(cx + side / 2),
    y2 = round(cy + side / 2)
  )
}
