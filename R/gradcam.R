# ==============================================================================
# R/gradcam.R: Triển khai Explainable AI (Grad-CAM) trong R torch
# ==============================================================================

#' Tạo bản đồ nhiệt Grad-CAM từ Feature Map và Gradient
#'
#' @param feature_map Tensor đặc trưng tầng cuối [B, C, H, W]
#' @param gradients Tensor đạo hàm riêng theo feature map [B, C, H, W]
#' @return Mảng ma trận bản đồ nhiệt 2D [H, W] chuẩn hóa [0, 1]
compute_gradcam_heatmap <- function(feature_map, gradients) {
  # 1. Tính trọng số alpha bằng Global Average Pooling trên gradients
  # Dim 3 và 4 là chiều không gian H, W trong R torch
  alpha <- gradients$mean(dim = c(3, 4), keepdim = TRUE)
  
  # 2. Tổ hợp tuyến tính có trọng số của các feature maps
  cam <- (alpha * feature_map)$sum(dim = 2, keepdim = TRUE)
  
  # 3. Đi qua hàm chỉnh lưu ReLU
  cam <- torch::torch_relu(cam)
  
  # 4. Chuẩn hóa min-max về khoảng [0, 1]
  cam_np <- as.matrix(cam$squeeze()$cpu())
  cam_min <- min(cam_np)
  cam_max <- max(cam_np)
  
  if (cam_max - cam_min > 1e-8) {
    (cam_np - cam_min) / (cam_max - cam_min)
  } else {
    matrix(0, nrow = nrow(cam_np), ncol = ncol(cam_np))
  }
}

#' Phủ bản đồ nhiệt màu (JET colormap) lên ảnh gốc trong R
#'
#' @param original_img Mảng ảnh gốc RGB [H, W, 3] với giá trị [0, 1]
#' @param heatmap Ma trận bản đồ nhiệt [H, W] với giá trị [0, 1]
#' @param alpha Độ trong suốt của bản đồ nhiệt (mặc định 0.5)
#' @return Mảng ảnh đã phủ nhiệt [H, W, 3]
overlay_gradcam_on_image <- function(original_img, heatmap, alpha = 0.5) {
  h <- nrow(heatmap)
  w <- ncol(heatmap)
  
  # Tạo bảng màu giả lập JET: Xanh dương (lạnh) -> Vàng -> Đỏ (nóng)
  jet_r <- pmin(pmax(1.5 - abs(4 * heatmap - 3), 0), 1)
  jet_g <- pmin(pmax(1.5 - abs(4 * heatmap - 2), 0), 1)
  jet_b <- pmin(pmax(1.5 - abs(4 * heatmap - 1), 0), 1)
  
  colored_cam <- array(0, dim = c(h, w, 3))
  colored_cam[, , 1] <- jet_r
  colored_cam[, , 2] <- jet_g
  colored_cam[, , 3] <- jet_b
  
  # Trộn alpha với ảnh gốc
  out_img <- (1 - alpha) * original_img + alpha * colored_cam
  pmin(pmax(out_img, 0), 1)
}
