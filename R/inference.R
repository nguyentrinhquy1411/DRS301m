# ==============================================================================
# R/inference.R: Pipeline suy luận cảm xúc và Face Tracker ổn định chuỗi thời gian
# ==============================================================================

source("R/models.R")
source("R/dataset.R")

#' Tính chỉ số giao trên hợp (IoU) giữa 2 bounding box c(x, y, w, h)
calc_bbox_iou <- function(box_a, box_b) {
  ax <- box_a[1]; ay <- box_a[2]; aw <- box_a[3]; ah <- box_a[4]
  bx <- box_b[1]; by <- box_b[2]; bw <- box_b[3]; bh <- box_b[4]
  
  inter_w <- max(0, min(ax + aw, bx + bw) - max(ax, bx))
  inter_h <- max(0, min(ay + ah, by + bh) - max(ay, by))
  inter_area <- inter_w * inter_h
  
  union_area <- (aw * ah) + (bw * bh) - inter_area
  if (union_area <= 0) return(0.0)
  inter_area / union_area
}

#' Face Tracker quản lý ID và làm mượt xác suất theo thời gian (EMA Smoothing)
new_face_tracker <- function(alpha = 0.4, iou_thresh = 0.3, max_missed = 10) {
  state <- new.env(parent = emptyenv())
  state$tracks <- list() # track_id -> list(bbox, probs, missed)
  state$next_id <- 1
  state$alpha <- alpha
  state$iou_thresh <- iou_thresh
  state$max_missed <- max_missed
  
  reset <- function() {
    state$tracks <- list()
    state$next_id <- 1
  }
  
  update <- function(bboxes, probs_matrix) {
    n_det <- length(bboxes)
    if (n_det == 0) {
      # Tăng biến đếm missed cho toàn bộ tracks hiện tại
      for (tid in names(state$tracks)) {
        state$tracks[[tid]]$missed <- state$tracks[[tid]]$missed + 1
        if (state$tracks[[tid]]$missed > state$max_missed) {
          state$tracks[[tid]] <- NULL
        }
      }
      return(list())
    }
    
    # Ghép cặp tham lam dựa trên IoU
    assigned_det <- rep(NA, n_det)
    used_tracks <- c()
    
    track_ids <- names(state$tracks)
    if (length(track_ids) > 0) {
      iou_pairs <- list()
      for (i in 1:n_det) {
        for (tid in track_ids) {
          iou_val <- calc_bbox_iou(bboxes[[i]], state$tracks[[tid]]$bbox)
          if (iou_val >= state$iou_thresh) {
            iou_pairs[[length(iou_pairs) + 1]] <- list(det_idx = i, track_id = tid, iou = iou_val)
          }
        }
      }
      
      # Sắp xếp IoU giảm dần
      if (length(iou_pairs) > 0) {
        iou_pairs <- iou_pairs[order(sapply(iou_pairs, function(p) p$iou), decreasing = TRUE)]
        for (pair in iou_pairs) {
          if (is.na(assigned_det[pair$det_idx]) && !(pair$track_id %in% used_tracks)) {
            assigned_det[pair$det_idx] <- pair$track_id
            used_tracks <- c(used_tracks, pair$track_id)
          }
        }
      }
    }
    
    # Cập nhật hoặc tạo track mới
    results <- list()
    for (i in 1:n_det) {
      cur_bbox <- bboxes[[i]]
      cur_probs <- probs_matrix[i, ]
      
      if (!is.na(assigned_det[i])) {
        # Đã ghép với track cũ -> Làm mượt EMA
        tid <- assigned_det[i]
        old_probs <- state$tracks[[tid]]$probs
        smoothed_probs <- (1 - state$alpha) * old_probs + state$alpha * cur_probs
        
        state$tracks[[tid]]$bbox <- cur_bbox
        state$tracks[[tid]]$probs <- smoothed_probs
        state$tracks[[tid]]$missed <- 0
        
        results[[i]] <- list(track_id = as.integer(tid), probs = smoothed_probs)
      } else {
        # Tạo Track ID mới
        new_tid <- as.character(state$next_id)
        state$next_id <- state$next_id + 1
        
        state$tracks[[new_tid]] <- list(
          bbox = cur_bbox,
          probs = cur_probs,
          missed = 0
        )
        results[[i]] <- list(track_id = as.integer(new_tid), probs = cur_probs)
      }
    }
    
    # Loại bỏ tracks mất dấu quá lâu
    for (tid in names(state$tracks)) {
      if (!(tid %in% used_tracks) && !(tid %in% assigned_det)) {
        state$tracks[[tid]]$missed <- state$tracks[[tid]]$missed + 1
        if (state$tracks[[tid]]$missed > state$max_missed) {
          state$tracks[[tid]] <- NULL
        }
      }
    }
    
    results
  }
  
  list(reset = reset, update = update)
}
