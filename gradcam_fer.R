# gradcam_fer.R -- dung R de dieu khien pipeline Grad-CAM cho FER2013 ResNet-18
#
# Model la checkpoint PyTorch (.pt) nen phan tinh toan model + backward (Grad-CAM)
# van chay bang Python, duoc goi tu R qua goi `reticulate`.
# Phan Task 2 (Jet colormap + alpha blending) duoc viet bang R thuan o ham overlay_heatmap_r().
#
# Cach dung:
#   source("gradcam_fer.R")
#   g   <- init_gradcam("duong/dan/toi/project")
#   res <- explain_image(g, "examples/input.jpg", class = "predicted", alpha = 0.4)
#   res$prediction; res$confidence; res$probabilities

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)

# ---------------------------------------------------------------------------------------
# 1. Khoi tao: nap cac module Python trong project/src (model chi nap MOT lan)
# ---------------------------------------------------------------------------------------
init_gradcam <- function(project_dir = ".", python = NULL, virtualenv = NULL, condaenv = NULL,
                         device = NULL, confidence_threshold = 0.5, use_tta = TRUE) {
  project_dir <- normalizePath(project_dir, mustWork = TRUE)
  if (!is.null(python))     use_python(python, required = TRUE)
  if (!is.null(virtualenv)) use_virtualenv(virtualenv, required = TRUE)
  if (!is.null(condaenv))   use_condaenv(condaenv, required = TRUE)

  for (pkg in c("torch", "timm", "cv2", "numpy")) {
    if (!py_module_available(pkg)) {
      stop(sprintf(paste0("Python thieu goi '%s'. Cai bang:\n",
                          "  reticulate::py_install(c('torch','torchvision','timm','opencv-python','numpy'), pip = TRUE)"),
                   pkg))
    }
  }
  # convert = FALSE: giu doi tuong Python nguyen ban (anh uint8 khong bi doi sang int32);
  # chi chuyen sang R khi can bang py_to_r().
  inf <- import_from_path("src.inference", path = project_dir, convert = FALSE)
  pre <- import_from_path("src.preprocessing", path = project_dir, convert = FALSE)
  cv2 <- import("cv2", convert = FALSE)

  explainer <- inf$EmotionExplainer(device = device,
                                    confidence_threshold = confidence_threshold,
                                    use_tta = use_tta)
  structure(list(ex = explainer, inf = inf, pre = pre, cv2 = cv2, project_dir = project_dir),
            class = "gradcam_session")
}

# ---------------------------------------------------------------------------------------
# 2. Task 2 bang R thuan: Jet colormap + alpha blending
#    overlay = (1 - alpha) * original + alpha * jet(heatmap)
# ---------------------------------------------------------------------------------------
# Bang mau Jet kinh dien (9 diem cach deu): xanh dam -> xanh -> cyan -> vang -> do -> do dam
jet_palette <- function(n = 256) {
  colorRampPalette(c("#00007F", "#0000FF", "#007FFF", "#00FFFF", "#7FFF7F",
                     "#FFFF00", "#FF7F00", "#FF0000", "#7F0000"))(n)
}

# heatmap: matrix [0,1]; tra ve array h x w x 3 (RGB, 0..1)
jet_rgb <- function(heatmap) {
  heat <- pmin(pmax(heatmap, 0), 1)
  idx  <- round(heat * 255) + 1                    # chi so 1..256
  pal  <- t(col2rgb(jet_palette(256))) / 255       # 256 x 3
  out  <- array(0, dim = c(nrow(heat), ncol(heat), 3))
  for (ch in 1:3) out[, , ch] <- matrix(pal[idx, ch], nrow(heat), ncol(heat))
  out
}

# Resize heatmap ve kich thuoc anh goc (bilinear, tren ban do vo huong truoc khi to mau)
resize_heatmap <- function(g, heatmap, width, height) {
  r <- g$cv2$resize(r_to_py(heatmap), tuple(as.integer(width), as.integer(height)),
                    interpolation = g$cv2$INTER_LINEAR)
  pmin(pmax(py_to_r(r), 0), 1)
}

# original_rgb: array h x w x 3 trong [0,1] (RGB!). alpha trong [0,1].
overlay_heatmap_r <- function(g, original_rgb, heatmap, alpha = 0.4) {
  stopifnot(alpha >= 0, alpha <= 1, length(dim(original_rgb)) == 3)
  h <- dim(original_rgb)[1]; w <- dim(original_rgb)[2]
  jet <- jet_rgb(resize_heatmap(g, heatmap, w, h))
  (1 - alpha) * original_rgb + alpha * jet
}

# Anh crop tu Python la BGR (OpenCV) -> doi sang RGB cho R. Day la cho de nham mau nhat.
bgr_to_rgb01 <- function(bgr) bgr[, , 3:1, drop = FALSE] / 255

# ---------------------------------------------------------------------------------------
# 3. Giai thich mot anh
# ---------------------------------------------------------------------------------------
# class: "predicted" | ten lop (vd "sad") | chi so 0-6
# mode:  "crop" (ve len mat da cat) | "full" (dan heatmap ve anh goc)
explain_image <- function(g, image_path, class = "predicted", alpha = 0.4, output_dir = "outputs",
                          face_selection = "largest", mode = "crop", detect_face = TRUE,
                          plot = TRUE, save = TRUE) {
  stopifnot(alpha >= 0, alpha <= 1)
  image_path <- normalizePath(image_path, mustWork = TRUE)
  img      <- g$pre$load_image(image_path)                       # anh BGR (Python)
  class_ix <- g$inf$parse_class(class, g$ex$class_names)         # None neu "predicted"
  res      <- g$ex$analyze_image(img, class_ix, detect_face, face_selection)

  status <- py_to_r(res$get("status"))
  if (status != "ok") {
    message(py_to_r(res$get("message")))                         # "No face detected."
    if (save) {
      stem <- tools::file_path_sans_ext(basename(image_path))
      g$inf$save_outputs(g$ex, img, res, output_dir, stem, alpha, mode)
    }
    return(invisible(list(status = status, message = py_to_r(res$get("message")))))
  }

  # Luu 4 anh + JSON bang Python (giong ban CLI), giu nguyen quy uoc ten file
  stem <- tools::file_path_sans_ext(basename(image_path))
  if (save) g$inf$save_outputs(g$ex, img, res, output_dir, stem, alpha, mode)

  faces <- py_to_r(res$get("faces"))
  out <- lapply(seq_along(faces), function(i) {
    f <- faces[[i]]
    probs <- py_to_r(f$probs)
    names(probs) <- unlist(py_to_r(g$ex$class_names))
    list(
      prediction     = py_to_r(f$prediction),
      class_index    = py_to_r(f$class_index),
      confidence     = py_to_r(f$confidence),
      explained      = py_to_r(f$target_name),
      explained_prob = py_to_r(f$target_prob),
      low_confidence = py_to_r(f$low_confidence),
      warnings       = unlist(py_to_r(f$warnings)),
      probabilities  = probs,
      heatmap        = py_to_r(f$heatmap),            # matrix 112 x 112, [0,1]
      crop_rgb       = bgr_to_rgb01(py_to_r(f$crop_bgr))
    )
  })

  if (plot) for (r in out) plot_explanation(g, r, alpha)
  invisible(if (length(out) == 1) out[[1]] else out)
}

# ---------------------------------------------------------------------------------------
# 4. Ve: Original | Grad-CAM | Overlay + bieu do xac suat 7 lop
# ---------------------------------------------------------------------------------------
plot_explanation <- function(g, r, alpha = 0.4) {
  h <- dim(r$crop_rgb)[1]; w <- dim(r$crop_rgb)[2]
  heat_rgb <- jet_rgb(resize_heatmap(g, r$heatmap, w, h))
  overlay  <- overlay_heatmap_r(g, r$crop_rgb, r$heatmap, alpha)

  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1, 2, 3, 4, 4, 4), nrow = 2, byrow = TRUE), heights = c(3, 2))
  par(mar = c(1, 1, 2, 1))
  for (p in list(list(r$crop_rgb, "Original"), list(heat_rgb, "Grad-CAM (Jet)"),
                 list(overlay, sprintf("Overlay (alpha = %.1f)", alpha)))) {
    plot(as.raster(pmin(pmax(p[[1]], 0), 1)), interpolate = FALSE)
    title(p[[2]], cex.main = 1)
  }
  par(mar = c(4, 4, 3, 1))
  cols <- ifelse(names(r$probabilities) == r$prediction, "firebrick", "grey70")
  barplot(r$probabilities, col = cols, ylim = c(0, 1), las = 2, ylab = "Probability",
          main = sprintf("Prediction: %s   Confidence: %.1f%%%s", r$prediction, 100 * r$confidence,
                         if (r$low_confidence) "   (Low confidence)" else ""))
  if (r$explained != r$prediction) {
    mtext(sprintf("Grad-CAM explains: %s (p = %.1f%%)", r$explained, 100 * r$explained_prob),
          side = 3, line = 0, cex = 0.8)
  }
  invisible(NULL)
}

# ---------------------------------------------------------------------------------------
# 5. Ca thu muc anh va video
# ---------------------------------------------------------------------------------------
explain_folder <- function(g, folder, ..., plot = FALSE) {
  files <- list.files(folder, pattern = "\\.(jpe?g|png|bmp|webp|tiff?)$", ignore.case = TRUE, full.names = TRUE)
  if (length(files) == 0) stop("Khong co anh hop le trong: ", folder)
  setNames(lapply(files, function(f) tryCatch(explain_image(g, f, ..., plot = plot),
                                              error = function(e) { message(basename(f), ": ", conditionMessage(e)); NULL })),
           basename(files))
}

# gradcam_every_n: 0 = tat, 1 = moi frame, 10 = moi 10 frame
explain_video <- function(g, source, output_path = "outputs/video_gradcam.mp4", gradcam_every_n = 10L,
                          alpha = 0.4, max_frames = NULL) {
  stats <- g$ex$process_video(source, output_path, as.integer(gradcam_every_n), alpha,
                              max_frames = if (is.null(max_frames)) NULL else as.integer(max_frames))
  py_to_r(stats)
}

# Giai phong hook khi xong viec
close_gradcam <- function(g) invisible(g$ex$close())

# ---------------------------------------------------------------------------------------
# Vi du:
# g <- init_gradcam("project", virtualenv = "~/.virtualenvs/fer")
# r <- explain_image(g, "project/examples/input.jpg")                    # lop duoc du doan
# r <- explain_image(g, "project/examples/input.jpg", class = "sad")     # giai thich lop "sad"
# res <- explain_folder(g, "samples", alpha = 0.5)
# explain_video(g, "clip.mp4", gradcam_every_n = 10)
# close_gradcam(g)
# ---------------------------------------------------------------------------------------
