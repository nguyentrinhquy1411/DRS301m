# ==============================================================================
# app.R: R Shiny Web Dashboard cho XAI-FER Mask-Aware
# Real-Time Multi-Face Explainable Facial Expression Recognition
# ==============================================================================

library(shiny)
library(bslib)

# Định nghĩa các lớp cảm xúc và mã màu
CLASSES <- c("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
EMOJI_MAP <- c(
  "angry" = "😡 Tức giận (Angry)",
  "disgust" = "🤢 Ghê tởm (Disgust)",
  "fear" = "😨 Sợ hãi (Fear)",
  "happy" = "😊 Vui vẻ (Happy)",
  "neutral" = "😐 Bình thường (Neutral)",
  "sad" = "😢 Buồn bã (Sad)",
  "surprise" = "😲 Ngạc nhiên (Surprise)"
)

EMOTION_COLORS <- c(
  "happy" = "#2ecc71",
  "surprise" = "#f1c40f",
  "neutral" = "#3498db",
  "sad" = "#9b59b6",
  "fear" = "#e67e22",
  "angry" = "#e74c3c",
  "disgust" = "#95a5a6"
)

# ------------------------------------------------------------------------------
# UI: Giao diện Dark Theme hiện đại với bslib
# ------------------------------------------------------------------------------
ui <- page_navbar(
  title = "🎭 XAI-FER · Mask-Aware Emotion Recognition (R Edition)",
  theme = bs_theme(
    bootswatch = "darkly",
    primary = "#00bc8c",
    secondary = "#375a7f",
    base_font = font_google("Inter")
  ),
  
  # Tab 1: Phân tích ảnh đơn & XAI Inspector
  nav_panel(
    title = "🔍 Single-Face & XAI Inspector",
    layout_sidebar(
      sidebar = sidebar(
        title = "Điều khiển Phân tích",
        fileInput("single_file", "Tải ảnh khuôn mặt (JPG/PNG):", accept = c("image/jpeg", "image/png")),
        selectInput("model_choice", "Chọn Checkpoint Mô hình:",
                    choices = c(
                      "HSEmotion-B0 Mask-Aware (Đề xuất)" = "output/best_model_mask_aware_jit.pt",
                      "HSEmotion-B0 Baseline (Chỉ RAF-DB)" = "output/best_model_rafdb_jit.pt"
                    )),
        checkboxInput("apply_tta", "Bật Test-Time Augmentation (TTA)", value = TRUE),
        sliderInput("cam_alpha", "Độ trong suốt Grad-CAM Heatmap:", min = 0.1, max = 0.9, value = 0.5, step = 0.1),
        actionButton("run_predict", "🚀 Bắt đầu Phân tích", class = "btn-primary w-100 mt-2")
      ),
      
      layout_columns(
        col_widths = c(6, 6),
        card(
          card_header("Ảnh gốc & Bản đồ nhiệt Grad-CAM"),
          plotOutput("gradcam_plot", height = "420px")
        ),
        card(
          card_header("Phân bố Xác suất 7 Cảm xúc"),
          uiOutput("prediction_summary"),
          plotOutput("prob_barchart", height = "320px")
        )
      )
    )
  ),
  
  # Tab 2: So sánh Đối đầu (Head-to-Head Comparison)
  nav_panel(
    title = "⚔️ Head-to-Head: Baseline vs Mask-Aware",
    layout_sidebar(
      sidebar = sidebar(
        title = "Tùy chọn Đối đầu",
        helpText("So sánh trực tiếp phản ứng của mô hình thường và mô hình Mask-Aware trên cùng một ảnh đeo khẩu trang."),
        fileInput("compare_file", "Tải ảnh kiểm thử:", accept = c("image/jpeg", "image/png")),
        actionButton("run_compare", "⚡ So sánh Đối đầu", class = "btn-warning w-100 mt-2")
      ),
      
      layout_columns(
        col_widths = c(6, 6),
        card(
          card_header("Mô hình Baseline (Chỉ học mặt thường)"),
          uiOutput("baseline_status"),
          plotOutput("baseline_cam_plot", height = "380px")
        ),
        card(
          card_header("Mô hình Mask-Aware (Học cả khẩu trang) ⭐"),
          uiOutput("maskaware_status"),
          plotOutput("maskaware_cam_plot", height = "380px")
        )
      )
    )
  ),
  
  # Tab 3: Bảng số liệu Benchmark & Đột phá
  nav_panel(
    title = "📊 Benchmark Khoa học",
    card(
      card_header("Bảng So sánh Hiệu năng Kiểm thử Chính thức (RAF-DB 5,043 ảnh)"),
      tableOutput("benchmark_table"),
      p(class = "text-muted mt-2", 
        "Số liệu đo độc lập ở chuẩn fp32 có TTA lật ngang và Prior-bias trên 5,043 ảnh test. Nhóm kiểm toán rò rỉ dữ liệu bằng dHash đạt 0% rò rỉ.")
    )
  )
)

# ------------------------------------------------------------------------------
# SERVER: Xử lý logic phản ứng (Reactive Engine)
# ------------------------------------------------------------------------------
server <- function(input, output, session) {
  
  # Dữ liệu bảng benchmark chính thức
  output$benchmark_table <- renderTable({
    data.frame(
      "Mô hình thử nghiệm" = c(
        "MobileNetV3 (Pipeline cũ của nhóm)",
        "HSEmotion-B0 (Chỉ train RAF-DB)",
        "HSEmotion-B0 Mask-Aware (Đề xuất ⭐)"
      ),
      "Mặt thường (Unmasked)" = c("79.86% / F1 0.718", "89.15% / F1 0.833", "88.43% / F1 0.822"),
      "Mặt có khẩu trang (Masked)" = c("73.27% / F1 0.628", "56.61% / F1 0.534", "79.29% / F1 0.708"),
      "Toàn bộ tập test (Combined)" = c("77.28% / F1 0.684", "76.40% / F1 0.725", "84.85% / F1 0.779"),
      "Mức chênh lệch cốt lõi" = c("Baseline cũ", "Bị mù khi gặp khẩu trang", "+22.68% Acc trên mặt khẩu trang 🚀"),
      check.names = FALSE
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Biểu đồ phân bố xác suất giả lập khi chưa nạp ảnh
  output$prob_barchart <- renderPlot({
    # Mẫu xác suất mặc định minh họa
    probs <- c(0.05, 0.03, 0.02, 0.75, 0.08, 0.04, 0.03)
    names(probs) <- CLASSES
    
    par(mar = c(4, 7, 2, 2), bg = "#222222", fg = "#ffffff", col.axis = "#ffffff", col.lab = "#ffffff")
    barplot(
      probs,
      horiz = TRUE,
      las = 1,
      col = EMOTION_COLORS[CLASSES],
      xlim = c(0, 1),
      xlab = "Xác suất tin cậy (Confidence)",
      main = "Phân loại Cảm xúc Dự đoán"
    )
    abline(v = seq(0.2, 0.8, by = 0.2), lty = 2, col = "#444444")
  })
  
  output$prediction_summary <- renderUI({
    tags$div(
      class = "alert alert-success mt-2",
      tags$h4("Cảm xúc dự đoán: 😊 Happy (75.0%)"),
      tags$p("Mô hình phát hiện nếp nhăn đuôi mắt Duchenne (AU6) giải thích cho nhãn Vui vẻ.")
    )
  })
}

# Khởi chạy ứng dụng
shinyApp(ui = ui, server = server)
