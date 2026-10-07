import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_paragraph(p, space_before=2, space_after=4, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_callout(doc, text, bold_prefix="💡 Điểm cốt lõi: "):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    borders_xml = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="none"/>
        <w:left w:val="single" w:sz="24" w:space="0" w:color="1D4ED8"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
    </w:tcBorders>
    '''
    cell._tc.get_or_add_tcPr().append(parse_xml(borders_xml))
    
    p = cell.paragraphs[0]
    style_paragraph(p, space_before=2, space_after=2)
    run_bold = p.add_run(bold_prefix)
    run_bold.bold = True
    run_bold.font.color.rgb = RGBColor(29, 78, 216)
    run_bold.font.size = Pt(10.5)
    
    run_text = p.add_run(text)
    run_text.font.size = Pt(10.5)
    run_text.font.color.rgb = RGBColor(30, 41, 59)
    run_text.italic = True
    
    p_after = doc.add_paragraph()
    style_paragraph(p_after, space_before=0, space_after=4)

def build_docx(output_path):
    doc = docx.Document()
    
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.85)
        sec.right_margin = Inches(0.85)
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)

    # Title
    p_title = doc.add_paragraph()
    style_paragraph(p_title, space_before=6, space_after=4)
    run_title = p_title.add_run("KẾ HOẠCH DỰ ÁN & PHƯƠNG PHÁP NGHIÊN CỨU\n(PROJECT METHODOLOGY & IMPLEMENTATION PLAN)")
    run_title.bold = True
    run_title.font.size = Pt(18)
    run_title.font.color.rgb = RGBColor(30, 58, 138)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_sub = doc.add_paragraph()
    style_paragraph(p_sub, space_before=0, space_after=12)
    run_sub = p_sub.add_run("Real-Time Multi-Face Explainable Facial Expression Recognition on In-The-Wild Faces & Real Masked Occlusion via Pre-Trained Deep Models")
    run_sub.italic = True
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Project Overview Info Box
    tbl_info = doc.add_table(rows=1, cols=1)
    tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_info = tbl_info.cell(0, 0)
    set_cell_shading(c_info, "F8FAFC")
    set_cell_margins(c_info, top=120, bottom=120, left=180, right=180)
    
    borders_info = f'''
    <w:tcBorders {nsdecls("w")}>
        <w:top w:val="single" w:sz="12" w:space="0" w:color="CBD5E1"/>
        <w:left w:val="single" w:sz="12" w:space="0" w:color="CBD5E1"/>
        <w:bottom w:val="single" w:sz="12" w:space="0" w:color="CBD5E1"/>
        <w:right w:val="single" w:sz="12" w:space="0" w:color="CBD5E1"/>
    </w:tcBorders>
    '''
    c_info._tc.get_or_add_tcPr().append(parse_xml(borders_info))
    
    p_box = c_info.paragraphs[0]
    style_paragraph(p_box, space_before=2, space_after=2)
    p_box.add_run("📌 QUY MÔ NHÓM: ").bold = True
    p_box.add_run("3 thành viên   |   ")
    p_box.add_run("THỜI LƯỢNG DỰ KIẾN: ").bold = True
    p_box.add_run("6 tuần   |   ")
    p_box.add_run("NỀN TẢNG: ").bold = True
    p_box.add_run("PyTorch (CUDA), MediaPipe, Streamlit\n")
    p_box.add_run("🌍 NGUỒN DỮ LIỆU THẬT 100%: ").bold = True
    p_box.add_run("RAF-DB (15,339 ảnh người thật in-the-wild) và RMFD (ảnh người thật đeo khẩu trang thật ngoài đời).\n")
    p_box.add_run("🎯 ĐỊNH HƯỚNG MÔ HÌNH: ").bold = True
    p_box.add_run("Tận dụng mô hình Pre-trained SOTA (MobileNetV3-Large, ResNet-18) với 2 chiến lược Feature Extraction & Deep Fine-Tuning, kết hợp bộ phát hiện MediaPipe BlazeFace quét nhiều khuôn mặt thời gian thực và trực quan hóa Grad-CAM.")

    # Section 1
    h1 = doc.add_paragraph()
    style_paragraph(h1, space_before=14, space_after=4)
    r1 = h1.add_run("1. ĐẶT VẤN ĐỀ VÀ CHIẾN LƯỢC BÁO CÁO (PITCHING STRATEGY)")
    r1.bold = True
    r1.font.size = Pt(14)
    r1.font.color.rgb = RGBColor(30, 58, 138)

    p = doc.add_paragraph()
    style_paragraph(p)
    p.add_run("1.1. Thách thức thực tiễn (Problem Statement):\n").bold = True
    p.add_run("• Vấn đề mô hình Black-Box: Các hệ thống FER truyền thống chỉ đưa ra nhãn cảm xúc mà thiếu cơ chế minh bạch trực quan, khó thuyết phục người dùng.\n")
    p.add_run("• Thách thức môi trường thực tế (In-The-Wild & Che khuất): Khuôn mặt người thật ngoài đời chịu ảnh hưởng của ánh sáng phức tạp, góc nghiêng, và các che khuất tự nhiên (kính, tay che mặt, khẩu trang y tế/vải). Mô hình học từ đầu trên tập ảnh nhỏ nhân tạo rất dễ sụp đổ khi ra môi trường thật.\n")
    p.add_run("• Quét nhiều khuôn mặt thời gian thực (Real-Time Multi-Face): Camera giám sát và ứng dụng thực tế luôn phải theo dõi đồng thời nhiều người trong khung hình, đòi hỏi mô hình vừa chính xác vừa có tốc độ suy luận cực cao (High FPS).")

    p = doc.add_paragraph()
    style_paragraph(p)
    p.add_run("1.2. Giải pháp kỹ thuật cốt lõi (Proposed Solution):\n").bold = True
    p.add_run("• Sử dụng dữ liệu thật 100% (RAF-DB & RMFD): Huấn luyện và đánh giá trên bộ dữ liệu người thật chụp ngoài đời thực RAF-DB (15,339 ảnh) và kiểm thử che khuất trên tập người thật đeo khẩu trang thật RMFD.\n")
    p.add_run("• Khai thác Pre-trained Backbones & Transfer Learning: Sử dụng MobileNetV3-Large (~21 MB, siêu nhanh) và ResNet-18 đã được tiền huấn luyện trên hàng triệu ảnh. Thực nghiệm cả 2 chiến lược: Feature Extraction (đóng băng) và Deep Fine-Tuning kết hợp Weighted Loss.\n")
    p.add_run("• Quét đa khuôn mặt thời gian thực: Tích hợp MediaPipe Face Detection phát hiện song song mọi khuôn mặt trong video stream và phân loại cảm xúc đồng thời.\n")
    p.add_run("• Minh bạch hóa với Explainable AI (Grad-CAM): Trực quan hóa vùng chú ý của AI trên khuôn mặt người thật, click chọn từng người để xem bản đồ nhiệt.")

    add_callout(
        doc,
        '"Instead of relying on toy datasets or artificial masks, our project builds a robust, real-time multi-face emotion recognition system grounded in 100% real-world in-the-wild human data (RAF-DB & RMFD). By leveraging lightweight pre-trained architectures (MobileNetV3) with targeted fine-tuning and integrating MediaPipe BlazeFace detection alongside interactive Grad-CAM explainability, we deliver transparent, high-speed multi-person emotion analytics ready for real-world deployment."',
        bold_prefix="🎯 Thông điệp thuyết trình bảo vệ đề tài (Elevator Pitch): "
    )

    # Section 2
    h2 = doc.add_paragraph()
    style_paragraph(h2, space_before=14, space_after=4)
    r2 = h2.add_run("2. KIẾN TRÚC HỆ THỐNG VÀ CÁC GIAI ĐOẠN THỰC HIỆN")
    r2.bold = True
    r2.font.size = Pt(14)
    r2.font.color.rgb = RGBColor(30, 58, 138)

    phases_data = [
        ("Phase 1: Real-World Data Pipeline (RAF-DB)", "Nạp và chuẩn hóa 15,339 ảnh người thật in-the-wild từ RAF-DB (12,271 train / 3,068 test) và tập ảnh che khuất thật RMFD. Tính toán phân bố nhãn và thiết lập Data Loader RGB 224x224."),
        ("Phase 2: Multi-Face Detection Module", "Tích hợp MediaPipe Face Detector (BlazeFace) quét đồng thời tất cả khuôn mặt trong luồng video/webcam với độ trễ cực thấp (< 15 ms/khung hình)."),
        ("Phase 3: Pre-trained Backbone Setup", "Thiết lập MobileNetV3-Large (tối ưu tốc độ, nhẹ ~21 MB) và ResNet-18 từ torchvision weights, thay thế Classifier Head 7 lớp cảm xúc."),
        ("Phase 4: Two-Stage Transfer Learning", "Giai đoạn 1: Warmup Classifier Head (đóng băng backbone). Giai đoạn 2: Deep Fine-Tuning với learning rate nhỏ kết hợp Weighted Cross-Entropy Loss xử lý mất cân bằng nhãn."),
        ("Phase 5: Đánh giá Đa chiều trên Dữ liệu Thật", "Đo lường Accuracy, Macro F1, Recall từng lớp trên tập kiểm thử người thật RAF-DB test. Đánh giá độ bền vững trên tập ảnh người thật đeo khẩu trang RMFD."),
        ("Phase 6: Explainable AI (Grad-CAM Inspector)", "Trích xuất bản đồ nhiệt Grad-CAM từ Conv layer cuối cùng, giải thích rõ AI tập trung vào vùng mắt, mũi hay miệng trên khuôn mặt người thật."),
        ("Phase 7: Real-Time Multi-Face Web Application", "Xây dựng Dashboard Streamlit: Quét webcam nhiều người trực tiếp, vẽ Bounding Boxes + Emoji + Tên cảm xúc + % Tin cậy, kèm tương tác click chọn khuôn mặt xem Grad-CAM.")
    ]

    tbl_phases = doc.add_table(rows=len(phases_data) + 1, cols=2)
    tbl_phases.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Giai đoạn (Phase)", "Nội dung kỹ thuật chi tiết"]
    for i, h in enumerate(headers):
        cell = tbl_phases.cell(0, i)
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10.5)

    col_widths = [Inches(2.5), Inches(4.3)]
    for r_idx, (p_title_text, p_desc) in enumerate(phases_data):
        row_cells = tbl_phases.rows[r_idx + 1].cells
        c1 = row_cells[0]
        c2 = row_cells[1]
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        set_cell_shading(c1, bg)
        set_cell_shading(c2, bg)
        set_cell_margins(c1, top=100, bottom=100, left=140, right=140)
        set_cell_margins(c2, top=100, bottom=100, left=140, right=140)
        
        p1 = c1.paragraphs[0]
        r1_txt = p1.add_run(p_title_text)
        r1_txt.bold = True
        r1_txt.font.size = Pt(10)
        
        p2 = c2.paragraphs[0]
        r2_txt = p2.add_run(p_desc)
        r2_txt.font.size = Pt(10)

    for row in tbl_phases.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

    # Section 3: Model Comparison Table
    h3 = doc.add_paragraph()
    style_paragraph(h3, space_before=16, space_after=4)
    r3 = h3.add_run("3. BẢNG SO SÁNH PHƯƠNG PHÁP & KẾT QUẢ KỲ VỌNG TRÊN DỮ LIỆU THẬT")
    r3.bold = True
    r3.font.size = Pt(14)
    r3.font.color.rgb = RGBColor(30, 58, 138)

    comp_headers = ["Phương pháp / Mô hình", "Đặc điểm huấn luyện", "Acc (RAF-DB Test)", "Macro F1", "Tốc độ (FPS)", "Độ phù hợp Multi-Face"]
    comp_rows = [
        ("Baseline CNN (Scratch)", "Huấn luyện từ đầu trên ảnh pixelated", "~62%", "~0.56", ">60 FPS", "Kém (dễ báo sai khi ra ngoài đời)"),
        ("MobileNetV3 (Feature Extract)", "Đóng băng backbone, chỉ train head", "~72%", "~0.68", "~50 FPS", "Tốt (hội tụ nhanh, nhẹ máy)"),
        ("MobileNetV3 (Fine-Tuning ⭐)", "Fine-tune sâu trên RAF-DB + Weighted Loss", "78 - 83%", "0.75 - 0.80", "~45 FPS", "Xuất sắc (chuẩn SOTA, chạy mượt real-time)")
    ]

    tbl_comp = doc.add_table(rows=len(comp_rows) + 1, cols=6)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    for i, h in enumerate(comp_headers):
        cell = tbl_comp.cell(0, i)
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9.5)

    c_widths_comp = [Inches(1.8), Inches(1.8), Inches(1.0), Inches(0.8), Inches(0.7), Inches(1.4)]
    for r_idx, row_data in enumerate(comp_rows):
        row_cells = tbl_comp.rows[r_idx + 1].cells
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            cell = row_cells[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if c_idx == 0:
                run.bold = True

    for row in tbl_comp.rows:
        for idx, width in enumerate(c_widths_comp):
            row.cells[idx].width = width

    # Section 4: Team Responsibilities
    h4 = doc.add_paragraph()
    style_paragraph(h4, space_before=16, space_after=4)
    r4 = h4.add_run("4. PHÂN CHIA TRÁCH NHIỆM TRONG NHÓM (3 THÀNH VIÊN)")
    r4.bold = True
    r4.font.size = Pt(14)
    r4.font.color.rgb = RGBColor(30, 58, 138)

    team_data = [
        ("Thành viên 1\n(Data & Detection Pipeline)", 
         "• Phân tích EDA trên bộ dữ liệu người thật RAF-DB và dữ liệu che khuất RMFD.\n"
         "• Xây dựng Custom Dataset Loader nạp ảnh RGB chuẩn ImageNet.\n"
         "• Tích hợp bộ phát hiện đa khuôn mặt MediaPipe Face Detector (BlazeFace).\n"
         "• Tính toán phân bố nhãn và trọng số Class Weights.",
         "• Module dữ liệu (dataset.py)\n"
         "• Module phát hiện đa khuôn mặt (face_detector.py)\n"
         "• Báo cáo EDA phân tích chất lượng ảnh người thật in-the-wild"),
         
        ("Thành viên 2\n(Deep Learning & Fine-Tuning)", 
         "• Thiết lập kiến trúc Pre-trained MobileNetV3-Large và ResNet-18.\n"
         "• Xây dựng pipeline Two-Stage Transfer Learning (Warmup Head -> Deep Fine-Tuning).\n"
         "• Cài đặt Weighted Cross-Entropy Loss xử lý mất cân bằng lớp.\n"
         "• Đánh giá định lượng trên RAF-DB test (Accuracy, Macro F1, Confusion Matrix).",
         "• Module mô hình (models.py)\n"
         "• Script huấn luyện & fine-tuning (train.py)\n"
         "• Checkpoint tối ưu (best_model_rafdb.pth)\n"
         "• Báo cáo so sánh thực nghiệm"),
         
        ("Thành viên 3\n(XAI & Real-Time Deployment)", 
         "• Triển khai thuật toán Grad-CAM tương thích với MobileNetV3/ResNet.\n"
         "• Trực quan hóa bản đồ nhiệt trên khuôn mặt người thật và mặt đeo khẩu trang.\n"
         "• Phát triển ứng dụng Web Streamlit: Quét webcam đa khuôn mặt thời gian thực.\n"
         "• Xây dựng tính năng click chọn khuôn mặt xem Grad-CAM tương tác.\n"
         "• Soạn thảo slide thuyết trình, video demo và kịch bản bảo vệ.",
         "• Module giải thích XAI (gradcam.py)\n"
         "• Ứng dụng Web hoàn chỉnh (app.py)\n"
         "• Dashboard webcam đa khuôn mặt\n"
         "• Slide báo cáo và kịch bản demo")
    ]

    tbl_team = doc.add_table(rows=len(team_data) + 1, cols=3)
    tbl_team.alignment = WD_TABLE_ALIGNMENT.CENTER

    team_headers = ["Thành viên & Vai trò", "Trách nhiệm chuyên môn chi tiết", "Sản phẩm bàn giao (Deliverables)"]
    for i, h in enumerate(team_headers):
        cell = tbl_team.cell(0, i)
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10)

    team_widths = [Inches(1.8), Inches(3.0), Inches(2.1)]
    for r_idx, (m_role, m_resp, m_deliv) in enumerate(team_data):
        row_cells = tbl_team.rows[r_idx + 1].cells
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate([m_role, m_resp, m_deliv]):
            cell = row_cells[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if c_idx == 0:
                run.bold = True

    for row in tbl_team.rows:
        for idx, width in enumerate(team_widths):
            row.cells[idx].width = width

    # Section 5: Roadmap Table
    h5 = doc.add_paragraph()
    style_paragraph(h5, space_before=16, space_after=4)
    r5 = h5.add_run("5. LỘ TRÌNH THỰC HIỆN DỰ ÁN (6 TUẦN)")
    r5.bold = True
    r5.font.size = Pt(14)
    r5.font.color.rgb = RGBColor(30, 58, 138)

    roadmap_data = [
        ("Tuần 1", "Khám phá Dữ liệu Thật & Đề cương", "Khám phá phân bố 15,339 ảnh RAF-DB, phân tích mất cân bằng nhãn, hoàn thiện tài liệu Proposal và bảo vệ đề cương."),
        ("Tuần 2", "Pre-trained Backbone & Face Detection", "Tích hợp MediaPipe BlazeFace phát hiện đa khuôn mặt. Thiết lập MobileNetV3-Large và ResNet-18 với custom head."),
        ("Tuần 3", "Two-Stage Transfer Learning Pipeline", "Thực hiện Warmup Classifier Head (đóng băng backbone) và Deep Fine-Tuning với Weighted Loss trên tập train RAF-DB."),
        ("Tuần 4", "Đánh giá Toàn diện & Phân tích Nhầm lẫn", "Đo lường Accuracy, Macro F1, Recall từng lớp trên RAF-DB test. Kiểm thử độ bền vững trên tập ảnh che khuất thật RMFD."),
        ("Tuần 5", "Explainable AI (Grad-CAM Inspector)", "Trích xuất Grad-CAM từ mô hình Fine-tuned, trực quan hóa vùng mắt, mũi, miệng kích hoạt trên ảnh người thật."),
        ("Tuần 6", "Streamlit Multi-Face Web App & Nghiệm thu", "Đóng gói ứng dụng Streamlit quét nhiều khuôn mặt thời gian thực trên webcam/video, quay video demo và hoàn tất báo cáo.")
    ]

    tbl_road = doc.add_table(rows=len(roadmap_data) + 1, cols=3)
    tbl_road.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    road_headers = ["Tuần", "Mục tiêu trọng tâm", "Công việc thực hiện chi tiết"]
    for i, h in enumerate(road_headers):
        cell = tbl_road.cell(0, i)
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(10)

    road_widths = [Inches(1.0), Inches(2.5), Inches(3.4)]
    for r_idx, (w_num, w_goal, w_desc) in enumerate(roadmap_data):
        row_cells = tbl_road.rows[r_idx + 1].cells
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate([w_num, w_goal, w_desc]):
            cell = row_cells[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if c_idx < 2:
                run.bold = True

    for row in tbl_road.rows:
        for idx, width in enumerate(road_widths):
            row.cells[idx].width = width

    # Section 6: Acceptance Criteria
    h6 = doc.add_paragraph()
    style_paragraph(h6, space_before=16, space_after=4)
    r6 = h6.add_run("6. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG VÀ KẾT QUẢ KỲ VỌNG")
    r6.bold = True
    r6.font.size = Pt(14)
    r6.font.color.rgb = RGBColor(30, 58, 138)

    p_acc = doc.add_paragraph()
    style_paragraph(p_acc)
    p_acc.add_run("1. Hiệu năng định lượng trên dữ liệu người thật (Quantitative Metrics):\n").bold = True
    p_acc.add_run("• Mô hình MobileNetV3 Fine-Tuned đạt Accuracy từ 78 - 83% trên tập kiểm thử người thật RAF-DB test.\n")
    p_acc.add_run("• Macro F1-Score đạt >= 0.75 nhờ tối ưu hóa Weighted Loss, cải thiện rõ rệt trên các cảm xúc khó (Fear, Disgust).\n")
    p_acc.add_run("2. Độ tin cậy và giải thích khoa học (Qualitative & XAI Insights):\n").bold = True
    p_acc.add_run("• Grad-CAM Heatmap phản ánh chính xác các vùng giải phẫu cơ mặt biểu cảm sinh học trên người thật (nụ cười Duchenne, cơ cau mày, miệng há ngạc nhiên).\n")
    p_acc.add_run("3. Ứng dụng thực tế quét nhiều người (Real-Time Multi-Face UX):\n").bold = True
    p_acc.add_run("• Hệ thống phát hiện và nhận diện đồng thời từ 2 - 5 khuôn mặt trong luồng video/webcam với tốc độ mượt mà (đạt 25 - 40+ FPS).\n")
    p_acc.add_run("• Giao diện Web trực quan, vẽ bounding box và cho phép người dùng click chọn từng khuôn mặt để mở kính lúp Grad-CAM.")

    doc.save(output_path)
    print(f"Successfully generated docx at: {output_path}")

if __name__ == '__main__':
    target = r'e:\dev\DSR301m\docs\Explainable_Facial_Expression_Recognition_Project_Plan.docx'
    build_docx(target)
