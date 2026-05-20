import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# --- 1. CẤU HÌNH GIAO DIỆN APP ---
st.set_page_config(
    page_title="AI GPA Predictor Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÙY CHỈNH CSS ĐỂ GIAO DIỆN ĐẸP HƠN & TƯƠNG THÍCH DARK MODE ---
st.markdown("""
    <style>
    /* 1. Tiêu đề chính */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #FFFFFF !important; /* Luôn là màu trắng */
        margin-bottom: 5px;
    }
    
    /* 2. Tiêu đề phụ */
    .sub-title {
        font-size: 1.1rem;
        color: #D1D5DB !important; /* Xám nhạt hơn một chút */
        margin-bottom: 25px;
    }
    
    /* 3. ĐÃ CHỈNH SỬA: Ô tiêu đề khối (Học thuật, Đời sống) */
    /* Cố định màu để không bị Streamlit thay đổi khi chuyển Dark/Light */
    .section-card {
        background-color: #374151 !important; /* Màu xám đậm (dark gray) */
        color: #FFFFFF !important;           /* Màu chữ trắng */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); /* Thêm đổ bóng nhẹ */
        border: 1px solid #4B5563;           /* Viền mờ */
    }
    
    /* 4. Định dạng text trong section card */
    .section-card b {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* 5. Icon trong slider/pills cho nổi bật hơn */
    div.stSlider label, div.stPills label {
        color: #E5E7EB !important;
    }
    
    /* 6. Margin cho pills/input cho rộng rãi */
    div[data-testid="stMarkdownContainer"] + div[data-testid="stHorizontalBlock"] {
        margin-top: -15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH ---
@st.cache_resource
def load_and_train_model():
    file_path = 'du_lieu_gpa_sinh_vien.csv'

    # Tự tạo dữ liệu nếu không tìm thấy file CSV
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data_source = "Dữ liệu thực tế (CSV)"
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'So_Gio_Hoc_Tuan': np.random.normal(15, 8, n).clip(0, 50),
            'So_Mon_Dang_Hoc': np.random.randint(3, 9, n),
            'Part_Time_Job': np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
            'Thoi_Gian_Ngu_Ngay': np.random.normal(7, 1.2, n).clip(4, 10),
            'Tham_Gia_CLB': np.random.choice(['Yes', 'No'], n, p=[0.5, 0.5]),
            'Attendance_Phan_Tram': np.random.normal(85, 10, n).clip(40, 100),
            'Hoc_Nhom_Hay_Tu_Hoc': np.random.choice(['Học nhóm', 'Tự học'], n),
            'Social_Media_Time_Ngay': np.random.normal(3, 2, n).clip(0, 8),
        })
        
        # Thiết lập công thức giả lập GPA có logic (Thang 4.0)
        gpa_simulated = (
            1.5
            + (df['So_Gio_Hoc_Tuan'] * 0.02)
            + (df['Attendance_Phan_Tram'] * 0.015)
            - (df['Social_Media_Time_Ngay'] * 0.08)
            - (df['Part_Time_Job'].map({'Yes': 0.1, 'No': 0}))
            + np.random.normal(0, 0.2, n)
        )
        df['GPA'] = gpa_simulated.clip(1.0, 4.0)
        data_source = "Dữ liệu Mô phỏng (Mock Data tự động)"

    df_clean = df.copy().dropna()

    # Chuyển đổi nhãn dán (Label Encoding)
    mapping_dict = {'Yes': 1, 'No': 0}
    if 'Part_Time_Job' in df_clean.columns:
        df_clean['Part_Time_Job'] = df_clean['Part_Time_Job'].map(mapping_dict)
    if 'Tham_Gia_CLB' in df_clean.columns:
        df_clean['Tham_Gia_CLB'] = df_clean['Tham_Gia_CLB'].map(mapping_dict)
    if 'Hoc_Nhom_Hay_Tu_Hoc' in df_clean.columns:
        df_clean['Hoc_Nhom_Hay_Tu_Hoc'] = df_clean['Hoc_Nhom_Hay_Tu_Hoc'].map({'Học nhóm': 1, 'Tự học': 0})

    X = df_clean.drop('GPA', axis=1)
    y = df_clean['GPA']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Đánh giá tầm quan trọng của Features
    feature_names = {
        'So_Gio_Hoc_Tuan': 'Số giờ tự học / tuần',
        'So_Mon_Dang_Hoc': 'Khối lượng môn học',
        'Part_Time_Job': 'Công việc làm thêm',
        'Thoi_Gian_Ngu_Ngay': 'Thời lượng giấc ngủ',
        'Tham_Gia_CLB': 'Hoạt động ngoại khóa (CLB)',
        'Attendance_Phan_Tram': 'Tỷ lệ chuyên cần (%)',
        'Hoc_Nhom_Hay_Tu_Hoc': 'Khuynh hướng Học nhóm/Tự học',
        'Social_Media_Time_Ngay': 'Thời gian dùng MXH'
    }

    display_names = [feature_names.get(c, c) for c in X.columns]
    importance_df = pd.DataFrame({
        'Biến số (Yếu tố)': display_names,
        'Mức độ đóng góp (%)': model.feature_importances_ * 100
    }).sort_values(by='Mức độ đóng góp (%)', ascending=True)

    return model, importance_df, len(df_clean), data_source

# Khởi chạy hàm lấy model
ai_model, importance_df, data_size, source_type = load_and_train_model()

# --- 3. SIDEBAR (THANH ĐIỀU HƯỚNG BÊN TRÁI) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ HỆ THỐNG AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3112/3112946.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📊 Thông tin dữ liệu")
    
    # Sử dụng card nhỏ để hiển thị thông tin sạch sẽ hơn
    st.info(f"📁 **Nguồn:**\n{source_type}")
    st.success(f"🤖 **Trạng thái:**\nĐã huấn luyện trên **{data_size}** mẫu khảo sát.")
    
    st.markdown("---")
    st.caption("⚡ Thiết kế tối ưu cho Nghiên cứu khoa học sinh viên © 2026")

# --- 4. GIAO DIỆN CHÍNH (MAIN WORKSPACE) ---
st.markdown("<div class='main-title'>🎯 Trợ Lý AI: Dự Đoán & Tối Ưu GPA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Cá nhân hóa lộ trình học tập của bạn bằng thuật toán <b>Random Forest Regressor</b>. Nhập thói quen hiện tại để nhận chẩn đoán từ AI.</div>", unsafe_allow_html=True)

# Chia thành các Tabs
tab_predict, tab_analytics, tab_research = st.tabs([
    "🔮 Bài Test Khảo Sát",
    "📈 Báo Cáo Yếu Tố Ảnh Hưởng",
    "💡 Góc Nhìn Học Thuật & NCKH"
])

# --- TAB 1: DỰ ĐOÁN ĐIỂM SỐ ---
with tab_predict:
    st.markdown("### 📝 Bảng Khảo Sát Hành Vi Thường Nhật")
    st.write("Vui lòng kéo các thanh trượt và chọn câu trả lời đúng nhất với tình trạng thực tế của ông bạn:")
    
    # Chia cụm input thành 2 cột chính cho gọn gàng
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-card'><b>📚 Khối Học Thuật & Trên Lớp</b></div>", unsafe_allow_html=True)
        gio_hoc = st.slider("⏱️ Số giờ tự học ngoài giờ lên lớp (Giờ/Tuần):", 0, 50, 15, help="Tổng thời gian bạn tự học, làm bài tập ở nhà hoặc thư viện.")
        attendance = st.slider("💯 Tỷ lệ chuyên cần trên lớp (%):", 0, 100, 90, help="Điểm danh đầy đủ hay cúp học nhiều?")
        so_mon = st.number_input("📖 Số môn học đăng ký trong học kỳ này:", 1, 15, 5)
        hoc_nhom_input = st.pills("👥 Hình thức học tập yêu thích của bạn:", ["Tự học", "Học nhóm"], default="Tự học")

    with col2:
        st.markdown("<div class='section-card'><b>🍕 Khối Đời Sống & Sinh Hoạt</b></div>", unsafe_allow_html=True)
        gio_ngu = st.slider("😴 Thời gian ngủ trung bình một đêm (Giờ):", 3.0, 12.0, 7.0, step=0.5)
        social_media = st.slider("📱 Thời gian lướt MXH, xem phim, giải trí (Giờ/Ngày):", 0.0, 12.0, 3.0, step=0.5)
        part_time_input = st.pills("💼 Bạn có đang đi làm thêm không?", ["Không", "Có"], default="Không")
        clb_input = st.pills("⚽ Bạn có tham gia CLB / Đội nhóm nào không?", ["Không", "Có"], default="Không")

    # Mapping dữ liệu để đưa vào model
    part_time = 1 if part_time_input == "Có" else 0
    clb = 1 if clb_input == "Có" else 0
    hoc_nhom = 1 if hoc_nhom_input == "Học nhóm" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    # Nút bấm tính toán thiết kế to, rõ ràng
    if st.button("🚀 CHẠY MÔ HÌNH & CHUẨN ĐOÁN KẾT QUẢ", type="primary", use_container_width=True):
        input_array = np.array([[gio_hoc, so_mon, part_time, gio_ngu, clb, attendance, hoc_nhom, social_media]])
        predicted_gpa = ai_model.predict(input_array)[0]

        st.markdown("---")
        st.markdown("### 🌟 Kết Quả Đánh Giá Từ Hệ Thống AI")
        
        # Tạo Layout hiển thị kết quả bắt mắt
        res_col1, res_col2 = st.columns([1, 2], gap="medium")
        
        with res_col1:
            st.metric(label="📊 ĐIỂM GPA DỰ ĐOÁN (Thang 4.0)", value=f"{predicted_gpa:.2f} / 4.0")
            
        with res_col2:
            if predicted_gpa >= 3.6:
                st.success("🌟 **Cực kỳ xuất sắc!** Thói quen sinh hoạt và học tập của ông bạn đang đạt chuẩn của nhóm sinh viên săn học bổng. Tiếp tục phát huy phong độ này nhé!")
            elif predicted_gpa >= 3.2:
                st.info("👍 **Xếp loại Khá - Giỏi!** Ông bạn quản lý thời gian khá tốt đấy. Thử tăng nhẹ thời gian tự học thêm 2-3 tiếng/tuần để bứt phá hẳn lên Xuất sắc xem sao.")
            elif predicted_gpa >= 2.5:
                st.warning("⚠️ **Xếp loại Trung bình - Khá.** Mức điểm này nằm ở vùng an toàn nhưng rất dễ bị tụt dốc. Hãy để ý cắt giảm bớt thời gian lướt TikTok/Facebook và tập trung hơn nhé.")
            else:
                st.error("🚨 **Báo động đỏ!** Phong cách học tập này cực kỳ nguy hiểm, nguy cơ rớt môn hoặc cảnh cáo học vụ rất cao. Cần siết chặt lại kỷ luật bản thân, đi học đầy đủ và tăng giờ tự học ngay!")

# --- TAB 2: BIỂU ĐỒ TRỌNG SỐ ---
with tab_analytics:
    st.markdown("### 📊 Trọng Số Tác Động Của Các Thói Quen Lên GPA")
    st.write("Biểu đồ này bóc tách thuật toán bên trong của mô hình Random Forest, cho thấy yếu tố nào thực sự quyết định đến điểm số của sinh viên.")

    # Biểu đồ Plotly sang xịn mịn hơn
    fig = px.bar(
        importance_df,
        x='Mức độ đóng góp (%)',
        y='Biến số (Yếu tố)',
        orientation='h',
        text_auto='.1f',
        color='Mức độ đóng góp (%)',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    fig.update_layout(
        xaxis_title="Mức độ ảnh hưởng quan trọng (%)",
        yaxis_title="Các biến số khảo sát",
        showlegend=False,
        height=450,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Xem bảng dữ liệu chi tiết (Phục vụ trích dẫn số liệu)"):
        st.dataframe(
            importance_df.sort_values(by='Mức độ đóng góp (%)', ascending=False)
            .style.background_gradient(cmap='Blues'), 
            use_container_width=True
        )

# --- TAB 3: TƯ LIỆU NGHIÊN CỨU ---
with tab_research:
    st.markdown("### 💡 Tài Liệu Hỗ Trợ Viết Báo Cáo Khoa Học (NCKH)")
    st.write("Nếu ông bạn đang dùng ứng dụng này để làm phôi đề tài nghiên cứu, đây là các đoạn văn mẫu học thuật hỗ trợ viết phần Thảo luận (Discussion) & Phương pháp (Methodology):")

    st.markdown("""
    > 📌 **Về Thuật Toán (Algorithm Justification):**  
    *Trong nghiên cứu Khai phá Dữ liệu Giáo dục (Educational Data Mining), mô hình Random Forest Regressor được lựa chọn nhờ khả năng tối ưu trong việc xử lý các mối quan hệ phi tuyến tính phức tạp (ví dụ: tác động của giấc ngủ tuân theo hàm phi tuyến – ngủ quá ít hay quá nhiều đều kéo giảm kết quả học tập). Thuật toán này hạn chế tối đa hiện tượng quá khớp (overfitting) thông qua cơ chế phân tách ngẫu nhiên các cây quyết định.*

    > 📌 **Hạn Chế Về Sai Số Hồi Tưởng (Limitations & Recall Bias):**  
    *Dữ liệu thu thập thông qua khảo sát dạng tự khai báo (Self-reporting) khó tránh khỏi Định kiến mong muốn xã hội (Social Desirability Bias). Sinh viên có xu hướng khai tăng số giờ tự học thực tế và ước lượng thấp hơn thời gian lãng phí cho các nền tảng mạng xã hội do thiếu công cụ theo dõi thời gian thực (Screen-time logs).*

    > 📌 **Hàm Ý Quản Trị & Khuyến Nghị (Practical Implications):**  
    *Dựa vào phân tích Trọng số tác động (Feature Importance), hai yếu tố **Tỷ lệ chuyên cần** và **Số giờ tự học** chiếm tỷ trọng chi phối cao nhất. Do đó, thay vì áp đặt các biện pháp cực đoan như cấm đoán sinh viên làm thêm (Part-time job), các nhà quản lý giáo dục nên tập trung vào việc số hóa quy trình điểm danh và tối ưu không gian tự học tại khuôn viên trường học.*
    """)