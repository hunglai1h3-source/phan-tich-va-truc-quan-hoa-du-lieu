import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore') # Tắt các cảnh báo không quan trọng

# 0. CHUẨN BỊ MÔI TRƯỜNG
# Tạo thư mục lưu biểu đồ nếu chưa có
os.makedirs("charts", exist_ok=True)

# Đọc dữ liệu
try:
    df = pd.read_csv("learnx.csv")
    print(" Đã tải dữ liệu thành công!\n")
except FileNotFoundError:
    print(" LỖI: Không tìm thấy file 'learnx.csv'. Vui lòng để file csv cùng thư mục với file code.")
    exit()

# ---------------------------------------------------------
# 1. GIỚI THIỆU VÀ MÔ TẢ DỮ LIỆU
# ---------------------------------------------------------
print("--- 1. TỔNG QUAN DỮ LIỆU ---")
print(f"Số lượng bản ghi (dòng): {df.shape[0]}")
print(f"Số lượng thuộc tính (cột): {df.shape[1]}")
print("Các thuộc tính chính:", df.columns.tolist())

# ---------------------------------------------------------
# 2. LÀM SẠCH DỮ LIỆU
# ---------------------------------------------------------
print("\n--- 2. LÀM SẠCH DỮ LIỆU ---")
# Lưu lại số dòng ban đầu để so sánh
initial_rows = len(df)

# Xóa dữ liệu trùng lặp
df = df.drop_duplicates()
print(f"Đã xóa {initial_rows - len(df)} dòng trùng lặp.")

# Xử lý giá trị thiếu (Missing values)
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median()) # Lấp đầy số bằng trung vị
    else:
        df[col] = df[col].fillna(df[col].mode()[0]) # Lấp đầy chữ bằng giá trị xuất hiện nhiều nhất
print("Đã xử lý xong các giá trị thiếu (missing values).")

# ---------------------------------------------------------
# 3. PHÂN TÍCH BẰNG BIỂU ĐỒ & 4. PHÁT HIỆN BẤT THƯỜNG
# ---------------------------------------------------------
# Định nghĩa các cột quan trọng (Dựa theo dữ liệu giả định của LearnX)
study_col = "avg_session_minutes"
visit_col = "sessions_per_week"
completion_col = "completion_rate"
course_col = "courses_enrolled"
spend_col = "total_spent_usd"

# Danh sách các cột số để vẽ Boxplot
numeric_cols = [study_col, visit_col, completion_col, course_col, spend_col]
# Lọc ra những cột thực sự tồn tại trong file csv
valid_cols = [c for c in numeric_cols if c in df.columns]

# --- VẼ HISTOGRAM (PHÂN PHỐI) ---
print("\n--- 3. ĐANG VẼ BIỂU ĐỒ ---")
chart_titles = {
    study_col: "Phan_phoi_thoi_gian_hoc",
    visit_col: "So_lan_truy_cap_moi_tuan",
    completion_col: "Muc_do_hoan_thanh_khoa_hoc"
}

for col, title in chart_titles.items():
    if col in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col], bins=20, kde=True, color="cornflowerblue")
        plt.title(title.replace("_", " "))
        plt.xlabel(col)
        plt.ylabel("Số lượng người dùng")
        plt.savefig(f"charts/{title}.png")
        plt.close() # Đóng biểu đồ để giải phóng bộ nhớ

# --- VẼ BOXPLOT (PHÁT HIỆN OUTLIERS) CHUẨN XÁC ---
if valid_cols:
    plt.figure(figsize=(15, 8))
    # Tạo các đồ thị con (subplots) để vẽ riêng từng cột
    for i, col in enumerate(valid_cols, 1):
        plt.subplot(2, 3, i) # Lưới 2 hàng, 3 cột
        sns.boxplot(y=df[col], color="lightgreen")
        plt.title(f"Outliers: {col}")
    
    plt.tight_layout()
    plt.savefig("charts/Boxplot_tong_hop.png")
    plt.close()

# --- TÍNH TOÁN BẤT THƯỜNG & OUTLIERS ---
outlier_result = pd.DataFrame()
for col in valid_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Lọc ra các dòng là outliers
    temp = df[(df[col] < lower) | (df[col] > upper)].copy()
    if not temp.empty:
        temp["outlier_column"] = col
        outlier_result = pd.concat([outlier_result, temp], ignore_index=True)

print("\n--- 4. HÀNH VI BẤT THƯỜNG ---")
print(f"Tổng số điểm dữ liệu ngoại lai (Outliers): {len(outlier_result)}")
if len(outlier_result) > 0:
    outlier_result.to_csv("outliers_learnx.csv", index=False)

# Tính toán các nhóm đặc biệt theo yêu cầu đề bài
high_study_count = len(df[df[study_col] > df[study_col].quantile(0.95)]) if study_col in df.columns else 0
many_course_no_study_count = len(df[(df[course_col] > df[course_col].quantile(0.75)) & (df[study_col] <= df[study_col].quantile(0.25))]) if course_col in df.columns and study_col in df.columns else 0
abnormal_spend_count = len(df[df[spend_col] > df[spend_col].quantile(0.95)]) if spend_col in df.columns else 0

print(f"Người dùng học cực kỳ nhiều: {high_study_count}")
print(f"Đăng ký nhiều nhưng học ít: {many_course_no_study_count}")
print(f"Chi tiêu bất thường: {abnormal_spend_count}")

# ---------------------------------------------------------
# XUẤT BÁO CÁO DẠNG TEXT
# ---------------------------------------------------------
with open("bao_cao_giai_doan_1.txt", "w", encoding="utf-8") as f:
    f.write("BÁO CÁO GIAI ĐOẠN 1 - KHÁM PHÁ DỮ LIỆU LEARNX\n")
    f.write("="*50 + "\n\n")
    
    f.write("1. Giới thiệu dữ liệu\n")
    f.write(f"- Số lượng bản ghi: {df.shape[0]}\n")
    f.write(f"- Số lượng thuộc tính: {df.shape[1]}\n")
    f.write(f"- Các thuộc tính chính: {', '.join(df.columns)}\n\n")

    f.write("2. Làm sạch dữ liệu\n")
    f.write("- Đã kiểm tra missing values, thay thế bằng giá trị trung vị (số) và mode (chữ).\n")
    f.write(f"- Đã xóa {initial_rows - len(df)} dòng trùng lặp trong dữ liệu.\n\n")

    f.write("3. Phân tích bằng biểu đồ\n")
    f.write("- Đã vẽ biểu đồ phân phối thời gian học, số lần truy cập, mức độ hoàn thành.\n")
    f.write("- Đã vẽ cụm Boxplot tách rời để phát hiện outliers một cách chuẩn xác nhất.\n\n")

    f.write("4. Phát hiện hành vi bất thường\n")
    f.write(f"- Tổng số Outliers phát hiện: {len(outlier_result)} (Đã xuất ra file outliers_learnx.csv)\n")
    f.write(f"- Người dùng học cực kỳ nhiều: {high_study_count}\n")
    f.write(f"- Người dùng đăng ký nhiều khóa nhưng học ít: {many_course_no_study_count}\n")
    f.write(f"- Người dùng chi tiêu bất thường: {abnormal_spend_count}\n\n")

    f.write("5. Insight cho đội sản phẩm\n")
    f.write("- Cần có chiến dịch nhắc nhở (push notification) qua app/email đối với nhóm đăng ký nhiều nhưng lười học.\n")
    f.write("- Nhóm học cực kỳ nhiều là khách hàng tiềm năng, hệ thống AI cần gợi ý ngay các khóa học nâng cao (Premium) cho họ.\n")
    f.write("- Cần phân tích sâu hơn nhóm chi tiêu bất thường để xác định xem đây là lỗi hệ thống thanh toán hay là người dùng mua tài khoản cho tổ chức/nhóm.\n")

print("\n HOÀN THÀNH GIAI ĐOẠN 1!")
print(" Hãy kiểm tra thư mục 'charts' để xem biểu đồ và mở file 'bao_cao_giai_doan_1.txt' để xem kết quả.")