# =========================================
# GIAI ĐOẠN 1: KHÁM PHÁ DỮ LIỆU (EDA)
# =========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================
# 1. ĐỌC DỮ LIỆU
# =========================================
print("Đang đọc dữ liệu...")

df = pd.read_excel(
    "learnx_user_behavior_dataset_10M.xlsx"
)

print("Đọc thành công!")
print(df.shape)

print("\nDanh sách cột:")
print(df.columns.tolist())

# =========================================
# 2. GIỚI THIỆU DỮ LIỆU
# =========================================

print("\n" + "="*60)
print("THÔNG TIN TỔNG QUAN DỮ LIỆU")
print("="*60)

print("Kích thước dữ liệu:")
print(df.shape)

print("\nDanh sách cột:")
print(df.columns.tolist())

print("\nKiểu dữ liệu:")
print(df.dtypes)

print("\n5 dòng đầu tiên:")
print(df.head())

print("\nThống kê mô tả:")
print(df.describe())

# =========================================
# 3. KIỂM TRA DỮ LIỆU THIẾU
# =========================================

print("\n" + "="*60)
print("GIÁ TRỊ THIẾU")
print("="*60)

print(df.isnull().sum())

# =========================================
# 4. KIỂM TRA DỮ LIỆU TRÙNG LẶP
# =========================================

print("\n" + "="*60)
print("DỮ LIỆU TRÙNG LẶP")
print("="*60)

print("Số dòng trùng:")
print(df.duplicated().sum())

# DỌN DẸP DỮ LIỆU
df = df.drop_duplicates() 
df = df.fillna(0)        

print("\nĐã làm sạch dữ liệu thành công!")

# =========================================
# 5. TẠO MẪU ĐỂ VẼ BIỂU ĐỒ
# =========================================


print("\nLấy mẫu 1000000 dòng để trực quan hóa...")

sample_df = df.sample(
    n=min(100000, len(df)),
    random_state=42
)

# =========================================
# 6. PHÂN PHỐI THỜI GIAN HỌC
# =========================================

plt.figure(figsize=(8,5))
sns.histplot(
    sample_df["avg_session_minutes"],
    bins=30,
    kde=True
)

plt.title("Phân phối thời gian học")
plt.xlabel("Thời gian học trung bình (phút)")
plt.ylabel("Tần suất")
plt.show()

# =========================================
# 7. BOXPLOT NGOẠI LỆ
# =========================================

plt.figure(figsize=(8,5))
sns.boxplot(
    x=sample_df["avg_session_minutes"]
)

plt.title("Ngoại lệ thời gian học")
plt.show()

# =========================================
# 8. SỐ PHIÊN HỌC MỖI TUẦN
# =========================================

plt.figure(figsize=(8,5))
sns.histplot(
    sample_df["sessions_per_week"],
    bins=20
)

plt.title("Số phiên học mỗi tuần")
plt.show()

# =========================================
# 9. TỶ LỆ HOÀN THÀNH
# =========================================

plt.figure(figsize=(8,5))
sns.histplot(
    sample_df["completion_rate"],
    bins=20
)

plt.title("Tỷ lệ hoàn thành khóa học")
plt.show()

# =========================================
# 10. XU HƯỚNG HỌC TẬP
# =========================================

trend = (
    sample_df.groupby("signup_days_ago")
    ["avg_session_minutes"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10,5))

plt.plot(
    trend["signup_days_ago"],
    trend["avg_session_minutes"]
)

plt.title("Xu hướng học tập")
plt.xlabel("Số ngày kể từ khi đăng ký")
plt.ylabel("Thời gian học trung bình")
plt.grid(True)
plt.show()

# =========================================
# 11. THỜI GIAN HỌC VS HOÀN THÀNH
# =========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=sample_df,
    x="avg_session_minutes",
    y="completion_rate"
)

plt.title(
    "Thời gian học và tỷ lệ hoàn thành"
)

plt.show()

# =========================================
# 12. VIDEO ĐÃ XEM VS MUA KHÓA HỌC
# =========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=sample_df,
    x="future_purchase",
    y="videos_watched"
)

plt.title(
    "Video đã xem và khả năng mua khóa học"
)

plt.show()

# =========================================
# 13. MA TRẬN TƯƠNG QUAN
# =========================================

numeric_cols = [
    "age",
    "signup_days_ago",
    "sessions_per_week",
    "avg_session_minutes",
    "videos_watched",
    "quizzes_taken",
    "forum_posts",
    "completion_rate",
    "courses_enrolled",
    "assignments_submitted",
    "total_spent_usd",
    "discount_used",
    "ai_recommend_click",
    "ai_recommend_enroll",
    "churn_risk",
    "future_purchase"
]

plt.figure(figsize=(12,8))

sns.heatmap(
    sample_df[numeric_cols].corr(),
    cmap="coolwarm"
)

plt.title("Ma trận tương quan")
plt.show()

# =========================================
# 14. PHÁT HIỆN HÀNH VI BẤT THƯỜNG
# =========================================

Q1 = df["avg_session_minutes"].quantile(0.25)
Q3 = df["avg_session_minutes"].quantile(0.75)

IQR = Q3 - Q1

nguoi_hoc_nhieu = df[
    df["avg_session_minutes"]
    > Q3 + 1.5 * IQR
]

print("\nSố người học cực nhiều:")
print(len(nguoi_hoc_nhieu))

dang_ky_nhieu_khong_hoc = df[
    (df["courses_enrolled"] >= 5)
    &
    (df["avg_session_minutes"] < 5)
]

print("\nĐăng ký nhiều nhưng không học:")
print(len(dang_ky_nhieu_khong_hoc))

chi_tieu_cao = df[
    df["total_spent_usd"]
    >
    df["total_spent_usd"].quantile(0.99)
]

print("\nNgười dùng chi tiêu cao:")
print(len(chi_tieu_cao))

# =========================================
# 15. THỐNG KÊ TỔNG HỢP
# =========================================

print("\nTHỜI GIAN HỌC TRUNG BÌNH:")
print(df["avg_session_minutes"].mean())

print("\nTỶ LỆ HOÀN THÀNH TRUNG BÌNH:")
print(df["completion_rate"].mean())

print("\nSỐ PHIÊN HỌC TRUNG BÌNH:")
print(df["sessions_per_week"].mean())

print("\nCHI TIÊU TRUNG BÌNH:")
print(df["total_spent_usd"].mean())
# =========================================
# 16. TOP 10 NGƯỜI CHI TIÊU CAO NHẤT
# =========================================

print("\nTOP 10 NGƯỜI CHI TIÊU CAO NHẤT")

print(
    df.nlargest(
        10,
        "total_spent_usd"
    )[["user_id", "total_spent_usd"]]
)

# =========================================
# 17. NHẬN XÉT
# =========================================

print("\n" + "="*60)
print("NHẬN XÉT")
print("="*60)

print("""
1. Thời gian học càng cao thì tỷ lệ hoàn thành khóa học càng lớn.

2. Có một nhóm người dùng học tập rất tích cực.

3. Một số người dùng đăng ký nhiều khóa học nhưng ít tham gia học.

4. Có sự khác biệt rõ ràng về mức chi tiêu giữa các người dùng.

5. Dữ liệu cho thấy có thể sử dụng AI Recommendation
để cải thiện tỷ lệ đăng ký khóa học.
""")

print("\n HOÀN THÀNH GIAI ĐOẠN 1")
