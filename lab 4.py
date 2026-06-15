import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# =====================================================================
# Bài 1: (2đ) Trực quan hóa dữ liệu bằng Heatmap
# =====================================================================
print("--- Đang chạy Bài 1 ---")
# Tải dataset Iris để tính toán [cite: 155]
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Tính ma trận tương quan (correlation matrix) [cite: 156, 157]
corr_matrix = df.corr()

# Sử dụng thư viện Seaborn để vẽ Heatmap [cite: 159, 160, 161]
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,  # Hiển thị số [cite: 164]
    cmap="coolwarm",  # Bản đồ màu [cite: 165]
    linewidths=0.5,  # Độ rộng đường viền ô [cite: 166]
    vmin=-1,
    vmax=1,
)
plt.title("Correlation Heatmap")  # [cite: 167]
plt.show()  # [cite: 168]

"""
Phân tích các cặp biến có tương quan cao (Bài 1): [cite: 169]
- Cặp 'petal length (cm)' và 'petal width (cm)' tương quan thuận cực mạnh (~0.96), nghĩa là chiều dài cánh hoa tăng thì chiều rộng cũng tăng theo.
- 'sepal length (cm)' và 'petal length (cm)' cũng tương quan thuận rất cao (~0.87).
"""

# =====================================================================
# Bài 2: (2đ) Pixel-based Visualization cho dữ liệu lớn
# =====================================================================
print("--- Đang chạy Bài 2 ---")
# Chuẩn bị dataset giả lập có nhiều bản ghi (10,000 phần tử) [cite: 171]
np.random.seed(42)
time = np.linspace(0, 50, 10000)
values = np.sin(time) + np.random.normal(0, 0.2, 10000)  # [cite: 172]

# Biến đổi danh sách bản ghi để biểu diễn mỗi bản ghi bằng một pixel [cite: 173, 174]
size = int(np.ceil(np.sqrt(len(values))))  # [cite: 175]
pixel_matrix = np.zeros(size * size)  # [cite: 176]
pixel_matrix[: len(values)] = values  # [cite: 177]
pixel_matrix = pixel_matrix.reshape(size, size)  # [cite: 178]

# Sử dụng màu sắc để biểu diễn giá trị dữ liệu [cite: 179, 180]
plt.figure(figsize=(6, 6))
plt.imshow(pixel_matrix, cmap="viridis")
plt.colorbar()  # [cite: 181]
plt.title("Pixel-based Visualization")  # [cite: 182]
plt.show()

"""
Phân tích mẫu dữ liệu quan sát được (Bài 2): [cite: 183]
- Biểu đồ hiển thị các dải màu sáng và tối xen kẽ nhau rất đều đặn theo chu kỳ từ trái qua phải, từ trên xuống dưới.
- Mẫu hoa văn này phản ánh trực quan tính chất tuần hoàn (hàm số sin) của dữ liệu mà mắt thường không thấy được trên bảng số.
"""

# =====================================================================
# Bài 3: (2đ) Trực quan dữ liệu đa biến bằng Star Glyph
# =====================================================================
print("--- Đang chạy Bài 3 ---")
# Chọn 4 thuộc tính có sẵn của dữ liệu Iris để biểu diễn [cite: 190, 191]
features = iris.feature_names  # [cite: 192]

# Chuẩn hóa dữ liệu về khoảng [0, 1] [cite: 193]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])  # [cite: 194]


# Hàm vẽ Star Glyph (Radar Chart) cho từng bản ghi dữ liệu [cite: 195, 196, 197]
def star_plot(values, feature_names, label):
    num_vars = len(values)  # [cite: 198]
    angles = np.linspace(
        0, 2 * np.pi, num_vars, endpoint=False
    )  # Tính góc vẽ [cite: 199]

    # Khép kín đa giác hình sao (nối điểm cuối về điểm đầu)
    values = np.concatenate((values, [values[0]]))  # [cite: 200]
    angles = np.concatenate((angles, [angles[0]]))  # [cite: 201]

    plt.figure(figsize=(4, 4))  # [cite: 202, 203]
    ax = plt.subplot(111, polar=True)  # Chuyển sang hệ tọa độ cực (vòng tròn) [cite: 204]

    ax.plot(angles, values, color="teal", linewidth=2)  # Vẽ đường biên [cite: 205]
    ax.fill(angles, values, color="teal", alpha=0.3)  # Đổ màu vùng bên trong [cite: 206]

    # Gán nhãn tên thuộc tính vào các góc tương ứng
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)
    ax.set_title(label, y=1.1)
    plt.show()  # [cite: 207]


# So sánh hình dạng giữa các nhóm dữ liệu thông qua 3 mẫu đầu [cite: 208, 209]
for i in range(3):
    star_plot(scaled_data[i], features, f"Sample {i+1}")  # [cite: 210]

"""
So sánh hình dạng (Bài 3):
- Cả 3 mẫu đầu tiên đều thuộc lớp hoa Setosa nên hình dạng 'ngôi sao' của chúng rất giống nhau: co hẹp mạnh ở các trục cánh hoa (petal) nhưng phình to ở trục sepal width.
"""

# =====================================================================
# Bài 4: (2đ) Trực quan dữ liệu bằng Chernoff Faces
# =====================================================================
print("--- Đang chạy Bài 4 ---")


# Hàm ánh xạ các thuộc tính dữ liệu vào các đặc điểm khuôn mặt [cite: 213, 218]
def draw_face(ax, data):
    face_size = 0.5 + data[0] * 0.5  # kích thước khuôn mặt [cite: 216, 219]
    eye_size = 0.05 + data[1] * 0.05  # kích thước mắt [cite: 214, 219]
    mouth_curve = data[2] - 0.5  # hình dạng miệng (độ cười/mếu) [cite: 215, 220]
    nose_size = 0.05 + data[3] * 0.05  # kích thước mũi [cite: 221]

    # Vẽ khuôn mặt (Hình tròn đại diện) [cite: 226, 227]
    face = plt.Circle((0.5, 0.5), face_size * 0.4, fill=False, linewidth=2)
    ax.add_patch(face)

    # Vẽ mắt trái và mắt phải [cite: 228, 229]
    left_eye = plt.Circle((0.35, 0.6), eye_size, color="black")
    right_eye = plt.Circle((0.65, 0.6), eye_size, color="black")
    ax.add_patch(left_eye)  # [cite: 230]
    ax.add_patch(right_eye)  # [cite: 231]

    # Vẽ mũi [cite: 232, 233]
    nose = plt.Circle((0.5, 0.5), nose_size, color="black")
    ax.add_patch(nose)

    # Vẽ miệng bằng đường cong Parabol [cite: 234]
    x = np.linspace(0.35, 0.65, 100)  # [cite: 235, 236]
    y = 0.35 + mouth_curve * (x - 0.5) ** 2 * -4  # [cite: 237]
    ax.plot(x, y, linewidth=2, color="black")  # [cite: 238]

    # Định cấu hình khung hiển thị nền
    ax.set_xlim(0, 1)  # [cite: 239]
    ax.set_ylim(0, 1)  # [cite: 240]
    ax.axis("off")  # Ẩn hệ trục tọa độ [cite: 241]


# Giả lập ma trận dữ liệu gồm 4 bản ghi để tạo khuôn mặt đại diện [cite: 217]
samples = np.array(
    [
        [0.2, 0.8, 0.9, 0.1],  # Mẫu 1: Mặt nhỏ, mắt to, miệng cười tươi
        [0.9, 0.2, 0.1, 0.9],  # Mẫu 2: Mặt to, mắt tí hí, miệng mếu buồn
        [0.5, 0.5, 0.5, 0.5],  # Mẫu 3: Các nét ở mức cân bằng trung bình
        [0.1, 0.1, 0.9, 0.2],  # Mẫu 4: Mặt siêu nhỏ, miệng cười rộng
    ]
)

# Tạo lưới biểu đồ gồm 2 hàng và 2 cột 
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):  # [cite: 244]
    draw_face(ax, samples[i])  # [cite: 245]
    ax.set_title(f"Sample {i+1}")  # [cite: 246]

plt.tight_layout()
plt.show()

# =====================================================================
# Bài 5: (2đ) Giảng viên mở rộng cho thêm (Violin Plot)
# =====================================================================
print("--- Đang chạy Bài 5 Mở rộng ---")
# Thêm cột target phân loại vào dataframe gốc để vẽ phân bố
df["species"] = iris.target

plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="sepal length (cm)", data=df, palette="muted")
plt.title("Violin Plot - Sepal Length Distribution by Species")
plt.xlabel("Species (Loại hoa)")
plt.ylabel("Sepal Length (cm)")
plt.show()
