import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Tải dataset Iris
iris = load_iris()

# 2. Chuyển thành DataFrame để dễ xử lý
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# 3. Vẽ biểu đồ Scatter Plot để xem phân bố dữ liệu
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    palette="Set1",
)

plt.title("Scatter Plot - Iris Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Tải dataset Wine
wine = load_wine()

# 2. Tạo DataFrame
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine["target"] = wine.target

# 3. Vẽ Pairplot (Scatter Matrix) cho một vài thuộc tính tiêu biểu để biểu đồ không bị quá dày
# Ở đây chọn 4 thuộc tính đầu tiên để quan sát rõ ràng hơn
selected_features = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "target"]
sns.pairplot(df_wine[selected_features], hue="target", diag_kind="kde", palette="Set2")

plt.show()
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

# 1. Tải dữ liệu Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["class"] = wine.target

# 2. Chọn ra 5 thuộc tính số như yêu cầu đề bài
features = ["alcohol", "malic_acid", "ash", "flavanoids", "proline"]
df_plot = df[features + ["class"]].copy()

# 3. Chuẩn hóa dữ liệu về khoảng [0, 1] bằng MinMaxScaler để các cột không bị lệch thang đo
scaler = MinMaxScaler()
df_plot[features] = scaler.fit_transform(df_plot[features])

# 4. Vẽ biểu đồ Parallel Coordinates
plt.figure(figsize=(10, 6))
parallel_coordinates(df_plot, "class", colormap=plt.cm.Set1, linewidth=1)

plt.title("Parallel Coordinates Plot - Wine Dataset (Normalized)")
plt.xlabel("Features")
plt.ylabel("Normalized Value")
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Tải dataset MNIST và lấy 2000 mẫu đầu tiên để chạy cho nhanh
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data[:2000]
y = mnist.target[:2000].astype(int)

# --- PHẦN 1: GIẢM CHIỀU BẰNG PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=10)
plt.title("PCA Visualization of MNIST")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Digit Label")
plt.show()

# --- PHẦN 2: GIẢM CHIỀU BẰNG T-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(7, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=10)
plt.title("t-SNE Visualization of MNIST")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(label="Digit Label")
plt.show()
# Gợi ý bài mở rộng: Vẽ Boxplot để tìm ngoại lai cho Iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris["species"] = iris.target

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_iris.drop(columns="species"), palette="Pastel1")
plt.title("Boxplot - Phát hiện dữ liệu ngoại lai (Outliers) trong Iris Dataset")
plt.ylabel("Giá trị (cm)")
plt.show()
