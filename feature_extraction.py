import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# Step 1: 自定义数据集类（加载 TIFF 文件，支持逐年读取）
class RemoteSensingDataset(Dataset):
    def __init__(self, base_folder, years=[2020], is_train=True):
        self.files = []
        self.labels = []
        for year in years:
            folder = f"{base_folder}_{year}"
            if not os.path.exists(folder):
                print(f"警告: {folder} 不存在，跳过 {year} 年")
                continue
            tiff_files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])
            self.files.extend([os.path.join(folder, f) for f in tiff_files])
            # 模拟标签: 0侵蚀, 1沉积, 2稳定（实际可基于 NDVI 阈值生成）
            labels_year = np.random.randint(0, 3, len(tiff_files))
            self.labels.extend(labels_year)

        if not self.files:
            raise ValueNotFoundError("未找到任何 TIFF 文件，请检查文件夹")

        # 分割训练/验证集
        self.files, self.val_files, self.labels, self.val_labels = train_test_split(
            self.files, self.labels, test_size=0.2, random_state=42
        )

        if is_train:
            self.data_files, self.data_labels = self.files, self.labels
        else:
            self.data_files, self.data_labels = self.val_files, self.val_labels

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        path = self.data_files[idx]
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)  # 读 NDVI 波段
        data = np.expand_dims(data, axis=0)  # 添加通道维度 (1, H, W)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(self.data_labels[idx], dtype=torch.long)
        return data, label


# Step 2: 定义优化 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)  # 添加 Dropout 防止过拟合
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))  # 自适应池化，处理不同尺寸
        self.fc = nn.Linear(32 * 10 * 10, 3)  # 输出 3 类

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def extract_features(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        return x.flatten(1)  # 提取特征


# Step 3: 训练和特征提取
def train_and_extract(base_folder='/home/qshao/Huang/YellowRiverProject', years=[2020]):
    train_dataset = RemoteSensingDataset(base_folder, years, is_train=True)
    val_dataset = RemoteSensingDataset(base_folder, years, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in tqdm(range(10), desc="训练 CNN"):  # 增加 epoch 数优化
        model.train()
        for data, label in train_loader:
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data, label in val_loader:
                out = model(data)
                val_preds.append(torch.argmax(out, dim=1).numpy())
                val_labels.append(label.numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        acc = accuracy_score(val_labels, val_preds) * 100
        print(f"验证准确率: {acc:.2f}%")

    # 提取特征和分类
    model.eval()
    features = []
    preds = []
    with torch.no_grad():
        for data, _ in val_loader:
            feat = model.extract_features(data).numpy()
            out = model(data)
            features.append(feat)
            preds.append(torch.argmax(out, dim=1).numpy())
    features = np.vstack(features)
    preds = np.concatenate(preds)

    # 可视化分类结果
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(preds)), preds)
    plt.title('地貌分类结果 (0:侵蚀, 1:沉积, 2:稳定)')
    plt.savefig('classification_result.png', dpi=300)
    plt.show()

    # 误差测算（混淆矩阵）
    cm = confusion_matrix(val_labels, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    acc = np.mean(preds == val_labels) * 100
    print(f"分类准确率: {acc:.2f}%")
    print(f"提取特征示例: {features[0][:5]}")

    return features, preds


# 主程序
if __name__ == "__main__":
    base_folder = '/home/qshao/Huang/YellowRiverProject'
    years = [2020, 2021, 2022, 2023]  # 逐年读取每个文件夹
    features, preds = train_and_extract(base_folder, years)
    print("✅ 第二问完成！特征提取和分类结果已保存。")