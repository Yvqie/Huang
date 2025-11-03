"""
rs_cnn_pipeline.py

功能：
1) 读取每年由 GEE 下载并已局部融合的 TIFF（优先读取 fused stack: dongying_fused_stack_YYYY.tif，
   若无则按每月 tiff 合成 stack）。
2) 从每个多波段影像（channels = 月份）生成滑窗 patch，按 patch 的 NDVI 均值自动生成标签（3 类）。
3) 构建可训练 PyTorch Dataset/Model，训练一个轻量 CNN。
4) 在验证集上评估并保存模型、混淆矩阵、分类面图（patch-wise 重构的地图）、t-SNE 特征图、时间序列统计CSV等。
5) 结果便于放论文：分类图、混淆矩阵、分类比例时间序列、特征可视化（t-SNE/UMAP）。

注意：脚本默认把“NDVI 值”直接作为输入（即你的 TIFF 已是 NDVI 或相当指标）。
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import math
import joblib

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------
# Configurable parameters
# ---------------------------------------
PATCH_SIZE = 64        # patch 大小 (平方 patch)
STRIDE = 48            # 滑动步长（可以小于 PATCH_SIZE 以得到重叠）
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NDVI_THRESH = {"erosion": 0.2, "deposition": 0.5}  # <0.2 侵蚀, >0.5 沉积, 其余稳定
RANDOM_SEED = 42

# ---------------------------------------
# Helper: 读取或构建多波段栈 (C, H, W)
# ---------------------------------------
def load_multiband_stack(folder):
    """
    优先读取 fused stack 文件 dongying_fused_stack_YYYY.tif（多波段，band = month）
    若未找到，则按目录下排序的 monthly *.tif 叠成 (C, H, W)
    返回: arr (C, H, W), meta (from rasterio)
    """
    fused_pattern = os.path.join(folder, "dongying_fused_stack_*.tif")
    fused = sorted(glob.glob(fused_pattern))
    if fused:
        path = fused[0]
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (bands, H, W)
            meta = src.meta.copy()
        print(f"读取 fused stack: {path} -> shape {arr.shape}")
        return arr, meta

    # else -> try monthly files (any .tif)
    tiffs = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.tif')])
    if not tiffs:
        raise FileNotFoundError(f"文件夹 {folder} 中未找到任何 tif。")
    # read first to get shape and meta
    with rasterio.open(tiffs[0]) as src:
        H, W = src.height, src.width
        meta = src.meta.copy()
    bands = []
    for p in tiffs:
        with rasterio.open(p) as src:
            band = src.read(1).astype(np.float32)
            if band.shape != (H, W):
                raise ValueError(f"图像尺寸不一致: {p}")
            bands.append(band)
    arr = np.stack(bands, axis=0)  # (C, H, W)
    print(f"从 monthly tif 构建 stack -> shape {arr.shape}")
    return arr, meta

# ---------------------------------------
# Patch generator (返回所有 patch 的左上角坐标)
# ---------------------------------------
def sliding_windows_indices(H, W, patch_size=PATCH_SIZE, stride=STRIDE):
    indices = []
    for y in range(0, max(1, H - patch_size + 1), stride):
        for x in range(0, max(1, W - patch_size + 1), stride):
            indices.append((y, x))
    # 如果右/下缘不足以形成完整 patch，则补上最后一列/行确保覆盖图像边缘
    if indices:
        max_y = indices[-1][0]
        if max_y + patch_size < H:
            for x0 in sorted(set([x for _, x in indices])):
                indices.append((H - patch_size, x0))
        max_x = max([x for _, x in indices])
        if max_x + patch_size < W:
            for y0 in sorted(set([y for y, _ in indices])):
                indices.append((y0, W - patch_size))
        # 角落
        if (H - patch_size, W - patch_size) not in indices:
            indices.append((H - patch_size, W - patch_size))
    else:
        # 图像比 patch 小，直接取单个 patch（会在 dataset 中 pad）
        indices.append((0, 0))
    # 去重并排序
    indices = sorted(list(set(indices)))
    return indices

# ---------------------------------------
# Patch dataset（延迟读取 large array）
# ---------------------------------------
class PatchDataset(Dataset):
    def __init__(self, arr, indices, patch_size=PATCH_SIZE, compute_label=True):
        """
        arr: np.array shape (C, H, W)
        indices: list of (y, x) 左上角
        compute_label: 是否自动以 patch 的均值生成 label（True: 使用 NDVI阈值）
        """
        self.arr = arr
        self.indices = indices
        self.patch_size = patch_size
        self.compute_label = compute_label
        self.C, self.H, self.W = arr.shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        y, x = self.indices[idx]
        ps = self.patch_size
        # pad if necessary
        patch = np.zeros((self.C, ps, ps), dtype=np.float32)
        h_end = min(self.H, y + ps)
        w_end = min(self.W, x + ps)
        h_size = h_end - y
        w_size = w_end - x
        patch[:, :h_size, :w_size] = self.arr[:, y:h_end, x:w_end]
        # replace nan by mean of patch
        if np.isnan(patch).any():
            m = np.nanmean(patch)
            if np.isnan(m):
                m = 0.0
            patch = np.nan_to_num(patch, nan=m)
        # label: use mean across channels+pixels
        if self.compute_label:
            mean_val = patch.mean()
            if mean_val < NDVI_THRESH["erosion"]:
                label = 0
            elif mean_val > NDVI_THRESH["deposition"]:
                label = 1
            else:
                label = 2
        else:
            label = -1
        # to tensor
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        return patch_tensor, torch.tensor(label, dtype=torch.long), (y, x)

# ---------------------------------------
# Simple CNN, 输入 channels 可变
# ---------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128 * 4 * 4, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        # 输出中间特征向量（展平后）
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x.detach()

# ---------------------------------------
# Train / evaluate 工具
# ---------------------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                p = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(p)
                gts.append(yb.numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        val_acc = accuracy_score(gts, preds)
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_acc

# ---------------------------------------
# Reconstruct classification map from patch predictions
# ---------------------------------------
def reconstruct_map(H, W, patch_size, stride, indices, preds, mode='majority'):
    """
    preds: 与 indices 对应的类标签数组
    mode: 'majority' 对重叠区域取多数票；'average' 对概率可做加权（这里用简单票数）
    返回: label_map (H, W)
    """
    count_map = np.zeros((3, H, W), dtype=np.int32)
    for (y, x), p in zip(indices, preds):
        y_end = min(H, y + patch_size)
        x_end = min(W, x + patch_size)
        count_map[p, y:y_end, x:x_end] += 1
    # 如果某像素没有任何覆盖（理论上不会），设为 2（稳定）
    label_map = np.argmax(count_map, axis=0)
    # mask for uncovered pixels
    covered = (count_map.sum(axis=0) > 0)
    label_map[~covered] = 2
    return label_map

# ---------------------------------------
# Save raster
# ---------------------------------------
def save_label_raster(label_map, meta, outpath):
    meta2 = meta.copy()
    meta2.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(outpath, 'w', **meta2) as dst:
        dst.write(label_map.astype(np.uint8), 1)

# ---------------------------------------
# Main pipeline per 年
# ---------------------------------------
def process_year(folder, outdir, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    对单年数据进行：读取、生成 patches、训练/验证、推断、保存结果
    返回字典 summary
    """
    os.makedirs(outdir, exist_ok=True)
    arr, meta = load_multiband_stack(folder)  # arr (C, H, W)
    C, H, W = arr.shape

    # normalize per-channel (简单标准化：去均值/除以 std)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    arr_mean = arr.mean(axis=(1,2), keepdims=True)
    arr_std = arr.std(axis=(1,2), keepdims=True) + 1e-6
    arr_norm = (arr - arr_mean) / arr_std

    # indices
    indices = sliding_windows_indices(H, W, patch_size, stride)
    print(f"生成 {len(indices)} 个 patch 索引 (patch={patch_size}, stride={stride})")

    # dataset & split
    dataset = PatchDataset(arr_norm, indices, patch_size=patch_size, compute_label=True)
    # 划分 train/val
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # model
    model = SimpleCNN(in_channels=C)
    model, best_val_acc = train_model(model, train_loader, val_loader)
    print(f"训练结束, best_val_acc={best_val_acc:.4f}")

    # 保存模型
    model_path = os.path.join(outdir, "cnn_model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": C
    }, model_path)

    # 在所有 patch 上推断并提取特征
    all_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    features = []
    coords = []
    model.eval()
    with torch.no_grad():
        for xb, yb, coord in all_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            p = torch.argmax(out, dim=1).cpu().numpy()
            feats = model.extract_features(xb).cpu().numpy()
            preds.extend(p.tolist())
            features.append(feats)
            coords.extend(coord)
    features = np.vstack(features)
    preds = np.array(preds)
    # reconstruct label map
    label_map = reconstruct_map(H, W, patch_size, stride, indices, preds)
    label_raster_path = os.path.join(outdir, "classification_map.tif")
    save_label_raster(label_map, meta, label_raster_path)
    print(f"分类地图保存: {label_raster_path}")

    # 混淆矩阵（对验证集）
    # gather val preds/gts
    val_preds = []
    val_gts = []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            val_preds.append(torch.argmax(out, dim=1).cpu().numpy())
            val_gts.append(yb.numpy())
    if val_preds:
        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)
        cm = confusion_matrix(val_gts, val_preds)
    else:
        cm = np.zeros((3,3), dtype=int)

    # 分类比例统计（像素级）
    counts = np.bincount(label_map.flatten().astype(int), minlength=3)
    area_percent = counts / counts.sum() * 100
    summary = {
        "model_path": model_path,
        "classification_map": label_raster_path,
        "confusion_matrix": cm,
        "area_percent": area_percent,
        "features": features,
        "preds": preds,
        "coords": indices
    }

    # 保存 features 和 summary
    joblib.dump(features, os.path.join(outdir, "features.joblib"))
    np.save(os.path.join(outdir, "patch_preds.npy"), preds)
    pd.DataFrame({"class_percent": area_percent}).to_csv(os.path.join(outdir, "area_percent.csv"))

    # 可视化并保存图像
    # 1) 混淆矩阵图
    try:
        import seaborn as sns
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title('Confusion Matrix (patch-level val)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=300)
        plt.close()
    except Exception:
        # minimal plotting if seaborn not available
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation='nearest')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=300)
        plt.close()

    # 2) 分类面图（可视化：0=red,1=green,2=blue） - 不指定颜色风格以兼容论文
    cmap = plt.get_cmap('viridis', 3)
    plt.figure(figsize=(10,8))
    plt.imshow(label_map, cmap=cmap, vmin=0, vmax=2)
    plt.title('Classification Map (0:Erosion,1:Deposition,2:Stable)')
    plt.colorbar(ticks=[0,1,2])
    plt.savefig(os.path.join(outdir, "classification_map.png"), dpi=300)
    plt.close()

    # 3) t-SNE 可视化（随机抽样一部分 features 防 OOM）
    sample_n = min(2000, features.shape[0])
    idx = np.random.choice(features.shape[0], sample_n, replace=False)
    feats_sample = features[idx]
    try:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', perplexity=30)
        emb = tsne.fit_transform(feats_sample)
        classes_sample = preds[idx]
        plt.figure(figsize=(6,5))
        for c in np.unique(classes_sample):
            mask = classes_sample == c
            plt.scatter(emb[mask,0], emb[mask,1], label=f'class {c}', s=6)
        plt.legend()
        plt.title('t-SNE of patch features (sample)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "tsne_features.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("TSNE failed:", e)

    # 4) 类别占比柱状图
    plt.figure(figsize=(6,4))
    plt.bar(['erosion','deposition','stable'], area_percent)
    plt.ylabel('Percent (%)')
    plt.title('Area percent by class')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "area_percent.png"), dpi=300)
    plt.close()

    print(f"年度处理完成, 产物保存在: {outdir}")
    return summary

# ---------------------------------------
# 批量处理多年度并生成时间序列 CSV
# ---------------------------------------
def run_pipeline(base_folder, years, output_root):
    os.makedirs(output_root, exist_ok=True)
    records = []
    for y in years:
        folder = f"{base_folder}_{y}"
        outdir = os.path.join(output_root, f"results_{y}")
        print("="*40)
        print(f"Processing year {y} -> {folder}")
        summary = process_year(folder, outdir)
        area = summary["area_percent"]
        records.append({"year": y, "erosion_pct": area[0], "deposition_pct": area[1], "stable_pct": area[2], "model": summary["model_path"]})
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_root, "time_series_area_percent.csv"), index=False)
    # 绘制时间序列图
    plt.figure(figsize=(8,4))
    plt.plot(df['year'], df['erosion_pct'], marker='o', label='erosion')
    plt.plot(df['year'], df['deposition_pct'], marker='o', label='deposition')
    plt.plot(df['year'], df['stable_pct'], marker='o', label='stable')
    plt.xlabel('Year')
    plt.ylabel('Percent (%)')
    plt.title('Class area percent through years')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "time_series_area_percent.png"), dpi=300)
    plt.close()
    print("全部年份处理完毕，时间序列已保存。")

# ---------------------------------------
# CLI
# ---------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RS CNN pipeline (fusion->patch->train->map)")
    parser.add_argument("--base_folder", type=str, default="/home/qshao/Huang/YellowRiverProject",
                        help="数据基路径，不含后缀年份 (示例: /home/qshao/Huang/YellowRiverProject)")
    parser.add_argument("--years", nargs="+", type=int, default=[2020,2021,2022,2023],
                        help="年份列表")
    parser.add_argument("--out", type=str, default="./pipeline_output", help="结果输出根目录")
    parser.add_argument("--patch", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    PATCH_SIZE = args.patch
    STRIDE = args.stride
    run_pipeline(args.base_folder, args.years, args.out)
'''命令示例：python rs_cnn_pipeline.py \
  --base_folder /home/qshao/Huang/YellowRiverProject \
  --years 2020 2021 2022 2023 2024 \
  --out ./pipeline_output
'''