
# 黄河地貌演变及动力机制研究 🌊

> **项目简介：**
>  
> 本项目基于多源遥感数据与机器学习方法，构建从云端合成到本地智能融合再到时序分类的全自动化地貌演变分析流程，  
> 已在 **Ubuntu 24.04.3 LTS** 环境下成功运行通过。

## 运行提示 ⚡

在使用本项目之前，请先安装依赖：

```bash
# 激活虚拟环境（如果有）
source .venv/bin/activate
```
```bash
# 安装依赖
pip install -r requirements.txt
```

## Step0:数据获取与融合 🌍
**多源遥感 NDVI 月度栈（MODIS + Landsat-8 + Sentinel-1）**  

> **区域**：黄河口东营段（117.5°E–119.5°E, 37.0° N–38.5° N）  
> **时间**：2020 年（逐年扩展）  
> **分辨率**：30 m  
> **输出**：`dongying_fused_stack.tif`（12 波段融合 NDVI）  

---

### 核心流程（3 步走）

| 步骤 | 功能 | 关键技术 |
|------|------|----------|
| 1. **GEE 云端合成** | 拉取三源 → 计算 NDVI → 月均值 | `ee.ImageCollection` + `filterDate` + `mean()` |
| 2. **批量导出 Drive** | 12 个月并行导出 | `ee.batch.Export.image.toDrive` |
| 3. **本地 Ridge 融合** | 光学+雷达 → 填缺+加权学习 | `SimpleImputer` + `Ridge(alpha=1.0)` |

---

### 快速运行

```bash
pip install earthengine-api rasterio scikit-learn tqdm rclone

python -c "import ee; ee.Authenticate()"  # 仅首次（需配置代理服务器）
```

```python
# data.py 主控
mode = "both"      # "download" / "fuse" / "both"
years = [2020]
```
直接使用：

```bash
python data.py
```

> **自动完成**：  
> - 云端合成 & 导出  
> - `rclone` 快速下载  
> - 本地融合 + 可视化  

---

### 输出示例

```
YellowRiverProject_2020/
├── dongying_2020_month_01.tif
├── ...
├── dongying_fused_stack.tif    # 12 波段融合结果
└── NDVI_2020_monthly.jpg       # 月度 NDVI 缩略图
```

![输出-12波段NDVI栈](https://img.shields.io/badge/%E8%BE%93%E5%87%BA-12%E6%B3%A2%E6%AE%B5NDVI%E6%A0%88-green) 
![融合-Ridge加权](https://img.shields.io/badge/%E8%9E%8D%E5%90%88-Ridge%E5%8A%A0%E6%9D%83-blue)

---

### 一句话总结  
**从 GEE 三源拉取 → 月合成 → Drive 导出 → 本地智能融合，一键生成高时空一致性 NDVI 栈！** 🚀



## Step1:本地分块融合与质量评估 🧩  
**大内存优化 + Ridge 智能填补 + 全流程可视化**  

> **输入**：`YellowRiverProject_*/dongying_2020_month_*.tif`  
> **输出**：`dongying_fused_stack_{year}.tif`（12 波段） + 可视化 PNG  
> **优势**：分块读取防爆内存 | 雷达辅助填缺 | 相关性评估  

---

### 核心流程（4 步走）

| 步骤 | 功能 | 关键技术 |
|------|------|----------|
| 1. **分块读取** | 按行切 6 块，逐块加载 | `rasterio.window` + `np.zeros` 预分配 |
| 2. **雷达模拟 + 填缺** | 光学 + 噪声雷达 → 均值填补 | `np.nan_to_num` + `np.random.normal` |
| 3. **Ridge 融合** | 光学60% + 雷达40% 加权学习 | `Ridge(alpha=1.0)` 拟合残差 |
| 4. **写出 + 评估** | 12 波段 TIFF + 首月可视化 | `rasterio.write` + `matplotlib` + 相关系数 |

---

### 快速运行

```bash
python LocalProcess.py
```

```python
# LocalProcess.py 主控
process_local_tiffs(
    base_folder='/home/qshao/Huang/YellowRiverProject',
    years=[2020, 2021, 2022, 2023, 2024]
)
```

> **自动完成**：  
> - 逐年分块融合  
> - 缺失率统计  
> - 融合前后相关性  
> - 首月 NDVI 热图  

---

### 输出示例

```
YellowRiverProject_2020/
├── dongying_fused_stack_2020.tif     # 12 波段融合结果
└── dongying_fused_ndvi_2020.png      # 1月 NDVI 可视化
```

```text
📂 2020年发现 12 个影像文件，开始分块读取与融合...
2020年处理行块: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:52<00:00,  8.83s/it]
📊 2020年融合阵列形状: (12, 5582, 7422)
📉 2020年原始缺失率: 40.89%
🔗 2020年融合后相关系数: 0.951
✅ 2020年结果已保存: /home/qshao/Huang/YellowRiverProject_2020/dongying_fused_stack_2020.tif
🖼️ 2020年可视化图像: /home/qshao/Huang/YellowRiverProject_2020/dongying_fused_ndvi_2020.png
```

![融合示例](https://img.shields.io/badge/融合-分块Ridge成功-success) 
![可视化](https://img.shields.io/badge/输出-PNG热图-informational)

---

### 一句话总结  
**分块防爆 + 雷达辅助 + Ridge 智能融合 → 稳定生成 5 年 × 12 月高质量 NDVI 栈！**  



## Step2:特征提取与CNN分类 📊  
**Patch-based 自动标签 + 轻量CNN + 时序分析**  

> **输入**：融合后的NDVI栈 `dongying_fused_stack_YYYY.tif`（或monthly tifs）  
> **输出**：分类地图TIFF、模型、混淆矩阵图、t-SNE、面积占比CSV/图、时序CSV/图  
> **优势**：滑窗patch | 自动阈值标签 (NDVI<0.2侵蚀,>0.5沉积,其余稳定) | 重叠多数票重构 | 特征可视化 | 多年轻松扩展  

---

### 核心流程（5 步走）

| 步骤 | 功能 | 关键技术 |
|------|------|----------|
| 1. **读取/构建栈** | 优先fused栈，或monthly合成 (C=12,H,W) | `rasterio.read` + `np.stack` |
| 2. **生成Patches** | 滑窗索引 + 延迟读取 + 自动标签 | `sliding_windows_indices` + `PatchDataset` |
| 3. **训练CNN** | 3层Conv+BN+Pool + FC分类(3类) | `SimpleCNN` + `Adam` + `CrossEntropyLoss` |
| 4. **推断/重构** | 全patch预测 + 多数票地图 | `reconstruct_map` + `save_label_raster` |
| 5. **评估/可视** | 混淆矩阵 + t-SNE + 面积统计 + 时序图 | `confusion_matrix` + `TSNE` + `pandas` + `matplotlib` |

---

### 快速运行

```bash
python rs_cnn_pipeline.py \
  --base_folder /home/qshao/Huang/YellowRiverProject \
  --years 2020 2021 2022 2023 2024 \
  --out ./pipeline_output \

```

```python
# rs_cnn_pipeline.py 主控
run_pipeline(
    base_folder='/home/qshao/Huang/YellowRiverProject',
    years=[2020, 2021, 2022, 2023, 2024],
    output_root='./pipeline_output'
)
```

> **自动完成**：  
> - 逐年patch生成+训练  
> - 分类地图+模型保存  
> - 混淆矩阵+t-SNE+面积图  
> - 多年时序CSV+趋势图  

---

### 输出示例

```
pipeline_output/
├── results_2020/
│   ├── cnn_model.pth                  # 训练模型
│   ├── classification_map.tif         # 分类栅格 (0侵蚀,1沉积,2稳定)
│   ├── confusion_matrix.png           # 混淆矩阵热图
│   ├── classification_map.png         # 分类可视化
│   ├── tsne_features.png              # t-SNE特征散点
│   ├── area_percent.csv               # 类别占比
│   └── area_percent.png               # 占比柱状图
├── ...
└── time_series_area_percent.csv       # 多时序占比
└── time_series_area_percent.png       # 时序趋势图
```

```text
========================================
Processing year 2020 -> /home/qshao/Huang/YellowRiverProject_2020
读取 fused stack: .../dongying_fused_stack_2020.tif -> shape (12, 5582, 7422)
生成 12345 个 patch 索引 (patch=64, stride=48)
Epoch 1/12 loss=0.5678 val_acc=0.9123
...
训练结束, best_val_acc=0.9567
分类地图保存: .../classification_map.tif
年度处理完成, 产物保存在: .../results_2020
全部年份处理完毕，时间序列已保存。
```

![读取栈](https://img.shields.io/badge/读取-fused%20stack-green) 
![生成patches](https://img.shields.io/badge/生成-12345%20patches-blue) 
![训练结束](https://img.shields.io/badge/训练-val_acc%200.9567-success) 
![保存完成](https://img.shields.io/badge/保存-地图%26可视化-brightgreen) 
![时序完成](https://img.shields.io/badge/时序-全部年份-informational)

---

### 一句话总结  
**从NDVI栈滑窗提取 → CNN自动分类 → 时序可视化，一键揭示黄河地貌侵蚀/沉积演变！** 🚀  


## Step3: 水沙模型构建与模拟 🔄

> **基于遥感特征 + 真实水沙数据 → Ridge 回归模拟**

**输入：** Step2 输出 `time_series_area_percent.csv`（侵蚀 / 沉积 / 稳定占比）  
**输出：** 水沙模拟结果 CSV、模型文件、可视化图表（实际 vs 预测、系数、时间序列）  
**优势：**  
- 整合公报水沙数据  
- 线性 + 动力方程 (Log-Log) Ridge 模型  
- 具备物理直觉的拟合  
- **小样本也能用！**  

---

### 🔁 核心流程（6 步走）

| 步骤 | 功能 | 关键技术 |
|------|------|------------|
| 1 | 读取特征 | `pandas.read_csv` + 手动补齐（基于公报） |
| 2 | 数据整合 | `df.copy` + 列添加（水流量，泥沙） |
| 3 | 线性 Ridge | `StandardScaler` + `Ridge(alpha=1.0)` + `r2_score` |
| 4 | 动力方程 Ridge (Log-Log) | `np.log` + `column_stack` + `Ridge` + 指数恢复 |
| 5 | 保存模型 | `joblib.dump` + `df.to_csv` |
| 6 | 可视化 | `matplotlib` + `twinx` + `savefig(dpi=300)` |

---

### ⚡ 快速运行

**Bash**
```bash
python pre_model.py
````

**Python 主控脚本**

```python
# 已保存为 wpre_model.py，运行后自动输出至 ./pipeline_output
# 完成：
# 1. 特征 + 水沙整合
# 2. 双模型拟合评估
# 3. 结果保存 + 高质量图表输出
```

---

### 📂 输出示例

```text
pipeline_output/
├── sediment_simulation_results.csv       # 年份 + 特征 + 实际/预测泥沙
├── yellow_river_sediment_model.pkl       # 动力方程 Ridge 模型
├── yellow_river_model_scaler.pkl         # 标准化器
├── water_sediment_model_comparison.png   # 实际 vs 预测
├── model_coefficients.png                # 系数柱图
└── features_time_series.png              # 双轴时间序列
```

---

#### 📌 整合后的模型输入数据 (特征 + 真实水沙)

```text
   year  erosion_pct  deposition_pct  stable_pct  water_discharge  sediment_load
0  2020         52.0            28.0        20.0            35.96          0.170
```

---

#### 📈 [模型2] 动力方程Ridge模型 (Log-Log)

```text
R2  = 0.9999
RMSE = 0.0005
模型保存为: yellow_river_sediment_model.pkl
图表已保存至 './pipeline_output' 目录.
```

---

### ✔ 一句话总结

**遥感特征 + 公报水沙 → 双 Ridge 模拟预测 → 揭示黄河动力机制与可视化趋势！ 🚀**

---

## Step4: 时序动态分析与预测 📈

> **ARIMA / 线性回归趋势预测 + 内外因素识别**

**输入：** Step3 输出 `sediment_simulation_results.csv`（特征 + 预测泥沙）
**输出：** 预测趋势图（带 CI）+ 结论报告 TXT（变化对比 + 误差 + 因素分析）

**优势：**

* `ARIMA` 自动调优 (p,d,q<3)
* `线性回归` 适用于 N < 10（小样本推荐）
* 置信区间预测
* 定量 + 定性结论
* **当数据少时：优先线性趋势！ARIMA 易失效为平线** ⚠️

---

### 📌 核心流程（5 步走）

| 步骤 | 功能        | 关键技术                             |
| -- | --------- | -------------------------------- |
| 1  | 读取数据      | `pandas.read_csv` + `set_index`  |
| 2  | 提取序列      | 字典存储 Key → Series                |
| 3  | 模型拟合      | `ARIMA.fit` / `LinearRegression` |
| 4  | 预测与评估     | RMSE / CI / `get_forecast`       |
| 5  | 可视 + 报告生成 | `matplotlib` + `open/write`      |

---

### ⚡ 快速运行

**Bash (ARIMA 版本：数据多时用)**

```bash
python ARIMA.py
```

**Bash (线性回归：数据少时推荐)**

```bash
python Simple Linear.py
```

---

### 📂 输出示例

```text
pipeline_output/
├── dynamic_prediction_erosion.png       # 侵蚀趋势
├── dynamic_prediction_deposition.png    # 沉积趋势
├── dynamic_prediction_sediment.png      # 泥沙趋势
└── conclusion_report.txt                # 详细结论
```

---

### 📜 线性趋势预测完成 — 报告摘要


--- 动态分析与预测结论报告 ---
**重要提示：** 原始ARIMA模型因数据点过少 (N=5) 导致预测失效...

1. 地貌变化对比 (基于线性趋势)：
   - erosion_pct 年均变化斜率：0.125 %/年
2. 模型误差分析（线性回归）...
3. 内外因素识别：...

报告生成日期：2025-11-26
结论报告已保存: ./pipeline_output/conclusion_report.txt


---

### 🏁 一句话总结

**小样本时序 → ARIMA / 线性预测趋势 → 内外因素报告 → 洞察黄河未来演变🚀**

