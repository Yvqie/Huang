import ee
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import rasterio
import os
import time
from tqdm import tqdm
import subprocess

# ========== Step 0: 模式控制 ==========
mode = "download"  # "download" / "fuse" / "both"

# ========== Step 1: GEE 初始化 ==========
try:
    ee.Initialize(project='ee-2171568961')
    print("✅ GEE 初始化成功")
except Exception as e:
    print(f"❌ GEE初始化失败: {e}\n请运行 ee.Authenticate() 并重试。")
    exit(1)

# ========== Step 2: 定义研究区 ==========
region = ee.Geometry.Rectangle([117.5, 37.0, 119.5, 38.5])

# ========== Step 3: 获取多源数据 ==========
def get_data(start_year, end_year):
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    modis = ee.ImageCollection('MODIS/061/MOD13Q1') \
        .filterDate(start_date, end_date).filterBounds(region) \
        .select('NDVI') \
        .map(lambda img: img.divide(10000).clamp(-1, 1).toFloat().set('system:time_start', img.get('system:time_start')))
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate(start_date, end_date).filterBounds(region) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
        .select(['SR_B4', 'SR_B5'])
    sentinel = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date).filterBounds(region) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select('VV')
    return modis, landsat, sentinel

# ========== Step 4: 预处理 ==========
def compute_ndvi(image):
    return image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI').toFloat().clamp(-1, 1)

def normalize_sentinel(img):
    return img.unitScale(-30, 0).rename('NDVI').toFloat().clamp(0, 1)

def process_collection(start_year, end_year, batch_size=12):
    modis, landsat, sentinel = get_data(start_year, end_year)
    landsat_ndvi = landsat.map(compute_ndvi)
    sentinel_ndvi = sentinel.map(normalize_sentinel)
    fused_collection = modis.merge(landsat_ndvi).merge(sentinel_ndvi)

    def monthly_mean(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        return fused_collection.filterDate(start, end).mean().toFloat().clip(region).set('system:time_start', start.millis())

    months = ee.List.sequence(0, (end_year - start_year + 1) * 12 - 1)
    dates = months.map(lambda m: ee.Date(f'{start_year}-01-01').advance(m, 'month'))
    monthly_images = ee.ImageCollection(dates.map(monthly_mean))
    print(f"📦 {start_year} 年共生成 {monthly_images.size().getInfo()} 个月数据")
    return monthly_images

# ========== Step 5: 导出函数 ==========
def export_batch(images, start_idx, batch_size, year, drive_folder):
    tasks = []
    image_list = images.toList(batch_size)
    for i in tqdm(range(batch_size), desc=f"导出批次 {start_idx // batch_size + 1}"):
        try:
            img = ee.Image(image_list.get(i))
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=f'dongying_{year}_month_{i + 1:02d}',
                folder=drive_folder,
                fileNamePrefix=f'dongying_{year}_month_{i + 1:02d}',
                scale=30,
                region=region,
                maxPixels=1e13
            )
            task.start()
            tasks.append(task)
        except Exception as e:
            print(f"任务 {start_idx + i} 启动失败: {e}")

    print("⏳ 正在监控任务进度...")
    completed = [False] * len(tasks)
    while not all(completed):
        for i, task in enumerate(tasks):
            if not completed[i]:
                state = task.status()['state']
                if state in ['COMPLETED', 'FAILED']:
                    completed[i] = True
                    print(f"任务 {i+1}/{batch_size} 状态: {state}")
        time.sleep(30)
    print(f"✅ {drive_folder} 导出完成")

# ========== Step 6: 本地融合 ==========
def process_local_tiffs(folder):
    print(f"🧩 开始融合 {folder}")
    tiff_files = sorted([f for f in os.listdir(folder) if f.endswith('.tif')])
    if not tiff_files:
        print(f"⚠️ 文件夹 {folder} 中无 .tif 文件，跳过。")
        return

    data_stack = []
    for file in tqdm(tiff_files, desc=f"读取 {folder}"):
        with rasterio.open(os.path.join(folder, file)) as src:
            data_stack.append(src.read(1))
            meta = src.meta

    data_optical = np.stack(data_stack, axis=0)
    data_radar = data_optical + np.random.normal(0, 0.1, data_optical.shape)

    imputer = SimpleImputer(strategy='mean')
    data_optical = imputer.fit_transform(data_optical.reshape(data_optical.shape[0], -1)).reshape(data_optical.shape)
    data_radar = imputer.fit_transform(data_radar.reshape(data_radar.shape[0], -1)).reshape(data_radar.shape)

    X = np.column_stack((data_optical.reshape(data_optical.shape[0], -1) * 0.6,
                         data_radar.reshape(data_optical.shape[0], -1) * 0.4))
    y = np.mean(X, axis=1)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    fused_local = model.predict(X).reshape(data_optical.shape)

    meta.update(count=fused_local.shape[0])
    out_path = os.path.join(folder, 'dongying_fused_stack.tif')
    with rasterio.open(out_path, 'w', **meta) as dst:
        for i in range(fused_local.shape[0]):
            dst.write(fused_local[i], i + 1)
    print(f"✅ 融合完成，输出文件: {out_path}")

# ========== Step 7: 检查已提交任务 ==========
def gee_tasks_status(prefix, months):
    """
    检查每个月份任务状态
    返回一个 dict: {month_index: 'SUCCEEDED' / 'CANCELLED' / 'PENDING' / 'NONE'}
    """
    status_dict = {i: 'NONE' for i in range(1, months+1)}
    try:
        ops = ee.data.listOperations()  # 返回 list
        for op in ops:
            meta = op.get('metadata', {}) if isinstance(op, dict) else {}
            name = meta.get('description', '') if isinstance(meta, dict) else ''
            state = meta.get('state', '') if isinstance(meta, dict) else ''
            for i in range(1, months+1):
                if name.startswith(f"{prefix}{i:02d}"):
                    # 保留最新 SUCCEEDED 状态
                    if status_dict[i] != 'SUCCEEDED':
                        status_dict[i] = state
        for i, st in status_dict.items():
            print(f"🟢 月份 {i:02d} 任务状态：{st}")
        return status_dict
    except Exception as e:
        print(f"⚠️ 检查 GEE 任务状态失败: {e}")
        return {i: 'NONE' for i in range(1, months+1)}


# ========== Step 8: rclone 下载 ==========
def download_from_drive(local_folder, drive_folder):
    """
    local_folder: 本地保存路径
    drive_folder: Google Drive 上的文件夹，保持不变
    """
    os.makedirs(local_folder, exist_ok=True)
    try:
        subprocess.run(['rclone', 'copy', f'drive:{drive_folder}', local_folder, '--progress'], check=True)
        print(f"✅ 已从 Google Drive 下载 '{drive_folder}' 到本地 '{local_folder}'")
    except Exception as e:
        print(f"❌ 下载失败: {e}")


# ========== Step 9: 主控制逻辑 ==========
batch_size = 12
years = [2020]

for year in years:
    local_folder = f'YellowRiverProject_{year}'  # 每年独立本地文件夹
    drive_folder = f'YellowRiverProject_{year}'
    os.makedirs(local_folder, exist_ok=True)

    # 下载阶段
    if mode in ["download", "both"]:
        prefix = f'dongying_{year}_month_'
        month_status = gee_tasks_status(prefix, batch_size)
        to_submit = [i for i, st in month_status.items() if st != 'SUCCEEDED']

        if not to_submit:
            print(f"✅ {year} 年所有月份任务已完成，直接下载到本地")
            download_from_drive(local_folder,drive_folder)  # 只指定本地文件夹，Drive 保持默认
        else:
            print(f"🚀 提交尚未完成的月份任务: {to_submit}")
            monthly_images = process_collection(year, year)
            export_batch(monthly_images, (year - 2020) * 12, batch_size, year, f'YellowRiverProject_{year}')


    # 融合阶段
    if mode in ["fuse", "both"]:
        process_local_tiffs(local_folder)
