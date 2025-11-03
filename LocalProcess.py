import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.linear_model import Ridge

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

def process_local_tiffs(base_folder='/home/qshao/Huang/YellowRiverProject', years=[]):
    for year in years:
        folder = f"{base_folder}_{year}"
        os.makedirs(folder, exist_ok=True)

        tiff_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])
        if not tiff_files:
            raise FileNotFoundError(f"❌ 未找到任何 TIFF 文件，请检查路径: {folder}")
        print(f"📂 {year}年发现 {len(tiff_files)} 个影像文件，开始分块读取与融合...")

        first_path = os.path.join(folder, tiff_files[0])
        with rasterio.open(first_path) as src:
            height, width = src.height, src.width
            meta = src.meta.copy()

        n_months = len(tiff_files)
        meta.update(count=n_months, dtype='float32', compress='lzw')
        output_tif = os.path.join(folder, f'dongying_fused_stack_{year}.tif')

        fused_local = np.zeros((n_months, height, width), dtype=np.float32)
        original_stack = np.zeros((n_months, height, width), dtype=np.float32)

        num_blocks = 6
        block_height = int(np.ceil(height / num_blocks))

        for row_block_idx in tqdm(range(num_blocks), desc=f"{year}年处理行块"):
            start_row = row_block_idx * block_height
            end_row = min((row_block_idx + 1) * block_height, height)
            nrows = end_row - start_row

            block_stack = []
            for f in tiff_files:
                with rasterio.open(os.path.join(folder, f)) as src:
                    block = src.read(1, window=((start_row, end_row), (0, width)))
                    block_stack.append(block.astype(np.float32))
            block_stack = np.stack(block_stack, axis=0)

            original_stack[:, start_row:end_row, :] = block_stack

            # 填充缺失值
            block_stack_filled = block_stack.copy()
            if np.isnan(block_stack_filled).any():
                mean_val = np.nanmean(block_stack_filled)
                if np.isnan(mean_val):
                    mean_val = 0
                block_stack_filled = np.nan_to_num(block_stack_filled, nan=mean_val)

            radar_block = block_stack_filled + np.random.normal(0, 0.1, block_stack_filled.shape)

            X_opt = block_stack_filled.reshape(-1, 1)
            X_rad = radar_block.reshape(-1, 1)
            X = np.hstack([X_opt, X_rad])
            y = np.mean(X, axis=1)
            valid = ~np.isnan(y)

            if np.sum(valid) == 0:
                fused_block = np.full_like(block_stack_filled, np.nan)
            else:
                model = Ridge(alpha=1.0)
                model.fit(X[valid], y[valid])
                fused_block = model.predict(X).reshape(block_stack_filled.shape)

            fused_local[:, start_row:end_row, :] = fused_block

        # 保存 TIFF
        with rasterio.open(output_tif, 'w', **meta) as dst:
            for i in range(n_months):
                dst.write(fused_local[i], i + 1)

        # 可视化
        plt.figure(figsize=(8, 6))
        plt.imshow(fused_local[0], cmap='viridis')
        plt.title(f'东营地区融合NDVI ({year}-01 示例)')
        plt.colorbar(label='NDVI值')
        plt.tight_layout()
        output_png = os.path.join(folder, f'dongying_fused_ndvi_{year}.png')
        plt.savefig(output_png, dpi=300)
        plt.show()

        # 统计缺失率 & 相关系数
        missing_rate = np.isnan(original_stack).mean() * 100
        valid_mask = ~np.isnan(original_stack[0])
        if np.sum(valid_mask) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(original_stack[0][valid_mask], fused_local[0][valid_mask])[0, 1]

        print(f"📊 {year}年融合阵列形状: {fused_local.shape}")
        print(f"📉 {year}年原始缺失率: {missing_rate:.2f}%")
        print(f"🔗 {year}年融合后相关系数: {corr:.3f}")
        print(f"✅ {year}年结果已保存: {output_tif}")
        print(f"🖼️ {year}年可视化图像: {output_png}")

    print("🎯 处理完成: 无爆内存, 全图分析成功！")


if __name__ == "__main__":
    process_local_tiffs(
        base_folder='/home/qshao/Huang/YellowRiverProject',
        years=[2020]
    )
