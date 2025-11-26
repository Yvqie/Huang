import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # <--- 替换ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")

try:
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
except:
    print("未找到文泉驿正黑字体，尝试使用 'SimHei'。")
    mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 步骤1: 读取第三问输出
output_root = './pipeline_output'
csv_path = os.path.join(output_root, 'sediment_simulation_results.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("成功读取第三问的预测结果数据:")
    print(df[['year', 'erosion_pct', 'deposition_pct', 'stable_pct', 'water_discharge', 'sediment_load',
              'predicted_sediment_load_Gt']])
else:
    raise FileNotFoundError("未找到第三问CSV文件，请先运行第三问代码生成 sediment_simulation_results.csv")

# 步骤2: 准备时间序列数据
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.set_index('year', inplace=True)

sequences = {
    'erosion_pct': df['erosion_pct'],
    'deposition_pct': df['deposition_pct'],
    'predicted_sediment_load_Gt': df['predicted_sediment_load_Gt']
}

forecast_steps = 5  # 预测未来5年 (2025-2029)


# 步骤3: 定义动态分析函数 (使用线性回归)
def dynamic_analysis_linear(series, title, ylabel, filename):
    """
    由于 N=5 数据量太少, ARIMA 不适用.
    本函数使用简单的线性回归 (y = a*t + b) 来拟合趋势.
    """
    # 1. 准备数据 (t = 0, 1, 2, 3, 4)
    t_history = np.arange(len(series)).reshape(-1, 1)
    y_history = series.values

    # 2. 拟合线性回归
    model = LinearRegression()
    model.fit(t_history, y_history)

    # 提取斜率 (即每年平均变化量)
    slope = model.coef_[0]

    # 3. In-sample 拟合评估
    fitted = model.predict(t_history)
    rmse = np.sqrt(mean_squared_error(y_history, fitted))
    mae = mean_absolute_error(y_history, fitted)

    # 4. 预测未来 (t = 5, 6, 7, 8, 9)
    t_future = np.arange(len(series), len(series) + forecast_steps).reshape(-1, 1)
    forecast = model.predict(t_future)

    # 5. 估算置信区间 (基于历史RMSE的启发式方法)
    # (注意: 这不是严格的统计置信区间, 而是基于历史误差的简单外推)
    conf_range = 1.96 * rmse
    conf_int_lower = forecast - conf_range
    conf_int_upper = forecast + conf_range

    # 6. 绘图
    plt.figure(figsize=(10, 6))
    years = series.index.year

    # 历史数据
    plt.plot(years, y_history, label='历史数据', marker='o', linewidth=2)

    # 历史拟合线
    plt.plot(years, fitted, label=f'线性拟合 (斜率: {slope:.3f}/年)', linestyle=':', color='green')

    # 预测趋势
    future_years = range(years[-1] + 1, years[-1] + 1 + forecast_steps)
    plt.plot(future_years, forecast, label='线性预测趋势', marker='x', linestyle='--', linewidth=2)

    # 置信区间
    plt.fill_between(future_years, conf_int_lower, conf_int_upper, color='gray', alpha=0.3,
                     label='95% 预测区间 (估算)')

    plt.xlabel('年份')
    plt.ylabel(ylabel)
    plt.title(f'{title} (基于N=5的线性趋势分析)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, filename), dpi=300)
    plt.close()

    # 返回 forecast (Series), slope, rmse, mae
    forecast_series = pd.Series(forecast, index=pd.date_range(start=series.index[-1] + pd.DateOffset(years=1),
                                                              periods=forecast_steps,
                                                              freq='AS-JAN'))
    return forecast_series, slope, rmse, mae


# 步骤4: 对每个序列进行分析
forecasts = {}
metrics = {}
for key, series in sequences.items():
    if key == 'erosion_pct':
        title = '侵蚀百分比动态预测'
        ylabel = '侵蚀百分比 (%)'
        filename = 'dynamic_prediction_erosion.png'
    elif key == 'deposition_pct':
        title = '沉积百分比动态预测'
        ylabel = '沉积百分比 (%)'
        filename = 'dynamic_prediction_deposition.png'
    else:
        title = '预测泥沙负荷动态预测'
        ylabel = '泥沙负荷 (Gt/year)'
        filename = 'dynamic_prediction_sediment.png'

    # --- 调用新的线性分析函数 ---
    forecast, slope, rmse, mae = dynamic_analysis_linear(series, title, ylabel, filename)

    forecasts[key] = forecast
    metrics[key] = {'slope': slope, 'rmse': rmse, 'mae': mae}

print("\n--- 线性趋势预测完成 ---")
print("预测趋势图已保存至 './pipeline_output' 目录: dynamic_prediction_*.png")

# 步骤5: [已修正] 对比地貌变化并识别内外因素 (使用'斜率'作为指标)

erosion_slope = metrics['erosion_pct']['slope']
deposition_slope = metrics['deposition_pct']['slope']
sediment_slope = metrics['predicted_sediment_load_Gt']['slope']


# 根据斜率判断趋势
def get_trend_str(slope):
    if slope > 0.01:
        return '上升'
    elif slope < -0.01:
        return '下降'
    else:
        return '平稳'


conclusion = f"""
--- 动态分析与预测结论报告 ---

**重要提示：** 原始ARIMA模型因数据点过少 (N=5) 导致预测失效 (变为平线)。
这里切换为 **简单线性回归 (y = a*t + b)** 来分析平均趋势。
**警告：** 基于5个数据点的任何外推预测都具有极高的不确定性，仅供参考。

1. 地貌变化对比 (基于线性趋势)：
   - 侵蚀百分比 (erosion_pct):
     - 历史年均变化 (斜率): {erosion_slope:.3f} %/年.
     - 未来5年预测 (线性): {np.round(forecasts['erosion_pct'].values, 3).tolist()}
     - 趋势: {get_trend_str(erosion_slope)}.

   - 沉积百分比 (deposition_pct):
     - 历史年均变化 (斜率): {deposition_slope:.3f} %/年.
     - 未来5年预测 (线性): {np.round(forecasts['deposition_pct'].values, 3).tolist()}
     - 趋势: {get_trend_str(deposition_slope)}.

   - 预测泥沙负荷 (predicted_sediment_load_Gt):
     - 历史年均变化 (斜率): {sediment_slope:.3f} Gt/年.
     - 未来5年预测 (线性): {np.round(forecasts['predicted_sediment_load_Gt'].values, 3).tolist()}
     - 趋势: {get_trend_str(sediment_slope)}.
     - (注: 2021年(0.24 Gt)为高水沙异常点, 2023年(0.13 Gt)为低点, 强烈的波动导致线性趋势 {get_trend_str(sediment_slope)}。)

2. 模型误差分析 (线性回归):
   - 侵蚀百分比: In-sample RMSE {metrics['erosion_pct']['rmse']:.4f}, MAE {metrics['erosion_pct']['mae']:.4f}.
   - 沉积百分比: In-sample RMSE {metrics['deposition_pct']['rmse']:.4f}, MAE {metrics['deposition_pct']['mae']:.4f}.
   - 预测泥沙负荷: In-sample RMSE {metrics['predicted_sediment_load_Gt']['rmse']:.4f}, MAE {metrics['predicted_sediment_load_Gt']['mae']:.4f}.
   - 分析：RMSE值反映了实际数据点与这条生硬的“平均趋势线”的偏离程度。RMSE值相对较大，说明数据本身波动性强，线性拟合效果一般。

3. 内外因素识别：
   - 内部因素：河流动力机制（如水沙关系幂律变化）、地貌演变（侵蚀/沉积平衡受河道几何影响）。
   - 外部因素：人类活动（水库拦截泥沙、生态恢复减少土壤侵蚀，导致泥沙负荷下降90%）；气候变化（降雨减少、水流量波动影响地貌动态）。
   - 结论：基于2020-2024数据，泥沙量在历史低位（0.1-0.2 Gt/年）剧烈波动，这主要归因于**外部因素**的短期扰动（如2021年的极端秋汛导致水沙激增），而长期的人类干预（如小浪底水库）则将泥沙基线压制在低位。

建议：
1. **正视局限性:** 必须在报告中强调，基于5个数据点的任何外推预测都**不具有统计上的可靠性**。
2. **寻求更多数据:** 解决此问题的唯一途径是获取更长的时间序列（如2000-2024年）来训练模型。
3. **定性分析为主:** 目前的结论应侧重于定性描述（如“受人类活动和极端气候影响，泥沙量在低位剧烈波动”），而非定量的趋势预测。

报告生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}
"""

# 保存结论报告
report_path = os.path.join(output_root, 'conclusion_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(conclusion)

print(conclusion)
print(f"结论报告已保存至: {report_path}")
print("动态分析与预测完成。输出：预测趋势图 & 结论报告。")