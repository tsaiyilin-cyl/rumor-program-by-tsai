import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# ---------------------- 1. 数据读取与清洗（不变） ----------------------
df = pd.read_excel('annotated_speeches.xlsx', sheet_name='Sheet1')
target_columns = [
    '情感维度得分',
    '来源可信度得分',
    '逻辑性维度得分',
    '总说服力得分',
    'influence'
]
df_target = df[target_columns].copy()
df_target.rename(columns={'influence': 'agent影响力'}, inplace=True)
df_target = df_target.dropna()
for col in df_target.columns:
    df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
df_target = df_target.dropna()

# ---------------------- 2. 相关性计算（不变） ----------------------
print("===== Pearson相关系数矩阵 =====")
corr_matrix = df_target.corr(method='pearson')
print(corr_matrix.round(4))
print("\n===== 各变量对显著性检验 =====")
cols = df_target.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr, p = pearsonr(df_target[cols[i]], df_target[cols[j]])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"{cols[i]} ↔ {cols[j]} | r={corr:.4f} | p={p:.6f} {sig}")

# 设置全局绘图风格（提升美观度）
sns.set_theme(style='whitegrid', palette='pastel', font='SimHei')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ---------------------- 3. 热力图（优化尺寸与注释） ----------------------
plt.figure(figsize=(9, 7))  # 正方形稍扁，适应5个变量
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',          # 红-白-蓝发散色，更醒目
    center=0,
    vmin=-1, vmax=1,
    square=True,            # 强制方格为正方形
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 11}   # 加大相关系数字体
)
plt.title('Pearson 相关性热力矩阵图', fontsize=15, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=600, bbox_inches='tight')
# plt.show()

# ---------------------- 4. 配对图（优化尺寸、点样式、增加回归线） ----------------------
pair_grid = sns.pairplot(
    df_target,
    kind='reg',
    diag_kind='kde',
    plot_kws={
        'scatter_kws': {'alpha': 0.6, 's': 40, 'edgecolor': 'w'},  # 散点样式
        'line_kws': {'color': 'red', 'lw': 1.5}                    # 回归线样式
    },
    diag_kws={'fill': True, 'alpha': 0.5},
    corner=False,
    height=2.2,
    aspect=1.2
)
# 调整全局标题和标签字体
pair_grid.fig.suptitle('两两变量关系图（散点+回归线 & 分布）', y=1.02, fontsize=16)
pair_grid.fig.subplots_adjust(top=0.93)  # 给大标题留空间

# 遍历所有子图，微调字体（可选）
for ax in pair_grid.axes.flat:
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)
        ax.tick_params(labelsize=8)

plt.savefig('pairplot_variables.png', dpi=600, bbox_inches='tight')
# plt.show()