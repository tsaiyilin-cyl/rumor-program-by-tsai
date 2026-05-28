import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv(r"D:\NIMI\Papers\多智能体认知韧性\data\影响力固执度空间.csv")

# 绘图设置（无警告）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
sns.set_style("white")

# 4张核心图（最科学、无冗余、匹配你的负相关结构）
plot_configs = [
    ("ave_Ip", "ave_In", "Group: Positive vs Negative Influence"),
    ("ave_Sp", "ave_Sn", "Group: Positive vs Negative Stubbornness"),
    ("max_Ip", "max_In", "Leader: Max Positive vs Max Negative Influence"),
    ("max_Sp", "max_Sn", "Leader: Max Positive vs Max Negative Stubbornness")
]

# 创建画布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (x_col, y_col, title) in enumerate(plot_configs):
    ax = axes[idx]

    # 构造绘图数据
    heat_df = pd.DataFrame({
        "x": df[x_col],
        "y": df[y_col],
        "value": df["ave_opinion"]
    })

    # 分箱（保证平滑）
    heat_df["x_bin"] = pd.cut(heat_df["x"], bins=15)
    heat_df["y_bin"] = pd.cut(heat_df["y"], bins=15)

    # 计算每个格子的平均观点
    pivot = heat_df.groupby(["x_bin", "y_bin"])["value"].mean().unstack()

    # 绘制热力图
    sns.heatmap(
        pivot.T,
        cmap="coolwarm",
        ax=ax,
        vmin=-1, vmax=1,
        cbar=False,
        linewidths=0.3,
        square=False
    )

    # ========== 显示横纵坐标真实取值 ==========
    ax.set_xticks(np.linspace(0, len(pivot.columns)-1, 15))
    ax.set_yticks(np.linspace(0, len(pivot.index)-1, 15))
    ax.set_xticklabels(np.round(np.linspace(df[x_col].min(), df[x_col].max(), 15), 2))
    ax.set_yticklabels(np.round(np.linspace(df[y_col].min(), df[y_col].max(), 15), 2))

    # 标题与标签
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)

# 统一颜色条 ✅ 修复语法错误
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
fig.colorbar(sm, cax=cbar_ax).set_label("Final Average Opinion\n(1=correct, -1=wrong)")

plt.tight_layout(rect=[0, 0, 0.91, 0.96])
plt.savefig(r"D:\NIMI\Papers\多智能体认知韧性\Figures\heatmap_4in1_final.png", dpi=300, bbox_inches='tight')
plt.show()