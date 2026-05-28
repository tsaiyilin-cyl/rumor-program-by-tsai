import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
lis = [
    [0, 0, 0, -1, 0, 0],   # 白人男性
    [0, 0, 0, -1, 0, 0],   # 黑人男性
    [0, 0, 0, 0, 0, 0],    # 黄人男性
    [0, 0, -1, 0, 0, 0],   # 白人女性
    [0, 0, 0, 0, 0, 0],    # 黑人女性
    [0, 0, -2, 0, 0, 0],   # 黄人女性
    [0, 0, 0, 0, 0, 0]     # 基线
]

rows = ['白人男性', '黑人男性', '黄种人男性', '白人女性', '黑人女性', '黄种人女性', '基线']
cols = [f'{i}' for i in range(1, 7)]
data = np.array(lis)

# 自定义连续 colormap：深蓝 -> 浅灰 -> 深红
colors = ['#0000cd', '#f0f0f0', '#cd0000']  # 中间淡灰色
positions = [0.0, 0.5, 1.0]  # 对应数据最小值-2，中间0，最大值2
cmap = LinearSegmentedColormap.from_list('custom_coolwarm', list(zip(positions, colors)))

plt.figure(figsize=(3, 6))
ax = sns.heatmap(data, annot=False, cmap=cmap, vmin=-2, vmax=2,
                 xticklabels=cols, yticklabels=rows,
                 linewidths=1, linecolor='black',
                 cbar=False)
plt.xlabel("议题")
plt.title("DeepSeek-V3.2固执度显著性矩阵")
plt.tight_layout()
plt.savefig('frame_sub.png', dpi=600)
plt.close()