import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==================== 1. 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']        # 使用黑体
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题

# ==================== 2. 参数范围 ====================
id_vals = np.arange(1.0, 1.51, 0.01)      # id: 1.00, 1.01, ..., 1.99
yd_vals = np.arange(1.0, 1.51, 0.01)     # yd: 1.00, 1.01, ..., 1.50

# 预分配布尔矩阵 (len(yd_vals) 行, len(id_vals) 列)
condition_matrix = np.zeros((len(yd_vals), len(id_vals)), dtype=bool)

# ==================== 3. 遍历所有组合 ====================
def compute_c(i, op, id_val, yd_val):
    return id_val ** (-i) * (op ** yd_val)

for i_idx, id_val in enumerate(id_vals):
    for y_idx, yd_val in enumerate(yd_vals):
        c12 = compute_c(1, 2, id_val, yd_val)
        c11_c21 = compute_c(1, 1, id_val, yd_val) + compute_c(2, 1, id_val, yd_val)
        c22 = compute_c(2, 2, id_val, yd_val)
        c11_c31 = compute_c(1, 1, id_val, yd_val) + compute_c(3, 1, id_val, yd_val)

        if c12 > c11_c21 > c22 > c11_c31:
            condition_matrix[y_idx, i_idx] = True

# ==================== 4. 绘制黑白二元色图（不用 BoundaryNorm）====================
cmap = ListedColormap(['white', 'black'])   # 0→白，1→黑
plt.figure(figsize=(10, 6))
im = plt.imshow(condition_matrix, origin='lower', aspect='auto',
                extent=[id_vals.min(), id_vals.max(), yd_vals.min(), yd_vals.max()],
                cmap=cmap, interpolation='nearest')

# colorbar 放到左边，标签距离拉近
cbar = plt.colorbar(im, ticks=[0, 1], location='left', pad=0.1)
cbar.ax.set_yticklabels(['不满足', '满足'])
cbar.set_label('符合约束条件', labelpad=1)   # 数值越小，标签越贴近色条

plt.scatter(1.2, 1.08, color='orange', marker='o', s=20, label='本文所选参数')
blue_points = [(1.04, 1.02), (1.4, 1.10), (1.4, 1.25)]
for x, y in blue_points:
    plt.scatter(x, y, color='#62B187', marker='o', s=20, label=None)

plt.legend(loc='upper left')
plt.xlabel('时间步衰减系数')
plt.ylabel('观点变化增益系数')
plt.title('参数空间')
plt.tight_layout()   # 自动调整布局，避免左右重叠
plt.savefig('parameter_space.png', dpi=600, bbox_inches='tight')
plt.show()