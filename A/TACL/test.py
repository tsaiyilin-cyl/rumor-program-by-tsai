import numpy as np
import matplotlib.pyplot as plt

# 定义参数范围
id_vals = np.arange(1.0, 2.0, 0.01)  # id: 1.00, 1.01, ..., 1.99
yd_vals = np.arange(1.0, 1.51, 0.01)  # yd: 1.00, 1.01, ..., 1.50

# 预分配布尔矩阵 (len(yd_vals) 行, len(id_vals) 列)
condition_matrix = np.zeros((len(yd_vals), len(id_vals)), dtype=bool)

# 遍历所有组合
for i_idx, id_val in enumerate(id_vals):
    for y_idx, yd_val in enumerate(yd_vals):
        # 根据当前 id_val 和 yd_val 定义 c 函数
        def c(i, op):
            return id_val ** (-i) * (op ** yd_val)


        # 计算各项
        c12 = c(1, 2)
        c11_c21 = c(1, 1) + c(2, 1)
        c22 = c(2, 2)
        c11_c31 = c(1, 1) + c(3, 1)

        # 检查条件
        if c12 > c11_c21 > c22 > c11_c31:
            condition_matrix[y_idx, i_idx] = True

# 绘制涂色图
plt.figure(figsize=(10, 6))
plt.imshow(condition_matrix, origin='lower', aspect='auto',
           extent=[id_vals.min(), id_vals.max(), yd_vals.min(), yd_vals.max()],
           cmap='gray_r')  # 白色为不满足，黑色为满足（涂色）
plt.colorbar(label='Condition satisfied')
plt.xlabel('id')
plt.ylabel('yd')
plt.scatter(1.2, 1.08, color='red', marker='o', s=50, label='(1.2, 1.08)')
plt.title('Matrix space: satest.pytisfied points are colored (black)')
plt.show()