import numpy as np
import matplotlib.pyplot as plt

# x 轴
x = np.arange(0, 11)

# 四条曲线的数据
source = np.full_like(x, -1.1, dtype=float)
truth = np.full_like(x, 1.07, dtype=float)

target_scenario_1 = np.array([
    1.0, -0.95, 0, -0.95, -0.95,
    -0.95, -0.95, -0.95, -0.95, -0.95, -0.95
])

target_scenario_2 = np.array([
    1.0, 1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1
])
target_scenario_3 = np.array([
    1.0, -1.03, 0, -1.03, 0,
    -1.03, 0, -1.03, 0, -1.03, -1.03
])
target_scenario_4 = np.array([
    1.0, 1, 1, -1.07, -0.04,
    -1.07, -0.04, -1.07, -0.04, -1.07, -1.07
])
target_scenario_5 = np.array([
    1.0, 1, 1, -0.91,  -0.91, -0.91,
    -0.91, -0.91, -0.91, -0.91, -0.91,
])
# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 9))

# 绘制所有线条，并保存为列表，同时记录图例标签和默认样式
lines = []
labels = ['source', 'target (scenario 1)', 'target (scenario 2)',
          'target (scenario 3)', 'target (scenario 4)','target (scenario 5)', 'truth']
colors = ['red', 'blue', 'purple', 'pink', 'orange','grey', 'green']
y_data = [source, target_scenario_1, target_scenario_2,
          target_scenario_3, target_scenario_4, target_scenario_5,truth]

for y, label, color in zip(y_data, labels, colors):
    line, = ax.plot(x, y, 'o-', color=color, label=label)
    lines.append(line)

# 坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
ax.set_ylabel('opinion_value', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.4)

# 创建图例，并使其可拾取（picker=True）
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.7, 0.85))
for leg_line, line in zip(leg.get_lines(), lines):
    leg_line.set_picker(True)   # 设置图例句柄可点击
    leg_line.set_pickradius(5)  # 点击敏感半径

# 定义点击事件处理函数
def on_pick(event):
    # 获取被点击的图例句柄
    leg_line = event.artist
    # 找到对应的数据线条（通过图例线条的标签匹配）
    label = leg_line.get_label()
    for line, orig_label in zip(lines, labels):
        if orig_label == label:
            # 切换线条可见性
            visible = not line.get_visible()
            line.set_visible(visible)
            # 改变图例句柄的透明度，表示隐藏/显示
            leg_line.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw_idle()
            break

# 连接事件
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()