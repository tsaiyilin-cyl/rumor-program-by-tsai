
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -------------------- 配置 --------------------
model_map = {
    1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct",
    4: "Gemini-3.1-Flash-Lite-Preview", 5: "Qwen3.5-Flash",
    6: "Qwen3.5-35B-A3B"
}

agent_colors = ['#8DC47E', '#F8D9B1', '#B7E1F5', '#D26764', '#DD9AB5', '#9788BC']
baseline_color = '#C8CB6B'
topic_markers = {1: 'o', 2: 's', 3: '^', 4: 'v', 5: 'D', 6: '*'}
topic_labels = {1: 'Topic 1', 2: 'Topic 2', 3: 'Topic 3', 4: 'Topic 4', 5: 'Topic 5', 6: 'Topic 6'}
agent_labels = [f'Agent {i}' for i in range(1, 7)]
with open('influence_data.pkl', 'rb') as f:
    loaded = pickle.load(f)

pos_influence = loaded['pos_influence']
neg_influence = loaded['neg_influence']
pos_stubborn = loaded['pos_stubborn']
neg_stubborn = loaded['neg_stubborn']

posinfa = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
posinfb = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
possuba = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
possubb = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
neginfa = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
neginfb = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
negsuba = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
negsubb = [[[ 0 for _ in range(7)]for _ in range(7)]for _ in range(7)]
# llm topic agent
for llm in range(1,7):
    for topic in range(1,7):
        for exp in range(1,36):
            for agent in range(1,7):
                print(llm,topic,exp,agent,sep='--')
                print(pos_influence[llm][topic][exp][agent])
                print(neg_influence[llm][topic][exp][agent])
                print(pos_stubborn[llm][topic][exp][agent])
                print(neg_stubborn[llm][topic][exp][agent])
                if (exp<=5):
                    possubb[llm][topic][agent] += pos_stubborn[llm][topic][exp][agent] /5
                    posinfb[llm][topic][agent] += pos_influence[llm][topic][exp][agent] /5
                    neginfb[llm][topic][agent] += neg_influence[llm][topic][exp][agent] / 5
                    negsubb[llm][topic][agent] += neg_stubborn[llm][topic][exp][agent] / 5
                else:
                    possuba[llm][topic][agent] += pos_stubborn[llm][topic][exp][agent] / 30
                    posinfa[llm][topic][agent] += pos_influence[llm][topic][exp][agent] / 30
                    neginfa[llm][topic][agent] += neg_influence[llm][topic][exp][agent] / 30
                    negsuba[llm][topic][agent] += neg_stubborn[llm][topic][exp][agent] / 30


# -------------------- 3. 绘图函数 --------------------
def plot_sign(llm, sign, same_scale=True):
    """为一个 LLM 绘制一张包含 8 个子图的图。sign: 'positive' 或 'negative'
       same_scale: 若为 True，则所有子图的横纵轴使用相同的全局尺度"""
    if sign == 'positive':
        inf_a = np.array(posinfa);  inf_b = np.array(posinfb)
        sub_a = np.array(possuba);  sub_b = np.array(possubb)
    else:
        inf_a = np.array(neginfa);  inf_b = np.array(neginfb)
        sub_a = np.array(negsuba);  sub_b = np.array(negsubb)

    # ----- 如果需要统一坐标尺度，先计算全局 min/max -----
    if same_scale:
        all_x = []
        all_y = []
        for topic in range(1, 7):
            for agent in range(1, 7):
                # a 组
                all_x.append(inf_a[llm, topic, agent])
                all_y.append(sub_a[llm, topic, agent])
                # b 组
                all_x.append(inf_b[llm, topic, agent])
                all_y.append(sub_b[llm, topic, agent])
        # 添加平均值图（子图2）中的点也会被包含，因为上面遍历了所有topic和agent
        global_xmin = np.min(all_x)
        global_xmax = np.max(all_x)
        global_ymin = np.min(all_y)
        global_ymax = np.max(all_y)
        # 添加 5% 的边距
        x_margin = 0.05 * (global_xmax - global_xmin) if global_xmax != global_xmin else 0.5
        y_margin = 0.05 * (global_ymax - global_ymin) if global_ymax != global_ymin else 0.5
        global_xlim = (global_xmin - x_margin, global_xmax + x_margin)
        global_ylim = (global_ymin - y_margin, global_ymax + y_margin)

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    # ---- 子图1：总图1（6个topic都用各自的符号） ----
    ax = axes[0]
    for topic in range(1, 7):
        marker = topic_markers[topic]
        x_a = inf_a[llm, topic, 1:7]
        y_a = sub_a[llm, topic, 1:7]
        x_b = inf_b[llm, topic, 1:7]
        y_b = sub_b[llm, topic, 1:7]
        for ag in range(6):
            ax.scatter(x_a[ag], y_a[ag], marker=marker, color=agent_colors[ag],
                       s=150, edgecolors='black', linewidth=0.5, label=None)
            ax.scatter(x_b[ag], y_b[ag], marker=marker, facecolors='none',
                       edgecolors=baseline_color, s=150, linewidth=1.5, label=None)
    ax.set_title(f'LLM {llm} — {sign.capitalize()} (All Topics, Mixed Markers)')
    ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
    ax.grid(True, alpha=0.3)
    if same_scale:
        ax.set_xlim(global_xlim); ax.set_ylim(global_ylim)

    # ---- 子图2：总图2（6个topic的平均，只用圆圈） ----
    ax = axes[1]
    mean_inf_a = np.mean(inf_a[llm, 1:7, 1:7], axis=0)
    mean_sub_a = np.mean(sub_a[llm, 1:7, 1:7], axis=0)
    mean_inf_b = np.mean(inf_b[llm, 1:7, 1:7], axis=0)
    mean_sub_b = np.mean(sub_b[llm, 1:7, 1:7], axis=0)
    for ag in range(6):
        ax.scatter(mean_inf_a[ag], mean_sub_a[ag], marker='o', color=agent_colors[ag],
                   s=150, edgecolors='black', linewidth=0.5, label=agent_labels[ag] if ag == 0 else "")
        ax.scatter(mean_inf_b[ag], mean_sub_b[ag], marker='o', facecolors='none',
                   edgecolors=baseline_color, s=150, linewidth=1.5,
                   label='Baseline' if ag == 0 else "")
    ax.set_title(f'LLM {llm} — {sign.capitalize()} (Average across Topics)')
    ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
    ax.grid(True, alpha=0.3)
    if same_scale:
        ax.set_xlim(global_xlim); ax.set_ylim(global_ylim)

    # ---- 子图3-8：分topic图（每个topic一张图） ----
    for idx, topic in enumerate(range(1, 7), start=2):
        ax = axes[idx]
        marker = topic_markers[topic]
        x_a = inf_a[llm, topic, 1:7]
        y_a = sub_a[llm, topic, 1:7]
        x_b = inf_b[llm, topic, 1:7]
        y_b = sub_b[llm, topic, 1:7]
        for ag in range(6):
            ax.scatter(x_a[ag], y_a[ag], marker=marker, color=agent_colors[ag],
                       s=150, edgecolors='black', linewidth=0.5)
            ax.scatter(x_b[ag], y_b[ag], marker=marker, facecolors='none',
                       edgecolors=baseline_color, s=150, linewidth=1.5)
        ax.set_title(f'Topic {topic}')
        ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
        ax.grid(True, alpha=0.3)
        if same_scale:
            ax.set_xlim(global_xlim); ax.set_ylim(global_ylim)

    # 图例
    legend_elements = []
    for ag in range(6):
        legend_elements.append(
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=agent_colors[ag],
                                    markersize=8, label=agent_labels[ag], markeredgecolor='black', markeredgewidth=0.5)
        )
    legend_elements.append(
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                markeredgecolor=baseline_color, markersize=8, label='Baseline', markeredgewidth=1.5)
    )
    fig.legend(handles=legend_elements, loc='lower center', ncol=7, frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.suptitle(f'{model_map[llm]} — {sign.capitalize()} Influence vs Stubbornness', fontsize=16)

    # 保存图片（文件名加 _samescale）
    folder = model_map[llm]
    os.makedirs(folder, exist_ok=True)
    if same_scale:
        filename = f'{folder}/{sign}_samescale.png'
    else:
        filename = f'{folder}/{sign}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')

for llm in range(1, 7):
    plot_sign(llm, 'positive', same_scale=True)
    plot_sign(llm, 'negative', same_scale=True)
print('All figures generated (same scale).')