
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
def plot_sign(llm, sign):
    """为一个 LLM 绘制一张包含 8 个子图的图。sign: 'positive' 或 'negative'"""
    if sign == 'positive':
        inf_a = np.array(posinfa);  inf_b = np.array(posinfb)
        sub_a = np.array(possuba);  sub_b = np.array(possubb)
    else:
        inf_a = np.array(neginfa);  inf_b = np.array(neginfb)
        sub_a = np.array(negsuba);  sub_b = np.array(negsubb)

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes = axes.flatten()

    # ---- 子图1：总图1（6个topic都用各自的符号） ----
    ax = axes[0]
    all_x = []  # 收集所有点的 x 坐标
    all_y = []  # 收集所有点的 y 坐标
    for topic in range(1, 7):
        marker = topic_markers[topic]
        x_a = inf_a[llm, topic, 1:7]   # agent 1..6
        y_a = sub_a[llm, topic, 1:7]
        x_b = inf_b[llm, topic, 1:7]
        y_b = sub_b[llm, topic, 1:7]
        # a组：实心，不同颜色
        for ag in range(6):
            ax.scatter(x_a[ag], y_a[ag], marker=marker, color=agent_colors[ag],
                       s=150, edgecolors='black', linewidth=0.8, label=None,alpha=0.5)
            all_x.append(x_a[ag]); all_y.append(y_a[ag])
        # b组：空心，同一baseline颜色
        ax.scatter(x_b, y_b, marker=marker, facecolors='none', edgecolors=baseline_color,
                   s=150, linewidth=1.6, label=None,alpha=0.5)
        for x, y in zip(x_b, y_b):
            all_x.append(x); all_y.append(y)
    # 绘制中位数线
    median_x = np.median(all_x)
    median_y = np.median(all_y)
    ax.axvline(median_x, color='red', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
    ax.axhline(median_y, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
    ax.set_title(f'LLM {llm} — {sign.capitalize()} (All Topics, Mixed Markers)')
    ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
    ax.grid(True, alpha=0.3)

    # ---- 子图2：总图2（6个topic的平均，只用圆圈） ----
    ax = axes[1]
    # 对每个agent计算6个topic的平均
    mean_inf_a = np.mean(inf_a[llm, 1:7, 1:7], axis=0)   # shape (6,)
    mean_sub_a = np.mean(sub_a[llm, 1:7, 1:7], axis=0)
    mean_inf_b = np.mean(inf_b[llm, 1:7, 1:7], axis=0)
    mean_sub_b = np.mean(sub_b[llm, 1:7, 1:7], axis=0)
    all_x = []  # 收集该子图所有点
    all_y = []
    for ag in range(6):
        ax.scatter(mean_inf_a[ag], mean_sub_a[ag], marker='o', color=agent_colors[ag],
                   s=150, edgecolors='black', linewidth=0.8,alpha = 0.5, label=agent_labels[ag] if ag == 0 else "")
        all_x.append(mean_inf_a[ag]); all_y.append(mean_sub_a[ag])
        ax.scatter(mean_inf_b[ag], mean_sub_b[ag], marker='o', facecolors='none',
                   edgecolors=baseline_color, s=150, linewidth=1.6,alpha=0.5,
                   label='Baseline' if ag == 0 else "")
        all_x.append(mean_inf_b[ag]); all_y.append(mean_sub_b[ag])
    # 绘制中位数线
    median_x = np.median(all_x)
    median_y = np.median(all_y)
    ax.axvline(median_x, color='red', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
    ax.axhline(median_y, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
    ax.set_title(f'LLM {llm} — {sign.capitalize()} (Average across Topics)')
    ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
    ax.grid(True, alpha=0.3)

    # ---- 子图3-8：分topic图（每个topic一张图） ----
    for idx, topic in enumerate(range(1, 7), start=2):
        ax = axes[idx]
        marker = topic_markers[topic]
        x_a = inf_a[llm, topic, 1:7]
        y_a = sub_a[llm, topic, 1:7]
        x_b = inf_b[llm, topic, 1:7]
        y_b = sub_b[llm, topic, 1:7]
        all_x = []
        all_y = []
        for ag in range(6):
            ax.scatter(x_a[ag], y_a[ag], marker=marker, color=agent_colors[ag],
                       s=150, edgecolors='black', linewidth=0.8,alpha=0.5)
            all_x.append(x_a[ag]); all_y.append(y_a[ag])
            ax.scatter(x_b[ag], y_b[ag], marker=marker, facecolors='none',
                       edgecolors=baseline_color, s=150, linewidth=1.6,alpha=0.5)
            all_x.append(x_b[ag]); all_y.append(y_b[ag])
        # 绘制中位数线
        median_x = np.median(all_x)
        median_y = np.median(all_y)
        ax.axvline(median_x, color='red', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
        ax.axhline(median_y, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
        ax.set_title(f'Topic {topic}')
        ax.set_xlabel('Influence'); ax.set_ylabel('Stubbornness')
        ax.grid(True, alpha=0.3)

    # ---- 创建统一图例（放在整个 figure 的右侧或底部） ----
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

    folder = model_map[llm]
    os.makedirs(folder, exist_ok=True)
    filename = f'{folder}/{sign}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')

# -------------------- 4. 主程序 --------------------
for llm in range(1, 7):
    plot_sign(llm, 'positive')
    plot_sign(llm, 'negative')
print('All figures generated.')