# -*- coding: utf-8 -*-
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def step(path="data_deepseek"):
    '''
    generate a picture with 6 subplots
    使用分布散点图表征此模型下的影响力分布（分为pos+neg+overall）
    选择小提琴图，且取平方根用以缩放宽度
    '''
    plots_dir = Path(__file__).parent /path/ "box_plot"
    plots_dir.mkdir(exist_ok=True)

    logs_dir = Path(__file__).parent /path/ "logs"
    topic_events = {}

    # 遍历 logs 下的所有子文件夹
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue

        topic = subdir.name.split('_')[0]  #  "topic0"

        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue
        lines = []
        with open(event_file, 'r', encoding='utf-8',errors="ignore") as f:
            lineq = [line for line in f]
            for _ in lineq:
                if('"type": "reflect"' in _):
                    lines.append(_)
        if topic not in topic_events:
            topic_events[topic] = []
        topic_events[topic].extend(lines)


    def to_dict(s):
        dic = {}
        lis = s.split(",")
        ts = int(lis[0].split(":")[1])
        agent = int(lis[1].split(":")[1][-2:-1])
        op = int(lis[3].split(":")[2][:-3])
        dic["ts"] = ts
        dic["agent"] = agent
        dic["op"] = op
        return dic


    def calc(iter,opi):
        return (1.2 ** (-iter)) * (opi ** 1.08 )


    def plot_all_topics_scatter(all_topic_data, plots_dir):
        """
        自定义小提琴图：对核密度值取平方根后缩放宽度，避免底部过宽、顶部过细。
        纵轴使用 symlog对数放缩，保留原始数据分布特征。
        """
        from scipy.stats import gaussian_kde

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        labels = ['Overall', 'Positive', 'Negative']

        for idx, (topic_idx, overall_data, positive_data, negative_data) in enumerate(all_topic_data):
            ax = axes[idx]
            data_list = [overall_data, positive_data, negative_data]

            # 为每个类别单独绘制自定义小提琴
            for pos, (vals, color) in enumerate(zip(data_list, colors)):
                if len(vals) < 1:
                    continue  # 数据太少，跳过

                # 估计概率密度函数 (PDF)
                try:
                    kde = gaussian_kde(vals)
                except:
                    continue

                # 在纵轴范围内生成评估点
                y_min, y_max = 0, 0.75
                y_vals = np.linspace(y_min, y_max, 200)
                density = kde(y_vals)  # 原始密度值

                # 对密度取平方根，减小极端差异；再归一化到 [0,1] 范围
                # density_sqrt = np.sqrt(np.sqrt(density))
                density_sqrt = np.sqrt(density)
                density_norm = density_sqrt / density_sqrt.max()

                max_width = 0.45
                width = density_norm * max_width

                # 绘制左半边和右半边
                left = pos - width
                right = pos + width
                # 使用 fill_betweenx 绘制填充区域
                ax.fill_betweenx(y_vals, left, right, facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)

                # 计算并绘制中位数和均值
                median_val = np.median(vals)
                mean_val = np.mean(vals)
                # 中位数：黑色横线（跨越宽度的一半）
                ax.hlines(median_val, pos - max_width*0.3, pos + max_width*0.3, colors='black', linewidth=2)
                # 均值：蓝色圆点
                ax.plot(pos, mean_val, 'o', color='blue', markersize=4)

            ax.set_title(f'Topic {topic_idx}')
            ax.set_ylabel('Influence')
            ax.set_yscale('symlog', linthresh=0.05, base=2)
            ax.set_ylim(0, 0.75)
            ax.set_yticks([0, 0.125, 0.25, 0.5])
            ax.set_yticklabels(['0', '0.125', '0.25', '0.5'])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(labels)
            ax.grid(True, linestyle='--', alpha=0.7)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.98), fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = plots_dir / 'all_topics_violin_sqrt.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # 主循环中收集数据
    all_topics_data = []

    for i in range(6):
        print()
        stri = "topic" + str(i)
        print(f"\n{stri}")
        if stri not in topic_events.keys():
            continue
        data = []
        for item in topic_events[stri]:
            dic = to_dict(item)
            data.append(dic)

        ini = [[None for _ in range(6)] for _ in range(11)]
        pos_matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        neg_matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        fl = 0
        config_idx = -1
        for item in data:
            ts = item['ts']
            if ts == 0:
                ini[ts][item['agent']] = item['op']
                if fl == 0:
                    config_idx += 1
                fl = 1
                continue
            fl = 0
            old_op = ini[ts - 1][item['agent']]
            new_op = item['op']

            if new_op > old_op:
                sources = [a for a in range(6) if ini[ts - 1][a] > old_op]
                direction = 'pos'
            elif new_op < old_op:
                sources = [a for a in range(6) if ini[ts - 1][a] < old_op]
                direction = 'neg'
            else:
                sources = []
                direction = None

            ini[ts][item['agent']] = new_op

            if sources and direction:
                change = abs(new_op - old_op)
                total_influence = calc(ts, change)

                # 加权分配
                weights = []
                for src in sources:
                    opinion_before = ini[ts - 1][src]
                    if opinion_before == 0 or change == 1:
                        weights.append(1)
                    else:
                        weights.append(2)
                total_weight = sum(weights)

                for idx, src in enumerate(sources):
                    contrib = total_influence * (weights[idx] / total_weight)
                    if direction == 'pos':
                        pos_matrix[config_idx][src][item['agent']] += contrib
                        matrix[config_idx][src][item['agent']] += contrib
                    else:
                        neg_matrix[config_idx][src][item['agent']] += contrib
                        matrix[config_idx][src][item['agent']] += contrib

        # 收集该 topic 所有 agent 的总影响力值（不区分 agent，全部合并）
        def flatten_data(mat):
            values = []
            for idx in range(30):
                for a in range(6):
                    for j in range(6):
                        if a != j:
                            values.append(mat[idx][a][j])
            return values

        overall_vals = flatten_data(matrix)
        positive_vals = flatten_data(pos_matrix)
        negative_vals = flatten_data(neg_matrix)

        all_topics_data.append((i, overall_vals, positive_vals, negative_vals))

    # plot_all_topics_boxplot(all_topics_data, plots_dir)
    plot_all_topics_scatter(all_topics_data, plots_dir)
def main():
    step()
if __name__ == "__main__":
    main()