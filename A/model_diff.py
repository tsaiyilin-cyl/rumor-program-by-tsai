# -*- coding: utf-8 -*-
import math
from pathlib import Path
import matplotlib.pyplot as plt

def step(path="data_deepseek"):
    '''
    generate a picture with 6 subplots
    从观点值角度，每个topic画一个子图，compute_topic_data2返回原值，compute_topic_data返回高斯核卷积值
    不区分agent，连点成线观察model/topic diff
    '''
    plots_dir = Path(__file__).parent /path/ "model_diff"
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
                if '"type": "reflect"' in _:
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

    def gaussian_weight(delta, sigma=1.0):
        """高斯核权重，delta = t - j"""
        return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * (delta / sigma) ** 2)
    def compute_topic_data(ini, sigma=1.0):
        """计算给定ini矩阵的所有点 (t, y) 并返回列表"""
        steps = len(ini)
        n_agents = len(ini[0])
        xs = []
        ys = []
        for agent in range(n_agents):
            history = [ini[t][agent] for t in range(steps)]
            for t in range(steps):
                if ini[t][agent] is None:
                    continue
                total_weight = 0.0
                weighted_sum = 0.0
                for j in range(t+1):
                    delta = t - j
                    w = gaussian_weight(delta, sigma)
                    total_weight += w
                    weighted_sum += history[j] * w
                if total_weight > 0:
                    y = weighted_sum / total_weight
                else:
                    y = 0.0
                xs.append(t)
                ys.append(y)
        return xs, ys

    def compute_topic_data2(ini):
        '''
        直接返回原值
        '''
        steps = len(ini)
        n_agents = len(ini[0]) if steps > 0 else 0
        xs = []
        ys = []
        for agent in range(n_agents):
            for t in range(steps):
                val = ini[t][agent]
                if val is not None:
                    xs.append(t)
                    ys.append(ini[t][agent])   # 直接取原始值
        return xs, ys
    def inf(topics_data, colors, sigma=1.0):

        # plt.figure(figsize=(10, 6))
        # for (topic_name, xs, ys),color in zip(topics_data,colors):
        #     plt.scatter(xs, ys, c=color, label=topic_name, alpha=0.1, s=30)
        # plt.xlabel('Iteration')
        # plt.ylabel('Opinion Value')
        # plt.title(f'All Topics - Opinion Distribution')
        # plt.legend()
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        # plt.grid(True, linestyle='--', alpha=0.7)
        # plt.tight_layout()
        # save_path = plots_dir / 'all_topics_influence.png'
        # plt.savefig(save_path, dpi=150)
        # plt.close()
        # print(f"Saved: {save_path}")

        def split_by_agent(xs, ys):
            segments = []
            if not xs:
                return segments
            start = 0
            for i in range(1, len(xs)):
                if xs[i] <= xs[i - 1]:  # 新智能体开始（x值变小或相等）
                    segments.append((xs[start:i], ys[start:i]))
                    start = i
            segments.append((xs[start:], ys[start:]))
            return segments

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, ((topic_name, xs, ys), color) in enumerate(zip(topics_data, colors)):
            ax = axes[idx]
            segments = split_by_agent(xs, ys)
            for seg_xs, seg_ys in segments:
                if len(seg_xs) >= 2:  # 至少两个点才能连线
                    ax.plot(seg_xs, seg_ys, color=color, alpha=0.1, linewidth=3)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Opinion Value')
            ax.set_title(topic_name)
            ax.grid(True, linestyle='--', alpha=0.7)

        for idx in range(len(topics_data), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = plots_dir / 'all_topics_influence_6axes_2.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    all_topics_data = []
    # colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']  # 对比度最好的6色
    # colors = ["#FF0000","#FFA500","#B3B32E","#00FF00","#0000FF","#800080"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(6):
        x = []
        y = []
        ini = [[None for _ in range(6)] for _ in range(11)]
        print()
        stri = "topic" + str(i)
        print(f"\n{stri}")
        if stri not in topic_events.keys():
            continue
        data = []
        for item in topic_events[stri]:
            dic = to_dict(item)
            data.append(dic)
        f = 0
        for item in data:
            ts = item['ts']
            if ts == 0 and item['agent'] == 0:
                if f == 1:
                    xs, ys = compute_topic_data(ini)
                    for _x in xs:
                        x.append(_x)
                    for _y in ys:
                        y.append(_y)
            f = 1
            ini[ts][item['agent']] = item['op']
        xs, ys = compute_topic_data(ini)
        for _x in xs:
            x.append(_x)
        for _y in ys:
            y.append(_y)
        all_topics_data.append((stri, x, y))
    inf(all_topics_data,colors)
def main():
    step()
if __name__ == "__main__":
    main()