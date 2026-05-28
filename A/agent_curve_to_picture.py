# -*- coding: utf-8 -*-
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def step(path="data_deepseek"):
    model = path.split('_')[1]
    '''
    generate 18 pictures and each topic owns 3
    每个topic会画三个图：
    overall positive negative各一
    每个配置的每个agent影响力求和作为一个数值点，从小到大画出，并涂满曲线下面积，可以看出来数据的趋势
    '''
    plots_dir = Path(__file__).parent /path/ "curve_picture"
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


    def calc(iter,opi):
        return (1.2 ** (-iter)) * (opi ** 1.08 )

    def picture(matrix, name, ind):##曲线填充图
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(range(6))

        # 预先计算每个 agent 的曲线数据
        agent_data = []
        for i in range(6):
            values = []
            for idx in range(30):
                total = 0
                for j in range(6):
                    if i != j and matrix[idx][i][j] != 0:
                        total += matrix[idx][i][j]
                values.append(total)
            sorted_vals = sorted(values)
            agent_data.append((i, sorted_vals, np.mean(sorted_vals)))  # 保存均值用于排序

        # 按均值降序排序（均值大的先绘制，作为底层）
        agent_data.sort(key=lambda x: x[2], reverse=True)

        # 先绘制填充（按排序后的顺序）
        for i, sorted_vals, _ in agent_data:
            x = range(len(sorted_vals))
            plt.fill_between(x, sorted_vals, alpha=0.5, color=colors[i])
        agent_data.sort(key=lambda x: x[0])
        # 再绘制线条
        for i, values, _ in agent_data:
            x = range(len(values))
            plt.plot(x, values, color=colors[i],alpha=1, label=f'Agent {i}', linewidth=2)

        plt.xlabel('Ordered Configuration Index')
        plt.ylabel('Total Influence')
        plt.title(f'{model} Topic{ind} {name} influence by source agent')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = plots_dir / f'topic{ind}_{name}_combined_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # def picture2(matrix, name,ind): ###曲线图
    #     # 创建 2 行 3 列的子图
    #     print(f"topic{ind} name")
    #     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    #     axes = axes.flatten()  # 将二维数组展平成一维，方便索引
    #
    #     # 遍历每个 agent (source)
    #     for i in range(6):
    #         # 提取该行中除对角线外的所有值
    #         values = []
    #         for index in range(30):
    #             o = 0
    #             for j in range(6):
    #                 if i != j and matrix[index][i][j] != 0:
    #                     o += matrix[index][i][j]
    #             # if (o != 0):values.append(o)
    #             values.append(o)
    #         print(values)
    #         print(sum(values))
    #
    #         values_sorted = sorted(values)
    #         axes[i].plot(range(len(values_sorted)), values_sorted,
    #                      marker='o', linestyle='-', markersize=4)
    #         axes[i].set_title(f'Agent {i}')
    #         axes[i].set_ylabel('Total Influence')
    #         axes[i].set_xlabel('Ordered Index')
    #         axes[i].grid(True, linestyle='--', alpha=0.7)
    #
    #     plt.tight_layout()
    #     fig.suptitle(f'Topic{ind} {name} influence (sorted)', fontsize=14, y=1.02)
    #
    #     # 保存图片
    #     save_path = plots_dir / f'topic{ind}_{name}_sorted_curve_panel.png'
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print(f"Saved: {save_path}")

    for i in range(6):
        index = -1 # 配置标签
        print()
        stri = "topic" + str(i)
        print(f"\n{stri}")
        if stri not in topic_events.keys():
            continue
        data = []
        for item in topic_events[stri]:
            dic = to_dict(item)
            data.append(dic)

        # 初始化观点记录表（假设最多10轮，包含第0轮）
        ini = [[None for _ in range(6)] for _ in range(11)]

        # 初始化正向和负向影响力矩阵
        pos_matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        neg_matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        matrix = [[[0.0 for _ in range(6)] for _ in range(6)] for _ in range(30)]
        fl = 0
        for item in data:
            ts = item['ts']
            if ts == 0:
                ini[ts][item['agent']] = item['op']
                if fl == 0:
                    index += 1  # 新配置
                fl = 1
                continue

            fl = 0
            old_op = ini[ts - 1][item['agent']]
            new_op = item['op']

            # 判断变化方向，找出可能的来源 agent
            if new_op > old_op:  # 正向变化
                sources = [a for a in range(6) if ini[ts - 1][a] > old_op]
                direction = 'pos'
            elif new_op < old_op:  # 负向变化
                sources = [a for a in range(6) if ini[ts - 1][a] < old_op]
                direction = 'neg'
            else:
                sources = []
                direction = None

            # 更新该 agent 当前轮的观点
            ini[ts][item['agent']] = new_op

            if sources and direction:
                change = abs(new_op - old_op)
                total_influence = calc(ts, change)

                # 按源 agent 的旧观点计算权重
                weights = []
                for src in sources:
                    opinion_before = ini[ts - 1][src]
                    # 权重规则：观点为0 或 变化幅度为1 → 权重1，否则权重2
                    if opinion_before == 0 or change == 1:
                        weights.append(1)
                    else:
                        weights.append(2)

                total_weight = sum(weights)
                # 按权重分配影响力
                for idx, src in enumerate(sources):
                    contrib = total_influence * (weights[idx] / total_weight)
                    if direction == 'pos':
                        pos_matrix[index][src][item['agent']] += contrib
                        matrix[index][src][item['agent']] += contrib
                    else:  # 'neg'
                        neg_matrix[index][src][item['agent']] += contrib
                        matrix[index][src][item['agent']] += contrib
        # print(pos_matrix)
        # print(neg_matrix)
        # print(matrix)
        picture(matrix,"overall_influence",i)
        picture(pos_matrix, "positive_influence",i)
        picture(neg_matrix, "negative_influence",i)
def main():
    step()

if __name__ == "__main__":
    main()