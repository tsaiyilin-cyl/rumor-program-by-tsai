# -*- coding: utf-8 -*-
import math
from pathlib import Path
import matplotlib.pyplot as plt

def step(path="data_deepseek"):
    model = path.split("_")[1]
    '''
    输出一份报告，有neg/pos影响力值和排名
    generate 6 pictures : each topic has 1
    衡量作为持有观点1/-1的时候agent的影响力，画柱状图（分pos-neg）
    最终输出：
    一个总报告量化6个agent的average-rank(分pos/neg)
    一个总报告量化6个agent的average-true-influence(分pos/neg)
    '''
    plots_dir = Path(__file__).parent /path/ "true_inf"
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

    def inf(matrix,matrix2,topic_name):
        pos_total = [sum(row) for row in matrix]
        neg_total = [sum(row) for row in matrix2]
        agents = list(range(6))
        width = 0.35
        plt.figure(figsize=(12,8))
        bars_pos = plt.bar([i - width / 2 for i in agents], pos_total, width=width,
                label='Positive Influence', color='green')
        bars_neg = plt.bar([i + width / 2 for i in agents], neg_total, width=width,
                label='Negative Influence', color='red')
        ##### add text
        for bar in bars_pos:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}', ha='center', va='bottom')
        for bar in bars_neg:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}', ha='center', va='bottom')
        #####

        plt.xlabel('Agent')
        plt.ylabel('Influence')
        plt.title(f'{model} {topic_name} - Agent True Influence')
        plt.xticks(agents)
        plt.legend()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        # 自动调整布局，确保图例不被裁剪
        plt.tight_layout()
        save_path = plots_dir / f'{topic_name}_influence.png'
        plt.savefig(save_path)
        plt.close()
        return pos_total,neg_total

    def table(topic_name, pos_totals, neg_totals):
        """输出指定话题下各 Agent 的正负影响力及其排名（并列排名）"""
        pos_items = [(value, idx) for idx, value in enumerate(pos_totals)]
        pos_items.sort(key=lambda x: x[0], reverse=True)
        pos_ranks = [0] * 6
        rank = 1
        for i, (value, idx) in enumerate(pos_items):
            if i > 0 and value == pos_items[i - 1][0]:
                pos_ranks[idx] = rank
            else:
                rank = i + 1
                pos_ranks[idx] = rank

        neg_items = [(value, idx) for idx, value in enumerate(neg_totals)]
        neg_items.sort(key=lambda x: x[0], reverse=True)
        neg_ranks = [0] * 6
        rank = 1
        for i, (value, idx) in enumerate(neg_items):
            if i > 0 and value == neg_items[i - 1][0]:
                neg_ranks[idx] = rank
            else:
                rank = i + 1
                neg_ranks[idx] = rank
        ##
        print(f"\n{topic_name} Influence Rankings:")
        print(f"{'Agent':<6}{'Pos Influence':<15}{'Pos Rank':<10}{'Neg Influence':<15}{'Neg Rank':<10}")
        for i in range(6):
            print(f"{i:<6}{pos_totals[i]:<15.2f}{pos_ranks[i]:<10}{neg_totals[i]:<15.2f}{neg_ranks[i]:<10}")
        return pos_ranks,neg_ranks

    all_pos_ranks = []
    all_neg_ranks = []
    all_pos_totals = []
    all_neg_totals = []
    ##主函数
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

        # 初始化观点记录表（假设最多10轮，包含第0轮）
        ini = [[None for _ in range(6)] for _ in range(11)]

        # 初始化正向和负向影响力矩阵
        pos_matrix = [[0.0 for _ in range(6)] for _ in range(6)]
        neg_matrix = [[0.0 for _ in range(6)] for _ in range(6)]
        matrix = [[0.0 for _ in range(6)] for _ in range(6)]

        for item in data:
            ts = item['ts']
            if ts == 0:
                ini[ts][item['agent']] = item['op']
                continue

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

                # 根据方向过滤源 agent（正向：初始观点为1；负向：初始观点为-1）
                if direction == 'pos':
                    filtered_sources = [src for src in sources if ini[0][src] == 1]
                else:
                    filtered_sources = [src for src in sources if ini[0][src] == -1]

                if filtered_sources:
                    # 计算权重（基于变化前的观点）
                    weights = []
                    for src in filtered_sources:
                        opinion_before = ini[ts - 1][src]
                        if opinion_before == 0 or change == 1:
                            weights.append(1)
                        else:
                            weights.append(2)
                    total_weight = sum(weights)

                    # 按权重分配
                    for idx, src in enumerate(filtered_sources):
                        contrib = total_influence * (weights[idx] / total_weight)
                        if direction == 'pos':
                            pos_matrix[src][item['agent']] += contrib
                            matrix[src][item['agent']] += contrib
                        else:
                            neg_matrix[src][item['agent']] += contrib
                            matrix[src][item['agent']] += contrib
        # print(pos_matrix)
        # print(neg_matrix)
        # print(matrix)
        pos_total, neg_total = inf(pos_matrix, neg_matrix,stri)
        pos_ranks, neg_ranks = table(stri, pos_total, neg_total)
        all_pos_ranks.append(pos_ranks)
        all_neg_ranks.append(neg_ranks)
        all_pos_totals.append(pos_total)
        all_neg_totals.append(neg_total)

    if all_pos_ranks:
        num_topics = len(all_pos_ranks)
        avg_pos_ranks = [0.0] * 6
        avg_neg_ranks = [0.0] * 6
        for agent in range(6):
            pos_sum = sum(ranks[agent] for ranks in all_pos_ranks)
            neg_sum = sum(ranks[agent] for ranks in all_neg_ranks)
            avg_pos_ranks[agent] = pos_sum / num_topics
            avg_neg_ranks[agent] = neg_sum / num_topics

        print("\n" + "=" * 50)
        print("Average Rankings Across Topics")
        print("=" * 50)
        print(f"{'Agent':<6}{'Avg Pos Rank':<15}{'Avg Neg Rank':<15}")
        for agent in range(6):
            print(f"{agent:<6}{avg_pos_ranks[agent]:<15.2f}{avg_neg_ranks[agent]:<15.2f}")
    else:
        print("no total ranks")
    if all_pos_totals:
        # 计算平均影响力
        avg_pos_inf = [0.0] * 6
        avg_neg_inf = [0.0] * 6
        for agent in range(6):
            pos_sum_inf = sum(totals[agent] for totals in all_pos_totals)
            neg_sum_inf = sum(totals[agent] for totals in all_neg_totals)
            avg_pos_inf[agent] = pos_sum_inf / num_topics
            avg_neg_inf[agent] = neg_sum_inf / num_topics

        print("\n" + "=" * 50)
        print("Average Influence Across Topics")
        print("=" * 50)
        print(f"{'Agent':<6}{'Avg Pos Influence':<18}{'Avg Neg Influence':<18}")
        for agent in range(6):
            print(f"{agent:<6}{avg_pos_inf[agent]:<18.2f}{avg_neg_inf[agent]:<18.2f}")
def main():
    step()
if __name__ == "__main__":
    main()