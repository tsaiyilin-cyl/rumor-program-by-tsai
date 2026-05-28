# -*- coding: utf-8 -*-
import math
from pathlib import Path
import pandas as pd
import numpy as np

def step(path="data_deepseek"):
    '''
    输出一份报告
    单个格子的异常值
    '''
    plots_dir = Path(__file__).parent /path/ "plots"
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


    def inf(matrix, name):
        """
        对矩阵进行方差分解（排除对角线元素），并输出离群格子。
        """
        matrix = np.array(matrix)
        n = matrix.shape[0]  # 假设方阵

        # 创建掩码，排除对角线
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)

        # 提取非对角线元素
        off_diag_vals = matrix[mask]

        # 总体均值（仅基于非对角线）
        mu = np.mean(off_diag_vals)

        # 行均值（仅基于该行非对角线元素）
        row_means = np.array([
            np.mean([matrix[i, j] for j in range(n) if j != i])
            for i in range(n)
        ])

        # 列均值（仅基于该列非对角线元素）
        col_means = np.array([
            np.mean([matrix[i, j] for i in range(n) if i != j])
            for j in range(n)
        ])

        # 行效应和列效应
        alpha = row_means - mu
        beta = col_means - mu

        # 拟合值（仅对非对角线格子有定义，对角线置为NaN或不参与）
        fitted = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i != j:
                    fitted[i, j] = mu + alpha[i] + beta[j]

        # 残差（仅非对角线）
        residuals = np.full((n, n), np.nan)
        residuals[mask] = matrix[mask] - fitted[mask]

        # 残差标准差（仅基于非对角线残差）
        s = np.std(residuals[mask], ddof=(2*n - 1))  # 自由度 = (n*(n-1) - (2n-1))

        # 标准化残差
        std_res = np.full((n, n), np.nan)
        std_res[mask] = residuals[mask] / s

        # 输出基本信息
        print(f"\n--- {name} ---")
        print(f"总体均值 μ (排除对角线) = {mu:.6f}")
        print(f"行效应 α: {alpha}")
        print(f"列效应 β: {beta}")
        print(f"残差标准差 s = {s:.6f}")

        # 离群检测（阈值 k=2/3）2sigma or 3sigma
        kk = [2,3]
        for k in kk:
            outlier_mask = np.abs(std_res) > k

            if np.any(outlier_mask):
                print(f"检测到离群格子 (|标准化残差| > {k}):")
                for i in range(n):
                    for j in range(n):
                        if i != j and outlier_mask[i, j]:
                            print(f"  格子 ({i}, {j}) : 观测值={matrix[i,j]:.4f}, "
                                  f"拟合值={fitted[i,j]:.4f}, 残差={residuals[i,j]:.4f}")
            else:
                print("未检测到离群格子。")

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
                        pos_matrix[src][item['agent']] += contrib
                        matrix[src][item['agent']] += contrib
                    else:
                        neg_matrix[src][item['agent']] += contrib
                        matrix[src][item['agent']] += contrib
        # print(pos_matrix)
        # print(neg_matrix)
        # print(matrix)
        # inf(matrix,"overall_influence")
        inf(pos_matrix, "positive_influence")
        inf(neg_matrix, "negative_influence")
def main():
    step()
if __name__ == "__main__":
    main()