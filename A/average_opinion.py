# -*- coding: utf-8 -*-
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot as plotly_plot

def step(path="data_DeepSeek-V3.2"):
    logs_dir = Path(__file__).parent / path / "logs"
    topic_events = {}
    # 遍历 logs 下的所有子文件夹
    for subdir in sorted(logs_dir.iterdir()):  # 字典序升序遍历，这样能够正确筛出baseline
        if not subdir.is_dir():
            continue
        topic = subdir.name.split('_')[0]  # "topic0"
        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue
        lines = []
        with open(event_file, 'r', encoding="utf-8", errors='ignore') as f:
            lineq = [line for line in f]
            for _ in lineq:
                if ('"type": "reflect"' in _):
                    lines.append(_)
        # 将行添加到对应 topic 的列表中
        if topic not in topic_events:
            topic_events[topic] = []
        topic_events[topic].extend(lines)

    # print(topic_events)

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

    def calc(iter, opi):
        return (1.2 ** (-iter)) * (opi ** 1.08)

    tot = [[] for _ in range(6)]
    for i in range(6):
        stri = "topic" + str(i)
        if (stri not in topic_events.keys()): continue
        data = []
        for item in topic_events[stri]:
            dic = to_dict(item)
            data.append(dic)

        ini = [[None for _ in range(6)] for _ in range(11)]
        fl = 0
        expr = -5
        for item in data:
            ts = item['ts']
            if (ts == 0):
                if fl == 0:
                    expr += 1
                    if expr>1:
                        temp2 = []
                        for index_ in range(11):
                            temp = []
                            for _ in range(6):
                                temp.append(ini[index_][_])
                            temp2.append(temp)
                        # print(temp2)
                        tot[i].append(temp2)
                        for _ in range(11):
                            for __ in range(6):
                                ini[_][__] = None
                fl = 1
                ini[ts][item['agent']] = item['op']
                continue
            fl = 0
            ini[ts][item['agent']] = item['op']
        # print(ini)
        tot[i].append(ini)
    return tot


def compute_agent_curves(experiments):
    """
    对于给定的 topic 下的所有实验数据，计算每个 agent 的群体平均观点曲线。

    参数:
        experiments: list，每个元素是一个 11×6 的矩阵（list of list），
                     元素值可能为 0, 1, -1 或 None。

    返回:
        curves: list of list，长度为 6，每个元素是一个长度为 11 的列表，
                表示对应 agent 在各个 iter 上的平均观点值（缺失处为 np.nan）。
    """
    # 初始化：每个 agent 对应一个列表，每个 iter 收集所有符合条件的实验的平均值
    agent_means = [[[] for _ in range(11)] for __ in range(6)]  # [agent][iter] = list of means

    for exp in experiments:
        # 确保 exp 是 11×6 的结构
        if len(exp) < 11 or any(len(row) != 6 for row in exp):
            continue

        # 找出第 0 轮中观点为 1 的 agent
        init_row = exp[0]
        correct_agents = [a for a, val in enumerate(init_row) if val == 1]

        # 如果第 0 轮没有观点为 1 的 agent（理论上不应该发生），跳过该实验
        if not correct_agents:
            continue

        # 一个实验中可能多个 agent 初始为 1？根据数据生成逻辑，每轮只有一个人持有正确观点
        # 但为了安全，每个 agent 单独处理
        for a in correct_agents:
            # 计算每个 iter 的平均观点值
            for t in range(11):
                row = exp[t]
                # 过滤掉 None
                valid = [v for v in row if v is not None]
                if valid:
                    mean_val = sum(valid) / len(valid)
                    agent_means[a][t].append(mean_val)
                # 如果全为 None，则不添加任何值

    # 计算每个 agent 在每个 iter 上的平均值（忽略缺失）
    curves = []
    for a in range(6):
        curve = []
        for t in range(11):
            means = agent_means[a][t]
            if means:
                curve.append(sum(means) / len(means))
            else:
                curve.append(np.nan)
        curves.append(curve)
    return curves


def plot_curves_matplotlib(curves, title, save_path, agent_colors=None,agent_labels=None):
    """
    使用 matplotlib 绘制 6 条曲线并保存为图片。

    参数:
        curves: list of list，长度为 6，每个元素为长度为 11 的曲线数据（可能含 nan）
        title: str，图标题
        save_path: Path 对象，保存路径（应包含 .png 后缀）
        agent_colors: list of str，颜色列表，默认为给定的六种颜色
    """
    if agent_colors is None:
        agent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    plt.figure(figsize=(10, 6))
    x = list(range(11))  # iter 0 ~ 10

    for a in range(6):
        y = curves[a]
        # 如果全为 nan，则不绘制（但 plt.plot 会自动处理，只是不会出现线）
        plt.plot(x, y, marker='o', linestyle='-', color=agent_colors[a], label=f'Agent {a}', linewidth=2, markersize=4)

    plt.title(title)
    plt.xlabel('Iteration (debate round)')
    plt.ylabel('Average group opinion')
    plt.legend(title='Agent (initial correct)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存图片: {save_path}")


def plot_all_topics_in_one_figure(all_curves_per_topic, model, save_path, agent_colors=None,agent_labels=None):
    """
    将6个topic的曲线绘制在一张大图中（2行×3列子图），图例统一放在右侧中央。
    """
    if agent_colors is None:
        agent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    x = list(range(11))

    for topic_idx in range(6):
        ax = axes[topic_idx]
        curves = all_curves_per_topic[topic_idx]
        if curves is None or not any(not np.isnan(v) for curve in curves for v in curve):
            ax.text(0.5, 0.5, f'Topic {topic_idx}\nNo Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Topic {topic_idx}')
            continue

        for a in range(6):
            y = curves[a]
            ax.plot(x, y, marker='o', linestyle='-', color=agent_colors[a],
                    label=agent_labels[a], linewidth=2, markersize=4)
        ax.set_title(f'Topic {topic_idx}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg opinion')
        ax.grid(True, linestyle='--', alpha=0.7)
        # 不在子图中显示图例

    # 隐藏多余子图（如果超过6个，实际不会）
    for i in range(6, len(axes)):
        axes[i].set_visible(False)

    # 获取第一个子图的图例句柄和标签（所有子图相同）
    handles, labels = axes[0].get_legend_handles_labels()
    # 在整张图的右侧添加图例
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5),
               title='Initial correct agent', fontsize='medium')

    plt.suptitle(f'{model}: Average group opinion per initial correct agent (all topics)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # 为右侧图例留出空间
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存复合图: {save_path}")


def main():
    lis = ["data_DeepSeek-V3.2",
           "data_GPT-5.1", "data_Llama-3.3-70b-instruct",
           "data_Gemini-3.1-Flash-Lite-Preview"]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    agent_labels = ["White Man", "Black Man", "Yellow Man",
                    "White Woman", "Black Woman", "Yellow Woman"]
    for i in lis:
        model = i.split("_")[-1]
        tot = step(i)  # 原有step函数，返回6个topic的实验数据列表

        plots_dir = Path(__file__).parent / "4.2"
        plots_dir.mkdir(exist_ok=True)

        # 存储每个topic的curves（用于复合图）
        all_topic_curves = []
        # 存储每个topic下每个agent的曲线，用于总图平均（可选）
        all_curves_for_overall = []

        for topic_idx in range(6):
            experiments = tot[topic_idx]
            if not experiments:
                print(f"Topic {topic_idx} 没有实验数据，跳过")
                all_topic_curves.append(None)
                continue

            curves = compute_agent_curves(experiments)  # 6×11
            all_topic_curves.append(curves)
            all_curves_for_overall.append(curves)

        # 1. 绘制复合图（6个topic在一张图中）
        if any(all_topic_curves):
            save_path_composite = plots_dir / f'{model}_all_topics_curves.png'
            plot_all_topics_in_one_figure(all_topic_curves, model, save_path_composite, colors,agent_labels=agent_labels)
        else:
            print(f"{model} 没有任何topic数据，跳过复合图")

        # 2. 可选：绘制整体平均图（跨topic平均）
        if all_curves_for_overall:
            overall_means = [[[] for _ in range(11)] for __ in range(6)]
            for curves in all_curves_for_overall:
                for a in range(6):
                    for t in range(11):
                        val = curves[a][t]
                        if not np.isnan(val):
                            overall_means[a][t].append(val)
            overall_curves = []
            for a in range(6):
                curve = []
                for t in range(11):
                    vals = overall_means[a][t]
                    curve.append(sum(vals)/len(vals) if vals else np.nan)
                overall_curves.append(curve)
            save_path_overall = plots_dir / f'{model}_overall_agent_curves.png'
            # 复用原来的单图绘制函数（需保留原plot_curves_matplotlib函数）
            plot_curves_matplotlib(overall_curves, f'{model} Overall average (across topics)',
                                   save_path_overall, colors,agent_labels=agent_labels)


if __name__ == "__main__":
    main()