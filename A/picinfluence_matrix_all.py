# -*- coding: utf-8 -*-
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
value_max = 1.2**(-1)*(2**1.08)*30
def step(path="data_deepseek"):
    model = path.split('_')[1]
    logs_dir = Path(__file__).parent /path/ "logs"
    topic_events = {}

    # 遍历 logs 下的所有子文件夹
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue

        topic = subdir.name.split('_')[0]  #  "topic0"

        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue
        lines=[]
        with open(event_file, 'r',encoding="utf-8",errors='ignore') as f:
            lineq = [line for line in f]
            for _ in lineq:
                if('"type": "reflect"' in _):
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
        dic["ts"]=ts
        dic["agent"]=agent
        dic["op"]=op
        return dic

    def calc(iter,opi):
        return (1.2 ** (-iter)) * (opi ** 1.08 )
    mat = [[0 for _ in range(6)]for _ in range(7)]
    for i in range(6):
        expr = -5
        matrix = [[0.0 for _ in range(7)] for _ in range(7)]
        stri="topic"+str(i)
        if(stri not in topic_events.keys()):continue
        data = []
        for item in topic_events[stri]:
            dic=to_dict(item)
            data.append(dic)

        ini = [[None for _ in range(6)] for _ in range(11)]
        fl = 0
        for item in data:
            num = []
            ts = item['ts']
            if (ts == 0):
                ini[ts][item['agent']] = item['op']
                if fl ==0 :
                    expr+=1
                    fl = 1
                continue
            fl = 0
            if(item['op'] < ini[ts-1][item['agent']]):
                for _ in range(len(ini[ts-1])):
                    if  ini[ts-1][_] < ini[ts-1][item['agent']] :num.append(_)
            elif(item['op'] > ini[ts-1][item['agent']]):
                for _ in range(len(ini[ts - 1])):
                    if ini[ts-1][_] > ini[ts - 1][item['agent']]: num.append(_)
            ini[ts][item['agent']]=item['op']
            op = abs(item['op'] - ini[ts - 1][item['agent']])

            all = calc(ts,op)
            weights = []
            for src in num:
                opinion_before = ini[ts - 1][src]
                if opinion_before == 0 or op == 1:
                    weights.append(1)
                else:
                    weights.append(2)

            total_weight = sum(weights)
            if(expr>0):
                for idx, src in enumerate(num):# 按权分配
                    contrib = all * (weights[idx] / total_weight)
                    matrix[src][item['agent']] += contrib
            else : matrix[6][6] += all

        for ix in range(7):
            for jx in range(7):
                mat[ix][i]+=matrix[ix][jx]
    return mat


def main():
    '''
    解释4.1.1
    每个子图意义是：对于一个LLM而言，画一个6*6的热力图矩阵，体现每个agent在每个topic下的总影响力
    数值为sqrt，不然极差太大
    DeepSeek-V3.2   GPT-5.1  Qwen3.5-Flash   Qwen3.5-35B-A3B
Gemini-3.1-Flash-Llite-Preview   Llama-3.3-70b-instruct
    '''

    lis = ["data_DeepSeek-V3.2",
           "data_GPT-5.1", "data_Qwen3.5-Flash",
           "data_Qwen3.5-35B-A3B", "data_Llama-3.3-70b-instruct",
           "data_Gemini-3.1-Flash-Lite-Preview"]
    labels = ["DeepSeek-V3.2","GPT-5.1","Qwen3.5-Flash","Qwen3.5-35B-A3B","Llama-3.3-70b-instruct","Gemini-3.1-Flash-Lite-Preview"]
    lis = ["data_DeepSeek-V3.2",
           "data_GPT-5.1",  "data_Llama-3.3-70b-instruct",
           "data_Gemini-3.1-Flash-Lite-Preview"]
    labels = ["DeepSeek-V3.2", "GPT-5.1", "Llama-3.3-70b-instruct",
              "Gemini-3.1-Flash-Lite-Preview"]

    tot = []
    for i in lis:
        ans = step(path=i)
        tot.append(ans)
    all_values = np.concatenate([mat for mat in tot])
    vmin = np.min(all_values)
    vmax = np.max(all_values)  # 统一量纲
    # fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    total = []
    for idx, (mat, label) in enumerate(zip(tot, labels)):
        total.append(mat)
        ax = axes[idx]
        sns.heatmap(mat, annot=mat, fmt='.2f', cmap='coolwarm',
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar=False, square=True,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(f'Model: {label}', fontsize=14)
        ax.set_xlabel('Topic Index')
        ax.set_ylabel('Agent Index')
        # 强制坐标轴显示所有刻度
        ax.set_xticks(np.arange(6) + 0.5)
        ax.set_yticks(np.arange(7) + 0.5)
        ax.set_xticklabels(range(6))
        ax.set_yticklabels([str(i) for i in range(6)] + ['baseline'])

    # 添加一个共享的颜色条
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Influence')
    fig.suptitle("Total influence across models and topics", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plots_dir = Path(__file__).parent/ "4.1.1"
    plots_dir.mkdir(exist_ok=True)
    total_neg_path = plots_dir / "totalinf_heatmap_agent_topic64.png"
    plt.savefig(total_neg_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved at{total_neg_path}")

if __name__ == "__main__":
    main()
'''
黄色系
YlOrRd
粉色系
RdPu
'''