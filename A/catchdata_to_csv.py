# -*- coding: utf-8 -*-
import math
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# value_max = 1.2**(-1)*(2**1.08)*30*10
overall = []
def step(writer,index:int,path="data_deepseek"):

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

    for i in range(6):
        stri="topic"+str(i)
        if(stri not in topic_events.keys()):continue
        data = []
        for item in topic_events[stri]:
            dic=to_dict(item)
            data.append(dic)

        ini = [[None for _ in range(6)] for _ in range(11)]
        fl = 0
        expr = 0
        inf = [0 for _ in range(6)]
        sub = [0 for _ in range(6)]
        for item in data:
            num = []
            ts = item['ts']
            if (ts == 0):
                if fl == 0:
                    expr+=1
                fl = 1
                ini[ts][item['agent']] = item['op']
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
            ag = item['agent']
            topic = i+1
            if(all > 0):
                sub [ag] = -all
                for idx, src in enumerate(num):# 按权分配
                    contrib = all * (weights[idx] / total_weight)
                    inf[src] += contrib
            if(ag == 5):#每个时间步记录6条
                for _ in range(6):
                    sex = _ //3 + 1
                    race = _ % 3 + 1
                    data = [sex,race,topic,index,expr,ts,inf[_],sub[_]]
                    writer.writerow(data)
                inf = [0 for _ in range(6)]
                sub = [0 for _ in range(6)]
                #清零

def main():
    '''
    提取数据：
    一个csv文件，共有8列（性别（1-2）、种族（1-3）、topic（1-6）、LLM（1-4）、实验试次（1-30）、讨论轮次（1-10）、影响力、固执度），
    如果在该轮次该agent没有影响力，则影响力为0即可。
    固执度直接取负数
    '''
    csv_file = Path(__file__).parent / "results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["sex", "race", "topic", "llm", "experiment_index", "debate_iter", "influence", "subornness"])
        lis = ["data_DeepSeek-V3.2",
               "data_GPT-5.1", "data_Qwen3.5-Flash",
               "data_Qwen3.5-35B-A3B", "data_Llama-3.3-70b-instruct",
               "data_Gemini-3.1-Flash-Lite-Preview"]
        lis = ["data_DeepSeek-V3.2",
               "data_GPT-5.1",  "data_Llama-3.3-70b-instruct",
               "data_Gemini-3.1-Flash-Lite-Preview"]

        for index,i in enumerate(lis,1):
            step(writer,index,path=i)
            # print(ans)
            # tot.append(ans)

if __name__ == "__main__":
    main()

'''
<=43200 rows
sex       1 male | 2 female
race      1 white | 2 black | 3 yellow
'''