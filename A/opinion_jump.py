# -*- coding: utf-8 -*-
import math
from pathlib import Path

def step(path="fdata_deepseek"):
    '''
    观点跳动值=sigma{abs(opi-opi(-1))}
    '''
    plots_dir = Path(__file__).parent/path / "plots"
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

    ans = 0

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
            ini[ts][item['agent']] = item['op']
            if ts == 0:
                continue

            old_op = ini[ts - 1][item['agent']]
            new_op = item['op']
            if(new_op==None):
                continue
            ans += abs(old_op-new_op)
    print(f"{path}opinion_jump={ans}")
def main():
    step()
if __name__ == "__main__":
    main()