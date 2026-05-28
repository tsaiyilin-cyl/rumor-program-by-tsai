# -*- coding: utf-8 -*-
import csv
from pathlib import Path
import numpy as np

def step(writer, index: int, path="data_deepseek"):
    logs_dir = Path(__file__).parent / path / "logs"
    topic_events = {}

    # 遍历 logs 下的所有子文件夹
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        topic = subdir.name.split('_')[0]  # "topic0"
        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue
        lines = []
        with open(event_file, 'r', encoding="utf-8", errors='ignore') as f:
            for line in f:
                if '"type": "reflect"' in line:
                    lines.append(line)
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
    #
    # def calc(iter, opi):
    #     return 100*(1.2 ** (-iter)) * (opi ** 1.08)

    for i in range(6):
        stri = "topic" + str(i)
        if stri not in topic_events.keys():
            continue
        data = []
        for item in topic_events[stri]:
            dic = to_dict(item)
            data.append(dic)

        ini = [[None for _ in range(6)] for _ in range(11)]
        fl = 0
        expr = -5
        topic = i + 1
        for item in data:
            ts = item['ts']
            if ts == 0:
                if fl == 0:
                    expr += 1
                fl = 1
                ini[ts][item['agent']] = item['op']
                if item['agent'] == 5:
                    for _ in range(6):
                        sex = _ // 3 + 1
                        race = _ % 3 + 1
                        writer.writerow([sex, race, topic, index, expr, ts, ini[ts][_]])
                continue
            fl = 0
            ini[ts][item['agent']] = item['op']
            ag = item['agent']
            if ag == 5:
                for _ in range(6):
                    sex = _ // 3 + 1
                    race = _ % 3 + 1
                    writer.writerow([sex, race, topic, index, expr, ts,ini[ts][_]])

def main():
    csv_file = Path(__file__).parent / "catch_opinion_6all.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 修改表头：增加 metric 列，合并 influence/subornness 为 value 列
        writer.writerow(["sex", "race", "topic", "llm", "experiment_index", "debate_iter", "opinion"])
        model_list = [
            "data_DeepSeek-V3.2",
            "data_GPT-5.1",
            "data_Llama-3.3-70b-instruct",
            "data_Gemini-3.1-Flash-Lite-Preview",
            "data_Qwen3.5-Flash",
            "data_Qwen3.5-35B-A3B"
        ]
        for idx, model_path in enumerate(model_list, start=1):
            step(writer, idx, path=model_path)

if __name__ == "__main__":
    main()