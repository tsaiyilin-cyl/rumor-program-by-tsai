# -*- coding: utf-8 -*-
import ast
import math
from pathlib import Path
import pandas as pd
import numpy as np
import random
import time

def step(path="data_deepseek",model=1):

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
        lines = None
        for enc in ['utf-8', 'gbk']:
            # print(enc)
            try:
                with open(event_file, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError as e:
                pass

        if lines is None:
            print(f"警告: {event_file} 编码无法识别，跳过")
            continue
        if topic not in topic_events:
            topic_events[topic] = []
        topic_events[topic].extend(lines)
    all = []
    for i in range(6):
        expr = -5
        ini = [[None for _ in range(6)] for _ in range(11)]
        print()
        stri = "topic" + str(i)
        print(f"\n{stri}")
        for item in topic_events[stri]:
            # print(item)
            try:
                x = ast.literal_eval(item)
                agent = (int)(x['owner'].split("_")[1])
                ts = (int)(x['ts'])
                # print(x)
                # print(type(x['item']))
                if x['type']== 'talk' and ini[ts-1][agent] != 0:
                    if(ini[ts-1][agent]==-1):
                        eve = [x['item']['content'],'B']
                    else:
                        eve = [x['item']['content'],'A']
                    all.append([model,i+1,agent,ts,eve[0],eve[1],expr])
                if x['type']== 'reflect':
                    ini[ts][agent] = int(x['item']['new_opinion'])
                    if(ts == 0 and agent == 0 ):
                        expr+=1
                        # print(expr)
            except Exception as err:###只有一个有格式错误，不过是listen的，无所谓
                print(item)
                print(err)
    return all
def main():
    all = [ ]
    x = 1
    lis = ["data_DeepSeek-V3.2",
           "data_GPT-5.1", "data_Qwen3.5-Flash",
           "data_Qwen3.5-35B-A3B", "data_Llama-3.3-70b-instruct",
           "data_Gemini-3.1-Flash-Lite-Preview"]
    for i in lis:
        print(i)
        at = step(i,x)
        x+=1
        for _ in at:
            all.append(_)
    df = pd.DataFrame(all, columns=['model','topic','agent','ts','content', 'opinion','expr'])
    df.to_excel('pragmatics.xlsx', index=True, engine='openpyxl')
    print("已保存至 pragmatics.xlsx")
if __name__ == "__main__":
    main()