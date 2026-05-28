# -*- coding: utf-8 -*-
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import xlsxwriter

# ----------------------------- 数据提取函数（恢复原始逻辑）---------------------------------
def extract_model_data(path):
    model = path.split("_")[-1]
    logs_dir = Path(__file__).parent /path/ "logs"
    topic_events = {}
    result = {model: {}}
    # 遍历 logs 下的所有子文件夹
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue
        topic = subdir.name.split('_')[0]  #  "topic0"
        if topic == "base":
            topic = subdir.name.split("_")[1]
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
        expr = -5
        inf = [[] for _ in range(7)]
        sub = [[] for _ in range(7)]
        for item in data:
            if expr <0:
                base = 1
            else :
                base = 0
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
            if(all > 0):
                if base == 1:
                    sub[6].append(-all)
                else:
                    sub[ag].append(-all)
                for idx, src in enumerate(num):# 按权分配
                    contrib = all * (weights[idx] / total_weight)
                    if base == 1:
                        inf[6].append(contrib)
                    else:
                        inf[src].append(contrib)
        print(f"{model}:influence_topic{i}")
        for it in inf:
            print(it)
        print(f"{model}:subbornness_topic{i}")
        for it in sub:
            print(it)

        # 存储结果
        result[model][i] = {
            agent: {"inf": inf[agent], "sub": sub[agent]} for agent in range(7)
        }

    return result


# ----------------------------- 构建最终表格（42行×8列）---------------------------------
def build_excel_table(all_models_data):
    """
    输入: all_models_data = {model_name: {topic: {agent: {'inf':list, 'sub':list}}}}
    输出: 生成 ans.xlsx，包含 6 topic × 7 agent = 42 行，4 model × 2 metric = 8 列。
    """
    models = list(all_models_data.keys())          # 4个模型
    topics = list(range(6))                        # 话题 0~5
    agents = list(range(7))                        # 个体 0~5 及 baseline(6)

    # 创建 MultiIndex 用于存储格式化字符串
    row_index = pd.MultiIndex.from_product([topics, agents], names=['topic', 'agent'])
    col_index = pd.MultiIndex.from_product([models, ['influence', 'subbornness']], names=['model', 'metric'])
    df = pd.DataFrame(index=row_index, columns=col_index)

    # 独立存储颜色信息
    color_info = {}

    # 1. 先计算所有平均值（数值型），并存储原始值用于检验
    #   同时保留原始数据列表，以便后续做检验
    # 为了简化，我们直接在后续循环中读取 all_models_data 进行检验，并同时填充 df

    for model in models:
        model_data = all_models_data[model]
        for topic in topics:
            if topic not in model_data:
                # 如果该模型没有这个 topic，全部填 0
                for agent in agents:
                    df.loc[(topic, agent), (model, 'influence')] = "0.0000"
                    df.loc[(topic, agent), (model, 'subbornness')] = "0.0000"
                    color_info[(topic, agent, model, 'influence')] = 'black'
                    color_info[(topic, agent, model, 'subbornness')] = 'black'
                continue

            topic_data = model_data[topic]
            # baseline 必须存在，否则无法做检验，但 baseline 本身的行仍需填写
            baseline_exists = (6 in topic_data) and (len(topic_data[6]['inf']) > 0) and (len(topic_data[6]['sub']) > 0)

            for agent in agents:
                if agent not in topic_data or not topic_data[agent]['inf'] or not topic_data[agent]['sub']:
                    # 无数据，平均值为 0
                    mean_inf = 0.0
                    mean_sub = 0.0
                    star_inf = ''
                    star_sub = ''
                    color_inf = 'black'
                    color_sub = 'black'
                else:
                    mean_inf = np.mean(topic_data[agent]['inf'])
                    mean_sub = np.mean(topic_data[agent]['sub'])

                    if agent == 6:   # baseline 行，不加星号，黑色
                        star_inf = ''
                        star_sub = ''
                        color_inf = 'black'
                        color_sub = 'black'
                    else:
                        # 个体行：与 baseline 比较
                        if not baseline_exists:
                            star_inf = ''
                            star_sub = ''
                            color_inf = 'black'
                            color_sub = 'black'
                        else:
                            # 影响力检验
                            try:
                                _, p_inf = mannwhitneyu(topic_data[agent]['inf'], topic_data[6]['inf'], alternative='two-sided')
                            except ValueError:
                                p_inf = 1.0
                            mean_base_inf = np.mean(topic_data[6]['inf'])
                            if p_inf < 0.01:
                                star_inf = '**'
                                color_inf = 'blue' if mean_inf < mean_base_inf else 'red'
                            elif p_inf < 0.05:
                                star_inf = '*'
                                color_inf = 'blue' if mean_inf < mean_base_inf else 'red'
                            else:
                                star_inf = ''
                                color_inf = 'black'

                            # 固执度检验
                            try:
                                _, p_sub = mannwhitneyu(topic_data[agent]['sub'], topic_data[6]['sub'], alternative='two-sided')
                            except ValueError:
                                p_sub = 1.0
                            mean_base_sub = np.mean(topic_data[6]['sub'])
                            if p_sub < 0.01:
                                star_sub = '**'
                                color_sub = 'blue' if mean_sub < mean_base_sub else 'red'
                            elif p_sub < 0.05:
                                star_sub = '*'
                                color_sub = 'blue' if mean_sub < mean_base_sub else 'red'
                            else:
                                star_sub = ''
                                color_sub = 'black'

                # 格式化字符串（保留4位小数）
                inf_str = f"{mean_inf:.2f}{star_inf}"
                sub_str = f"{mean_sub:.2f}{star_sub}"
                df.loc[(topic, agent), (model, 'influence')] = inf_str
                df.loc[(topic, agent), (model, 'subbornness')] = sub_str
                color_info[(topic, agent, model, 'influence')] = color_inf
                color_info[(topic, agent, model, 'subbornness')] = color_sub

    # 2. 写入 Excel
    output_path = Path(__file__).parent / "ans.xlsx"
    workbook = xlsxwriter.Workbook(output_path)
    worksheet = workbook.add_worksheet("Results")

    # 定义格式
    formats = {
        'black': workbook.add_format({'font_color': 'black'}),
        'red': workbook.add_format({'font_color': 'red'}),
        'blue': workbook.add_format({'font_color': 'blue'}),
    }

    # 写表头（第一行，共 2 + 8 列）
    worksheet.write(0, 0, "topic", formats['black'])
    worksheet.write(0, 1, "agent", formats['black'])
    col = 2
    for model in models:
        worksheet.write(0, col, f"{model}\ninfluence", formats['black'])
        worksheet.write(0, col+1, f"{model}\nsubbornness", formats['black'])
        col += 2

    # 写数据行
    row = 1
    for topic in topics:
        for agent in agents:
            worksheet.write(row, 0, topic, formats['black'])
            agent_label = "baseline" if agent == 6 else str(agent)
            worksheet.write(row, 1, agent_label, formats['black'])
            col = 2
            for model in models:
                # influence
                cell_value = df.loc[(topic, agent), (model, 'influence')]
                color = color_info.get((topic, agent, model, 'influence'), 'black')
                worksheet.write(row, col, str(cell_value), formats[color])
                # subbornness
                cell_value = df.loc[(topic, agent), (model, 'subbornness')]
                color = color_info.get((topic, agent, model, 'subbornness'), 'black')
                worksheet.write(row, col+1, str(cell_value), formats[color])
                col += 2
            row += 1

    workbook.close()
    print(f"表格已保存至: {output_path}")


# ----------------------------- 主函数 -------------------------------------
def main():
    model_paths = [
        "data_DeepSeek-V3.2",
        "data_GPT-5.1",
        "data_Llama-3.3-70b-instruct",
        "data_Gemini-3.1-Flash-Lite-Preview"
    ]
    all_data = {}
    for path in model_paths:
        print(f"正在处理模型: {path}")
        result = extract_model_data(path)
        all_data.update(result)
    build_excel_table(all_data)


if __name__ == "__main__":
    main()