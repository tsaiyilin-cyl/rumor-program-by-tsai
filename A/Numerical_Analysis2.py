# -*- coding: utf-8 -*-
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import xlsxwriter

def extract_model_data():
    # ... （保持不变，与原脚本相同）
    df = pd.read_csv("results_addbase.csv")
    model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct", 4: "Gemini-3.1-Flash-Lite-Preview",
                 5:"data_Qwen3.5-Flash",6:"data_Qwen3.5-35B-A3B"}
    def sex_race_to_agent(sex, race):
        return (sex - 1) * 3 + (race - 1)
    all_data = {}
    for _, row in df.iterrows():
        llm_val = int(row['llm'])
        model_name = model_map.get(llm_val)
        if model_name is None:
            continue
        topic_val = int(row['topic']) - 1
        influence = None
        subornness = None
        if row['metric'] == 1:
            influence = row['value']
        else:
            subornness = row['value']
        sex = int(row['sex'])
        race = int(row['race'])
        agent = sex_race_to_agent(sex, race)
        all_data.setdefault(model_name, {})
        all_data[model_name].setdefault(topic_val, {})
        if row['experiment_index'] <= 0:
            all_data[model_name][topic_val].setdefault(6, {'inf': [], 'sub': []})
            if row['metric'] == 1:
                all_data[model_name][topic_val][6]['inf'].append(influence)
            else:
                all_data[model_name][topic_val][6]['sub'].append(subornness)
        else:
            all_data[model_name][topic_val].setdefault(agent, {'inf': [], 'sub': []})
            if row['metric'] == 1:
                all_data[model_name][topic_val][agent]['inf'].append(influence)
            else:
                all_data[model_name][topic_val][agent]['sub'].append(subornness)
    return all_data

def build_excel_table(all_models_data):
    models = list(all_models_data.keys())
    topics = list(range(6))
    agents = list(range(7))
    row_index = pd.MultiIndex.from_product([topics, agents], names=['topic', 'agent'])
    col_index = pd.MultiIndex.from_product([models, ['influence', 'subbornness']], names=['model', 'metric'])
    df = pd.DataFrame(index=row_index, columns=col_index)
    color_info = {}

    for model in models:
        model_data = all_models_data[model]
        for topic in topics:
            if topic not in model_data:
                for agent in agents:
                    df.loc[(topic, agent), (model, 'influence')] = "0.0000"
                    df.loc[(topic, agent), (model, 'subbornness')] = "0.0000"
                    color_info[(topic, agent, model, 'influence')] = 'black'
                    color_info[(topic, agent, model, 'subbornness')] = 'black'
                continue

            topic_data = model_data[topic]
            baseline_exists = (6 in topic_data) and len(topic_data[6]['inf']) > 0 and len(topic_data[6]['sub']) > 0

            for agent in agents:
                if agent not in topic_data or not topic_data[agent]['inf'] or not topic_data[agent]['sub']:
                    mean_inf = 0.0
                    std_inf = 0.0
                    mean_sub = 0.0
                    std_sub = 0.0
                    star_inf = ''
                    star_sub = ''
                    color_inf = 'black'
                    color_sub = 'black'
                else:
                    mean_inf = np.mean(topic_data[agent]['inf'])
                    std_inf = np.std(topic_data[agent]['inf'], ddof=1)  # 样本标准差
                    mean_sub = np.mean(topic_data[agent]['sub'])
                    std_sub = np.std(topic_data[agent]['sub'], ddof=1)

                    if agent == 6:
                        star_inf = ''; star_sub = ''
                        color_inf = 'black'; color_sub = 'black'
                    else:
                        if not baseline_exists:
                            mean_inf = 0.0; mean_sub = 0.0
                            star_inf = ''; star_sub = ''
                            color_inf = 'black'; color_sub = 'black'
                        else:
                            # --- 影响力检验 ---
                            n1_inf = len(topic_data[agent]['inf'])
                            n2_inf = len(topic_data[6]['inf'])
                            mean_base_inf = np.mean(topic_data[6]['inf'])
                            q95_inf_agent = np.percentile(topic_data[agent]['inf'], 95)
                            q3_inf_agent = np.percentile(topic_data[agent]['inf'], 75)
                            q95_inf_base = np.percentile(topic_data[6]['inf'], 95)
                            q3_inf_base = np.percentile(topic_data[6]['inf'], 75)
                            std_inf_agent = np.std(topic_data[agent]['inf'], ddof=1)  # 样本标准差
                            std_inf_base = np.std(topic_data[6]['inf'], ddof=1)
                            try:
                                _, p_inf = mannwhitneyu(topic_data[agent]['inf'], topic_data[6]['inf'], alternative='two-sided')
                            except ValueError:
                                p_inf = 1.0
                            # 打印结果
                            print(topic_data[agent]['inf'])
                            print(f"[{model}] topic{topic} agent{agent} | influence | n1={n1_inf} n2={n2_inf} | "
                                  f"mean1={mean_inf:.4f}±{std_inf_agent:.4f} mean2={mean_base_inf:.4f}±{std_inf_base:.4f} | "
                                  f"agent: q3={q3_inf_agent:.4f} q95={q95_inf_agent:.4f} | "
                                  f"base: q3={q3_inf_base:.4f} q95={q95_inf_base:.4f} | "
                                  f"p={p_inf:.4f} {'*' if p_inf < 0.05 else ''}{'*' if p_inf < 0.01 else ''}")
                            if p_inf < 0.01:
                                star_inf = '**'
                                color_inf = 'blue' if mean_inf < mean_base_inf else 'red'
                            elif p_inf < 0.05:
                                star_inf = '*'
                                color_inf = 'blue' if mean_inf < mean_base_inf else 'red'
                            else:
                                star_inf = ''
                                color_inf = 'black'

                            # --- 固执度检验 ---
                            n1_sub = len(topic_data[agent]['sub'])
                            n2_sub = len(topic_data[6]['sub'])
                            mean_base_sub = np.mean(topic_data[6]['sub'])
                            q95_sub_agent = np.percentile(topic_data[agent]['sub'], 95)
                            q3_sub_agent = np.percentile(topic_data[agent]['sub'], 75)
                            q95_sub_base = np.percentile(topic_data[6]['sub'], 95)
                            q3_sub_base = np.percentile(topic_data[6]['sub'], 75)
                            std_sub_agent = np.std(topic_data[agent]['sub'], ddof=1)
                            std_sub_base = np.std(topic_data[6]['sub'], ddof=1)
                            try:
                                _, p_sub = mannwhitneyu(topic_data[agent]['sub'], topic_data[6]['sub'], alternative='two-sided')
                            except ValueError:
                                p_sub = 1.0
                            print(topic_data[agent]['sub'])
                            print(f"[{model}] topic{topic} agent{agent} | subbornness | n1={n1_sub} n2={n2_sub} | "
                                  f"mean1={mean_sub:.4f}±{std_sub_agent:.4f} mean2={mean_base_sub:.4f}±{std_sub_base:.4f} | "
                                  f"agent: q3={q3_sub_agent:.4f} q95={q95_sub_agent:.4f} | "
                                  f"base: q3={q3_sub_base:.4f} q95={q95_sub_base:.4f} | "
                                  f"p={p_sub:.4f} {'*' if p_sub < 0.05 else ''}{'*' if p_sub < 0.01 else ''}")
                            if p_sub < 0.01:
                                star_sub = '**'
                                color_sub = 'blue' if mean_sub < mean_base_sub else 'red'
                            elif p_sub < 0.05:
                                star_sub = '*'
                                color_sub = 'blue' if mean_sub < mean_base_sub else 'red'
                            else:
                                star_sub = ''
                                color_sub = 'black'

                inf_str = f"{mean_inf:.2f}({std_inf:.2f}){star_inf}"
                sub_str = f"{mean_sub:.2f}({std_sub:.2f}){star_sub}"
                df.loc[(topic, agent), (model, 'influence')] = inf_str
                df.loc[(topic, agent), (model, 'subbornness')] = sub_str
                color_info[(topic, agent, model, 'influence')] = color_inf
                color_info[(topic, agent, model, 'subbornness')] = color_sub

    # 写入 Excel（与原代码相同）
    output_path = Path(__file__).parent / "ans.xlsx"
    workbook = xlsxwriter.Workbook(output_path)
    worksheet = workbook.add_worksheet("Results")
    formats = {
        'black': workbook.add_format({'font_color': 'black'}),
        'red': workbook.add_format({'font_color': 'red'}),
        'blue': workbook.add_format({'font_color': 'blue'}),
    }
    worksheet.write(0, 0, "topic", formats['black'])
    worksheet.write(0, 1, "agent", formats['black'])
    col = 2
    for model in models:
        worksheet.write(0, col, f"{model}\ninfluence", formats['black'])
        worksheet.write(0, col+1, f"{model}\nsubbornness", formats['black'])
        col += 2
    row = 1
    for topic in topics:
        for agent in agents:
            worksheet.write(row, 0, topic, formats['black'])
            agent_label = "baseline" if agent == 6 else str(agent)
            worksheet.write(row, 1, agent_label, formats['black'])
            col = 2
            for model in models:
                cell_value = df.loc[(topic, agent), (model, 'influence')]
                color = color_info.get((topic, agent, model, 'influence'), 'black')
                worksheet.write(row, col, str(cell_value), formats[color])
                cell_value = df.loc[(topic, agent), (model, 'subbornness')]
                color = color_info.get((topic, agent, model, 'subbornness'), 'black')
                worksheet.write(row, col+1, str(cell_value), formats[color])
                col += 2
            row += 1
    workbook.close()
    print(f"表格已保存至: {output_path}")

def main():
    all_data = extract_model_data()
    build_excel_table(all_data)

if __name__ == "__main__":
    main()