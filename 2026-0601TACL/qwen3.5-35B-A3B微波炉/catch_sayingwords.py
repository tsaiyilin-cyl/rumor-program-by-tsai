# -*- coding: utf-8 -*-
import ast
import json
from pathlib import Path
import pandas as pd

def load_topic_experiments(data_dir_name, target_topic):
    logs_dir = Path(__file__).parent / data_dir_name / "logs"
    matched_dirs = []
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue
        folder_name = subdir.name
        topic = folder_name.split('_')[0]
        if topic == target_topic:
            matched_dirs.append(subdir)
    matched_dirs.sort(key=lambda d: d.name)
    experiments = []
    for idx, subdir in enumerate(matched_dirs):
        experiment_id = idx + 1
        is_baseline = (idx < 5)
        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue
        lines = None
        for enc in ['utf-8', 'gbk']:
            try:
                with open(event_file, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                pass
        if lines is None:
            print(f"警告: {event_file} 编码无法识别，跳过")
            continue
        experiments.append((experiment_id, is_baseline, lines))
    return experiments

def parse_events(event_lines):
    events = []
    for line in event_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = ast.literal_eval(line)
        except:
            event = json.loads(line)
        events.append(event)
    return events

def get_opinions_at_ts(events, ts):
    opinions = {}
    for e in events:
        if e.get('type') == 'reflect' and e.get('ts') == ts:
            agent = int(e['owner'].split('_')[1])
            opinions[agent] = e['item']['new_opinion']
    return opinions

def compute_influence_for_agent(agent_target, target_opinion, opinions0, opinions1):
    """
    计算所有其他智能体（不包括agent_target）的观点变化中，
    有利于 target_opinion (1 或 -1) 的总影响力。
    """
    total = 0.0
    for agent, op0 in opinions0.items():
        if agent == agent_target:
            continue
        op1 = opinions1.get(agent, op0)
        change = abs(op1 - op0)
        if change == 0:
            continue
        # 判断变化是否有利于目标
        if target_opinion == 1:
            favorable = (op0 == 0 and op1 == 1) or (op0 == -1 and op1 == 0) or (op0 == -1 and op1 == 1)
        else:  # target_opinion == -1
            favorable = (op0 == 0 and op1 == -1) or (op0 == 1 and op1 == 0) or (op0 == 1 and op1 == -1)
        if favorable:
            total += (1.2 ** (-1)) * (change ** 1.08)
    return total

def extract_round1_talk(events, experiment_id, is_baseline, model_name):
    opinions_ts0 = get_opinions_at_ts(events, 0)
    opinions_ts1 = get_opinions_at_ts(events, 1)

    # 找出初始观点为1和-1的智能体
    target_pos1 = None
    target_neg1 = None
    for agent, op in opinions_ts0.items():
        if op == 1:
            target_pos1 = agent
        elif op == -1:
            target_neg1 = agent

    if target_pos1 is None or target_neg1 is None:
        print(f"警告: 实验 {experiment_id} 缺少初始观点1或-1的智能体，跳过")
        return []

    result = []
    for e in events:
        if e.get('type') != 'talk' or e.get('ts') != 1:
            continue
        agent = int(e['owner'].split('_')[1])
        content = e['item'].get('content', '')
        opinion0 = opinions_ts0.get(agent)
        if opinion0 not in (1, -1):
            continue

        # 计算该发言者的影响力：所有其他智能体变化中有利于其目标的总和
        influence = compute_influence_for_agent(agent, opinion0, opinions_ts0, opinions_ts1)

        result.append({
            'model': model_name,
            'experiment_id': experiment_id,
            'is_baseline': is_baseline,
            '发言文本': content,
            '观点': opinion0,
            'truth': target_pos1,
            'rumor': target_neg1,
            'current_agent': agent,
            'influence': influence,
        })
    return result

def main():
    target_topic = "topic2"
    model_configs = [("data_Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B")]

    all_data = []
    for data_dir, model_name in model_configs:
        print(f"处理 {model_name} 数据...")
        experiments = load_topic_experiments(data_dir, target_topic)
        print(f"  找到 {len(experiments)} 个实验文件夹")

        for exp_id, is_base, event_lines in experiments:
            events = parse_events(event_lines)
            speeches = extract_round1_talk(events, exp_id, is_base, model_name)
            all_data.extend(speeches)

    if not all_data:
        print("没有找到符合条件的发言。")
        return

    df = pd.DataFrame(all_data)
    df = df.sort_values(['experiment_id', 'current_agent'])

    output_path = Path(__file__).parent / "extracted_speeches.xlsx"
    df.to_excel(output_path, index=False)
    print(f"提取完成，共 {len(df)} 条记录，已保存至 {output_path}")

if __name__ == "__main__":
    main()