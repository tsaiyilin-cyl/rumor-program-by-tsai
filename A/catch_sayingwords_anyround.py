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

def extract_round_x_talk(events, experiment_id, is_baseline, model_name, round_x):
    """
    提取第 round_x 轮观点为 ±1 的智能体在第 round_x+1 轮的发言，
    并基于第 round_x 轮与第 round_x+1 轮的观点变化计算影响力。
    """
    opinions_ts_x = get_opinions_at_ts(events, round_x)
    opinions_ts_x1 = get_opinions_at_ts(events, round_x + 1)

    # 找出第 round_x 轮观点为 1 和 -1 的智能体（各取第一个遇到的）
    target_pos1 = None
    target_neg1 = None
    for agent, op in opinions_ts_x.items():
        if op == 1 and target_pos1 is None:
            target_pos1 = agent
        elif op == -1 and target_neg1 is None:
            target_neg1 = agent
        if target_pos1 is not None and target_neg1 is not None:
            break

    if target_pos1 is None or target_neg1 is None:
        print(f"警告: 实验 {experiment_id} 在第 {round_x} 轮缺少观点1或-1的智能体，跳过")
        return []

    result = []
    for e in events:
        if e.get('type') != 'talk' or e.get('ts') != round_x + 1:
            continue
        agent = int(e['owner'].split('_')[1])
        content = e['item'].get('content', '')
        opinion0 = opinions_ts_x.get(agent)
        if opinion0 not in (1, -1):
            continue

        influence = compute_influence_for_agent(agent, opinion0, opinions_ts_x, opinions_ts_x1)

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
    target_topic = "topic4"
    # 可自行修改此轮次：x = 0 表示初始轮，x = 1 表示第一轮后... 以此类推
    target_round = 1   # 例如提取第0轮观点为±1的agent在第1轮的发言
    model_configs = [("data_Qwen3.5-Flash", "Qwen3.5-Flash")]

    all_data = []
    for data_dir, model_name in model_configs:
        print(f"处理 {model_name} 数据...")
        experiments = load_topic_experiments(data_dir, target_topic)
        print(f"  找到 {len(experiments)} 个实验文件夹")

        for exp_id, is_base, event_lines in experiments:
            # print(exp_id)
            events = parse_events(event_lines)
            speeches = extract_round_x_talk(events, exp_id, is_base, model_name, target_round)
            all_data.extend(speeches)

    if not all_data:
        print("没有找到符合条件的发言。")
        return

    df = pd.DataFrame(all_data)
    df = df.sort_values(['experiment_id', 'current_agent'])

    # 输出文件名中加入轮次信息，避免覆盖不同轮次的结果
    output_path = Path(__file__).parent / f"speeches_Qwen3.5-Flash-星座_round{target_round}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"提取完成（第{target_round}轮观点→第{target_round+1}轮发言），共 {len(df)} 条记录，已保存至 {output_path}")

if __name__ == "__main__":
    main()