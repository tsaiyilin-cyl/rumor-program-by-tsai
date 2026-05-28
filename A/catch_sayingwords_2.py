# -*- coding: utf-8 -*-
import ast
import json
from pathlib import Path
import pandas as pd


def load_topic_experiments(data_dir_name, target_topic):
    """读取指定模型数据目录下，特定 topic 的所有实验文件夹，返回 (experiment_index, event_lines) 列表"""
    logs_dir = Path(__file__).parent / data_dir_name / "logs"
    matched_dirs = []
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue
        topic = subdir.name.split('_')[0]
        if topic == target_topic:
            matched_dirs.append(subdir)
    matched_dirs.sort(key=lambda d: d.name)

    experiments = []
    for idx, subdir in enumerate(matched_dirs):
        experiment_index = idx - 4   # idx=0 -> -4, ..., idx=34 -> 30
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
        experiments.append((experiment_index, lines))
    return experiments


def parse_events(event_lines):
    """将每行字符串解析为字典，返回事件列表"""
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


def extract_speeches(events, model_name, experiment_index, target_rounds=(1, 2)):
    """
    提取指定轮次（ts=1 和 ts=2）的所有 talk/listen 发言。
    返回每条发言的：round, 内容, opinion_0, opinion_1, opinion_2（完整填写）
    """
    # 记录每个 agent 在各时间点的观点（关注 ts=0,1,2）
    opinions = {0: {}, 1: {}, 2: {}}
    for e in events:
        if e['type'] == 'reflect':
            agent = int(e['owner'].split('_')[1])
            ts = e['ts']
            if ts in (0, 1, 2):
                opinions[ts][agent] = e['item']['new_opinion']

    result = []
    for e in events:
        if e['type'] not in ('talk', 'listen'):
            continue
        ts = e['ts']
        if ts not in target_rounds:
            continue

        agent = int(e['owner'].split('_')[1])
        if e['type'] == 'listen':
            content = e['item'].get('reply', '')
        else:
            content = e['item'].get('content', '')

        opinion_0 = opinions[0].get(agent)
        opinion_1 = opinions[1].get(agent, opinion_0)   # 如果没有 ts=1 的观点，则沿用初始观点
        opinion_2 = opinions[2].get(agent, opinion_1)   # 如果没有 ts=2 的观点，则沿用 ts=1 观点

        rec = {
            'model': model_name,
            'experiment_index': experiment_index,
            'if_base': 1 if experiment_index <= 0 else 0,
            'agent': agent,
            'round': ts,
            'type': e['type'],
            '第一轮辩论内容（talk是发言,listen是倾听完别人发言的反馈）': content,
            'opinion_0': opinion_0,
            'opinion_1': opinion_1,
            'opinion_2': opinion_2,
        }
        result.append(rec)
    return result


def main():
    target_topic = "topic0"
    model_configs = [
        ("data_DeepSeek-V3.2", "DeepSeek-V3.2", [-1, 0]),
        ("data_GPT-5.1", "GPT-5.1", [1, 0])
    ]
    all_data = []

    for data_dir, model_name, target_initial_ops in model_configs:
        print(f"处理 {model_name} 数据...")
        experiments = load_topic_experiments(data_dir, target_topic)
        print(f"  找到 {len(experiments)} 个实验文件夹")

        for exp_idx, event_lines in experiments:
            events = parse_events(event_lines)
            speeches = extract_speeches(events, model_name, exp_idx)
            # 根据初始观点过滤（opinion_0 必须在目标列表中）
            filtered = [s for s in speeches if s['opinion_0'] in target_initial_ops]
            all_data.extend(filtered)

    if not all_data:
        print("没有找到符合条件的发言。")
        return

    df = pd.DataFrame(all_data)
    df = df.sort_values(['model', 'experiment_index', 'agent', 'round'])

    # 调整列顺序
    cols = ['model', 'experiment_index', 'if_base', 'agent', 'round', 'type',
            '第一轮辩论内容（talk是发言,listen是倾听完别人发言的反馈）',
            'opinion_0', 'opinion_1', 'opinion_2']
    df = df[cols]

    output_path = Path(__file__).parent / "extracted_speeches.xlsx"
    df.to_excel(output_path, index=False)
    print(f"提取完成，共 {len(df)} 条记录，已保存至 {output_path}")

    # 简要统计
    print("\n=== 数据概览 ===")
    for model in df['model'].unique():
        print(f"\n{model}:")
        for r in [1,2]:
            cnt = len(df[(df['model']==model) & (df['round']==r)])
            print(f"  第{r}轮发言数: {cnt}")


if __name__ == "__main__":
    main()