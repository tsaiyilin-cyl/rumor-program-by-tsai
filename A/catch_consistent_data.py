# -*- coding: utf-8 -*-
import ast
from pathlib import Path
import pandas as pd


def step(data_dir, model_name):
    """
    从指定模型的日志目录中提取 talk 事件数据。

    参数:
        data_dir (str): 日志根目录下的模型数据文件夹名（如 "data_DeepSeek-V3.2"）
        model_name (str): 模型显示名称（如 "DeepSeek-V3.2"）

    返回:
        list: 每条 talk 记录为一个列表，顺序为 [model, topic, agent, ts, expr, content, opinion]
    """
    logs_dir = Path(__file__).parent / data_dir / "logs"
    topic_events = {}

    # 遍历 logs 下的所有子文件夹（每个 topic 对应一个子文件夹）
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue

        topic = subdir.name.split('_')[0]  # 例如 "topic0"
        event_file = subdir / "event.txt"
        if not event_file.exists():
            print(f"警告: {event_file} 不存在，跳过")
            continue

        # 尝试不同编码读取文件
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

        if topic not in topic_events:
            topic_events[topic] = []
        topic_events[topic].extend(lines)

    all_records = []

    # 处理每个 topic（共6个）
    for i in range(6):
        expr = -5  # 每个 topic 独立计数
        ini = [[None for _ in range(6)] for _ in range(11)]  # 存储每个 agent 在每个时间步后的观点
        topic_name = f"topic{i}"
        print(f"\n处理 {topic_name}")

        for item_str in topic_events.get(topic_name, []):
            try:
                item = ast.literal_eval(item_str)
                agent = int(item['owner'].split("_")[1])
                ts = int(item['ts'])

                if item['type'] == 'talk':
                    content = item['item']['content']
                    # 观点取该时间步开始前的值（即上一时间步结束后的观点）
                    opinion = ini[ts - 1][agent] if ts > 0 else None
                    all_records.append([
                        model_name,  # model
                        i + 1,  # topic（1~6）
                        agent,  # agent
                        ts,  # ts
                        expr,  # expr
                        content,  # content
                        opinion  # opinion
                    ])

                elif item['type'] == 'reflect':
                    new_opinion = int(item['item']['new_opinion'])
                    ini[ts][agent] = new_opinion
                    if ts == 0 and agent == 0:
                        expr += 1

            except Exception as err:
                print(f"解析失败: {item_str}\n错误: {err}")

    return all_records


def main():
    # 主题列表（顺序对应 topic 1~6）
    topics = [
        "各类秘密组织或精英人士对全球重大事件、经济体及政府施加着显著且常具恶意的控制。",
        "癌症、艾滋病和许多其他疾病的治疗方法其实早就被发现了，但政府和大型制药公司隐瞒或封锁了这些信息，因为一旦公布就会让他们破产。他们既贪婪又不关心民众。",
        "微波炉加热食物会致癌。",
        "WiFi辐射伤害身体。",
        "星座决定性格和人品。",
        "大脑只被用了 10%，很多地方没被开发。"
    ]

    # 模型数据文件夹名（与 logs 父目录名对应）
    data_dirs = [
        "data_DeepSeek-V3.2",
        "data_GPT-5.1",
        "data_Llama-3.3-70b-instruct",
        "data_Gemini-3.1-Flash-Lite-Preview",
        # "data_Qwen3.5-Flash",
        # "data_Qwen3.5-35B-A3B"
    ]
    # 对应的显示名称
    model_names = [
        # "DeepSeek-V3.2",
        # "GPT-5.1",
        "Llama-3.3-70b-instruct",
        # "Gemini-3.1-Flash-Lite-Preview",
        # "Qwen3.5-Flash",
        # "Qwen3.5-35B-A3B"
    ]

    all_data = []
    for data_dir, model_name in zip(data_dirs, model_names):
        print(f"\n开始处理 {model_name}，数据目录：{data_dir}")
        records = step(data_dir, model_name)   # step 返回的每个记录是 [model_name, topic, agent, ts, expr, content, opinion]
        all_data.extend(records)

    # 创建 DataFrame
    df = pd.DataFrame(all_data, columns=['model', 'topic', 'agent', 'ts', 'expr', 'content', 'opinion'])

    # 根据 topic 序号映射主题文本
    df['topic_text'] = df['topic'].apply(lambda t: topics[t-1] if 1 <= t <= len(topics) else None)

    column_order = ['model', 'topic', 'topic_text', 'agent', 'ts', 'expr', 'content', 'opinion']
    df = df[column_order]

    df.to_excel('consistent_llama.xlsx', index=False, engine='openpyxl')
    print("\n已保存至 consistent_llama.xlsx")


if __name__ == "__main__":
    main()