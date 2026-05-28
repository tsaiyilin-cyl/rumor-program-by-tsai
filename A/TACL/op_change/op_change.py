import pandas as pd
import numpy as np


def sex_race_to_agent(sex, race):
    return (sex - 1) * 3 + (race - 1) + 1

model_map = {
    1: "DeepSeek-V3.2",
    2: "GPT-5.1",
    3: "Llama-3.3-70b-instruct",
    4: "Gemini-3.1-Flash-Lite-Preview",
    5: "Qwen3.5-Flash",
    6: "Qwen3.5-35B-A3B"
}

df = pd.read_csv('catch_opinion_6all.csv')
df['agent'] = df.apply(lambda row: sex_race_to_agent(row['sex'], row['race']), axis=1)
df['if_base'] = (df['experiment_index'] <= 0).astype(int)
df['llm'] = df['llm'].map(model_map)

# 筛选初始观点为0的Agent
init_zero = df[(df['debate_iter'] == 0) & (df['opinion'] == 0)]
groups = init_zero.groupby(['topic', 'llm', 'if_base', 'experiment_index', 'agent'])

result_rows = []

for (topic, llm, if_base, exp_idx, agent), _ in groups:
    # 获取该Agent在该实验中的所有数据
    agent_data = df[(df['topic'] == topic) &
                    (df['llm'] == llm) &
                    (df['experiment_index'] == exp_idx) &
                    (df['agent'] == agent)].sort_values('debate_iter')

    # 找到最大辩论轮次（应该是0到某个最大值）
    if agent_data.empty:
        continue
    max_round = agent_data['debate_iter'].max()

    # 提取从第1轮到max_round的观点，要求连续存在
    opinions = []
    missing = False
    for r in range(1, max_round + 1):
        row = agent_data[agent_data['debate_iter'] == r]
        if len(row) == 0:
            missing = True
            break
        opinions.append(row['opinion'].iloc[0])

    if missing or len(opinions) == 0:
        variance = np.nan
        flip_count = np.nan
    else:
        full_seq = [0] + opinions  # 长度 = max_round + 1
        variance = np.var(full_seq, ddof=0)
        flip_count = sum(1 for i in range(max_round) if full_seq[i] != full_seq[i + 1])

    result_rows.append({
        'LLM': llm,
        'topic': topic,
        'if_base': if_base,
        'experiment_index': exp_idx,
        'agent': agent,
        '本次实验最大伦次': max_round,
        '方差': variance,
        '观点变化次数': flip_count
    })

result_df = pd.DataFrame(result_rows)
# 可移除 max_round_used 列，若不需要
# result_df = result_df.drop(columns=['max_round_used'])
result_df.to_csv('op_change.csv', index=False)

print("计算完成，结果已保存至op_change.csv")