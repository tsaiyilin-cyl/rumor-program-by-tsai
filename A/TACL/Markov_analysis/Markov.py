import pandas as pd
import numpy as np


def sex_race_to_agent(sex, race):
    """将性别和种族映射为agent编号（1~6）"""
    return (sex - 1) * 3 + (race - 1) + 1

model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct", 4: "Gemini-3.1-Flash-Lite-Preview",
             5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}
# 读取数据
df = pd.read_csv('catch_opinion_6all.csv')

# 添加agent编号和基线标识
df['agent'] = df.apply(lambda row: sex_race_to_agent(row['sex'], row['race']), axis=1)
df['if_base'] = (df['experiment_index'] <= 0).astype(int)
df['llm'] = df['llm'].map(model_map)
# 分组依据：议题、LLM类型、是否基线、实验试次
groups = df.groupby(['topic', 'llm', 'if_base', 'experiment_index'])

all_rows = []

for (topic, llm, if_base, exp_idx), group in groups:
    # 按辩论轮次排序
    group = group.sort_values('debate_iter')
    iters = sorted(group['debate_iter'].unique())

    # 构建观点矩阵：行=轮次，列=agent，值=opinion
    pivot = group.pivot(index='debate_iter', columns='agent', values='opinion')
    # 确保列包含所有agent 1~6
    for a in range(1, 7):
        if a not in pivot.columns:
            pivot[a] = np.nan
    pivot = pivot.reindex(columns=range(1, 7))

    # 对每个存在的轮次生成输出行
    for t in iters:
        # 获取当前轮次各agent的观点
        if t in pivot.index:
            opinions_t = pivot.loc[t].values  # 长度6
        else:
            opinions_t = [np.nan] * 6

        # 计算转移概率（仅当t>=1且前一回合存在时）
        if t >= 1 and (t - 1) in pivot.index:
            opinions_prev = pivot.loc[t - 1].values
            # 统计转移计数（3x3矩阵，索引0:-1, 1:0, 2:1）
            counts = np.zeros((3, 3))
            for p, c in zip(opinions_prev, opinions_t):
                if pd.isna(p) or pd.isna(c):
                    continue
                i = p + 1  # -1->0, 0->1, 1->2
                j = c + 1
                counts[i, j] += 1
            # 计算概率（若某源状态无样本则全NaN）
            trans = np.full((3, 3), np.nan)
            for i in range(3):
                total = counts[i].sum()
                if total > 0:
                    trans[i, :] = counts[i, :] / total
            trans_flat = trans.flatten()
        else:
            trans_flat = [np.nan] * 9  # 第0轮或前一回合缺失

        # 构建输出行
        row = {
            'LLM': llm,
            'topic': topic,
            'if_base': if_base,
            'experiment_index': exp_idx,
            'debate_iter': t
        }
        # 添加agent观点
        for a in range(1, 7):
            row[f'agent{a}'] = opinions_t[a - 1]
        # 添加9个转移概率
        trans_names = [
            'trans_-1_to_-1', 'trans_-1_to_0', 'trans_-1_to_1',
            'trans_0_to_-1', 'trans_0_to_0', 'trans_0_to_1',
            'trans_1_to_-1', 'trans_1_to_0', 'trans_1_to_1'
        ]
        for name, val in zip(trans_names, trans_flat):
            row[name] = val

        all_rows.append(row)

# 生成最终DataFrame并保存
result_df = pd.DataFrame(all_rows)
result_df.to_csv('Markov.csv', index=False)

print("计算完成，结果已保存至Markov.csv")