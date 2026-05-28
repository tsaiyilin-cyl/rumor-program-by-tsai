import pandas as pd
import pickle

# -------------------- 1. 读取原始数据 --------------------
df = pd.read_csv('convergence_results.csv')

# -------------------- 2. 加载 influence_data.pkl --------------------
with open('influence_data.pkl', 'rb') as f:
    loaded = pickle.load(f)

pos_influence = loaded['pos_influence']  # 维度: [7][7][36][7] (1-based)
neg_influence = loaded['neg_influence']
pos_stubborn = loaded['pos_stubborn']
neg_stubborn = loaded['neg_stubborn']

# -------------------- 3. 模型名称 -> 索引 --------------------
name_to_idx = {
    "DeepSeek-V3.2": 1,
    "GPT-5.1": 2,
    "Llama-3.3-70b-instruct": 3,
    "Gemini-3.1-Flash-Lite-Preview": 4,
    "Qwen3.5-Flash": 5,
    "Qwen3.5-35B-A3B": 6
}

# -------------------- 4. 定义 rumor & truth 映射表 --------------------
# 负实验：exp = -4, -3, -2, -1, 0
neg_map_type1 = [[1, 5], [5, 0], [5, 1], [5, 2], [5, 3]]  # 前4个模型
neg_map_type2 = [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4]]  # 后2个模型

# 正实验：exp = 1 .. 30
pos_map = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 3], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 2], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3], [4, 5],
    [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]
]

# -------------------- 5. 为每一行提取 24 个指标 + rumor/truth --------------------
agent_cols = [f'agent{i}_{stat}' for i in range(1, 7) for stat in ['posi', 'negi', 'poss', 'negs']]

new_metrics = []  # 存放 24 个指标
rumor_truth = []  # 存放 (rumor, truth)

for _, row in df.iterrows():
    llm_name = row['LLM']
    topic = row['topic']
    experiment = row['experiment']

    # ----- 5.1 获取 influence 指标 -----
    llm_idx = name_to_idx[llm_name]
    topic_idx = int(topic)
    exp_idx = int(experiment) + 5  # 映射到 1..35

    row_metrics = []
    for agent in range(1, 7):
        posi = pos_influence[llm_idx][topic_idx][exp_idx][agent]
        negi = neg_influence[llm_idx][topic_idx][exp_idx][agent]
        poss = pos_stubborn[llm_idx][topic_idx][exp_idx][agent]
        negs = neg_stubborn[llm_idx][topic_idx][exp_idx][agent]
        row_metrics.extend([posi, negi, poss, negs])
    new_metrics.append(row_metrics)

    # ----- 5.2 计算 rumor 和 truth -----
    exp_val = int(experiment)
    if exp_val <= 0:  # -4 .. 0
        # 确定使用哪个负映射表
        if llm_name in {"DeepSeek-V3.2", "GPT-5.1", "Llama-3.3-70b-instruct", "Gemini-3.1-Flash-Lite-Preview"}:
            neg_map = neg_map_type1
        else:  # data_Qwen3.5-Flash 或 data_Qwen3.5-35B-A3B
            neg_map = neg_map_type2
        # exp_val = -4 -> index 0, -3 -> 1, ..., 0 -> 4
        idx = exp_val + 4  # 因为 -4+4=0, 0+4=4
        rumor, truth = neg_map[idx]
    else:  # 1 .. 30
        idx = exp_val - 1  # exp=1 -> index 0
        rumor, truth = pos_map[idx]

    rumor_truth.append((rumor+1, truth+1))

# 创建新列
df_new = df.copy()
df_new[agent_cols] = new_metrics
df_new['rumor'] = [rt[0] for rt in rumor_truth]
df_new['truth'] = [rt[1] for rt in rumor_truth]

# -------------------- 6. 添加 8 个聚合统计列 --------------------
# 分别提取 6 个 agent 的 posi, negi, poss, negs
posi_cols = [f'agent{i}_posi' for i in range(1, 7)]
negi_cols = [f'agent{i}_negi' for i in range(1, 7)]
poss_cols = [f'agent{i}_poss' for i in range(1, 7)]
negs_cols = [f'agent{i}_negs' for i in range(1, 7)]

# 计算平均值和最大值
df_new['ave_Ip'] = df_new[posi_cols].mean(axis=1)
df_new['ave_In'] = df_new[negi_cols].mean(axis=1)
df_new['ave_Sp'] = df_new[poss_cols].mean(axis=1)
df_new['ave_Sn'] = df_new[negs_cols].mean(axis=1)

df_new['max_Ip'] = df_new[posi_cols].max(axis=1)
df_new['max_In'] = df_new[negi_cols].max(axis=1)
df_new['max_Sp'] = df_new[poss_cols].max(axis=1)
df_new['max_Sn'] = df_new[negs_cols].max(axis=1)

# -------------------- 7. 保存为 Excel 文件 --------------------
df_new.to_excel('last_op_34.xlsx', index=False)
print("新文件已保存: last_op_34.xlsx")