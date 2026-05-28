'''
1.  每个时间步的平均观点
'''
import get_data
model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct", 4: "Gemini-3.1-Flash-Lite-Preview",
             5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}
def sex_race_to_agent(sex, race):
    return (sex - 1) * 3 + (race - 1) + 1
def c(i,op):
    return 1.2**(-i)* op ** (1.08)
data = get_data.step()

convs = []
convs_op = []
posneg = []

grouped = data.groupby(['llm', 'topic', 'experiment_index'])
pos_influencet = [[[[[[] for _ in range(7)]for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
neg_influencet = [[[[[[] for _ in range(7)]for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
pos_influence = [[[[0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
neg_influence = [[[[0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
pos_stubborn = [[[[0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
neg_stubborn = [[[[0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
# sub: llm topic expr agent=[]
# inf: llm topic expr [source][target]=[]
for (llm, topic, exp_idx), group in grouped:
    # pos_stubborn[0][0][0].append(5)
    # print(pos_stubborn[0][0][0])
    # exit(0)
    group = group.sort_values(['debate_iter', 'sex', 'race'])
    agents = list(group[['sex', 'race']].drop_duplicates().itertuples(index=False, name=None))
    # print(agents)
    opinions_by_iter = {}
    for t in range(0, 11):  # 时间步 0~10
        sub = group[group['debate_iter'] == t]
        opinions_by_iter[t] = {(row.sex, row.race): row.opinion for row in sub.itertuples()}
    for i in range(1, 11):
        prev = opinions_by_iter[i - 1]
        curr = opinions_by_iter[i]

        for agent in agents:
            nx = sex_race_to_agent(agent[0],agent[1])
            old_op = prev.get(agent)
            new_op = curr.get(agent)
            if old_op is None or new_op is None:
                # print(llm,topic,exp_idx)
                continue
            delta = new_op - old_op
            if delta == 0:
                continue
            c_val = c(i, abs(delta))
            if delta > 0:
                pos_stubborn[llm][topic][exp_idx+5][nx] -= c_val
            else:
                neg_stubborn[llm][topic][exp_idx+5][nx] -= c_val

            # 影响力
            S = []
            for other in agents:
                if other == agent:
                    continue
                prev_op = prev.get(other)
                if prev_op is None:
                    continue
                if delta > 0:# positive
                    if prev_op > old_op:
                        S.append(other)
                else:  # negative
                    if prev_op < old_op:
                        S.append(other)
            weights = {}
            for s in S:
                if prev[s] == 0 or abs(delta) == 1:
                    weights[s] = 1
                else:
                    weights[s] = 2
            total_weight = sum(weights.values())
            if total_weight > 0:
                for s in S:
                    influence_value = c_val * weights[s] / total_weight
                    if delta > 0:
                        pos_influencet[llm][topic][exp_idx+5][sex_race_to_agent(s[0],s[1])][nx].append(influence_value)
                    else:
                        neg_influencet[llm][topic][exp_idx+5][sex_race_to_agent(s[0],s[1])][nx].append(influence_value)
def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0
for llm in range(1,7):
    for topic in range(1,7):
        for expr in range(1,36):
            for agent in range(1,7):
                for target in range(1,7):
                    pos_influence[llm][topic][expr][agent]+=safe_mean(pos_influencet[llm][topic][expr][agent][target])
                    neg_influence[llm][topic][expr][agent]+=safe_mean(neg_influencet[llm][topic][expr][agent][target])
print("down!")
import pickle

# 把四个大列表打包成一个字典
save_data = {
    'pos_influence': pos_influence,
    'neg_influence': neg_influence,
    'pos_stubborn': pos_stubborn,
    'neg_stubborn': neg_stubborn
}

with open('influence_data.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print("数据已保存到 influence_data.pkl")
'''
       sex  race  topic  llm  experiment_index  debate_iter  opinion
0        1     1      1    1                -4            0        0
1        1     2      1    1                -4            0       -1
2        1     3      1    1                -4            0        0
3        2     1      1    1                -4            0        0
4        2     2      1    1                -4            0        0
...    ...   ...    ...  ...               ...          ...      ...
73795    1     2      6    6                30           10        1
73796    1     3      6    6                30           10        0
73797    2     1      6    6                30           10        1
73798    2     2      6    6                30           10        1
73799    2     3      6    6                30           10        1
'''