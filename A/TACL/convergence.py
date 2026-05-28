'''
1.  计算每次实验的收敛速度（达到稳态需要的轮数）
2.	计算每次实验是（1）达成共识（2）极化（3）碎片化
3.	计算达成共识的情况下，最终观点是1或者-1的比例
'''
import get_data
model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct", 4: "Gemini-3.1-Flash-Lite-Preview",
             5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}
def sex_race_to_agent(sex, race):
    return (sex - 1) * 3 + (race - 1)

data = get_data.step()

convs = []
convs_op = []
posneg = []

grouped = data.groupby(['llm', 'topic', 'experiment_index'])

for (llm, topic, exp_idx), group in grouped:
    max_iter = group['debate_iter'].max()
    final_round = group[group['debate_iter'] == max_iter]
    ave = final_round['opinion'].mean()

    group_sorted = group.sort_values('debate_iter')
    agents = group_sorted[['sex', 'race']].drop_duplicates()
    steady_round = 0
    for t in range(0, max_iter + 1):
        stable = True
        for _, agent in agents.iterrows():
            sex = agent['sex']
            race = agent['race']
            # 获取该agent在轮次>=t且<=max_iter的opinion
            agent_data = group_sorted[
                (group_sorted['sex'] == sex) & (group_sorted['race'] == race) & (group_sorted['debate_iter'] >= t)]
            opinions = agent_data['opinion'].values
            if len(set(opinions)) != 1:
                stable = False
                break
        if stable:
            break
        steady_round += 1
    # print(final_round['opinion'],final_round['opinion'].mean())
    opinions = final_round['opinion'].unique()
    if len(opinions) == 1:
        convergence_round = max_iter
    else:
        convergence_round = 11

    if_base = 0 if exp_idx > 0 else 1
    llm_name = model_map.get(llm, f"Unknown({llm})")
    agree_state = "None"
    if (convergence_round !=11):
        agree_state = "agreement"
    else:
        if(len(opinions)==3):
            agree_state = "-1_0_1"
        else:
            if(-1 in opinions and 0 in opinions):
                agree_state = "-1_0"
            elif ( -1 in opinions and 1 in opinions):
                agree_state = "polar"
            else: agree_state = "0_1"

    if agree_state == "agreement":
        agree_state = "consistent"
        agree_op = opinions[0]
    else:
        agree_op = "False"
    convs.append({
        "LLM": llm_name,
        "topic": topic,
        "experiment": exp_idx,
        "if_base": if_base,
        "稳态轮次": steady_round,
        "达成共识轮次": convergence_round,
        "共识观点": agree_op,
        "末轮观点分布": agree_state,
        "末轮平均观点": ave
    })
import pandas as pd
convs_df = pd.DataFrame(convs)
convs_df.to_csv("convergence_results.csv", index=False, encoding="utf-8-sig")
# convs_op_df = pd.DataFrame(convs_op)
# convs_op_df.to_csv("conv_op_results.csv", index=False, encoding="utf-8-sig")
# posneg_df = pd.DataFrame(posneg)
# posneg_df.to_csv("posorneg.csv", index=False, encoding="utf-8-sig")
print("down!")
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