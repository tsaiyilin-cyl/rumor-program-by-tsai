'''
1.  每个时间步的平均观点
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
    avg_by_iter = {}
    group_sorted = group.sort_values('debate_iter')
    agents = group_sorted[['sex', 'race']].drop_duplicates()
    for t in range(0, max_iter + 1):
        agent_data = group_sorted[(group_sorted['debate_iter'] == t)]
        # print(agent_data)
        opinions = agent_data['opinion'].values
        avg_val = opinions.mean() if len(opinions) > 0 else None
        avg_by_iter[t] = avg_val

        if_base = 0 if exp_idx > 0 else 1
        llm_name = model_map.get(llm, f"Unknown({llm})")

        convs.append({
            "LLM": llm_name,
            "topic": topic,
            "experiment": exp_idx,
            "if_base": if_base,
            "时间步": t,
            "平均观点": opinions.mean()
        })

    if max_iter < 10:
        #print(avg_by_iter)
        last_avg = avg_by_iter[max_iter]  # 最后一轮的平均观点
        for t in range(max_iter + 1, 11):
            if_base = 0 if exp_idx > 0 else 1
            llm_name = model_map.get(llm, f"Unknown({llm})")
            convs.append({
                "LLM": llm_name,
                "topic": topic,
                "experiment": exp_idx,
                "if_base": if_base,
                "时间步": t,
                "平均观点": last_avg
            })
import pandas as pd
convs_df = pd.DataFrame(convs)
convs_df.to_csv("time_op_ave2.csv", index=False, encoding="utf-8-sig")

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