import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import warnings
import os
warnings.filterwarnings('ignore')

output_dir = "heatmaps"
os.makedirs(output_dir, exist_ok=True)
# ----------------------------- 1. 加载原始数据和收敛结果（只需一次） -----------------------------
def load_original_data():
    """返回原始观点数据 DataFrame"""
    import get_data
    return get_data.step()


def compute_convergence_results(data_df):
    """计算收敛性结果，返回 DataFrame（与 convergence.py 逻辑一致）"""
    model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct",
                 4: "Gemini-3.1-Flash-Lite-Preview", 5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}

    convs = []
    grouped = data_df.groupby(['llm', 'topic', 'experiment_index'])

    for (llm, topic, exp_idx), group in grouped:
        max_iter = group['debate_iter'].max()
        final_round = group[group['debate_iter'] == max_iter]
        ave = final_round['opinion'].mean()

        group_sorted = group.sort_values('debate_iter')
        agents = group_sorted[['sex', 'race']].drop_duplicates()

        # 计算稳态轮次
        steady_round = 0
        for t in range(0, max_iter + 1):
            stable = True
            for _, agent in agents.iterrows():
                sex = agent['sex']
                race = agent['race']
                agent_data = group_sorted[(group_sorted['sex'] == sex) &
                                          (group_sorted['race'] == race) &
                                          (group_sorted['debate_iter'] >= t)]
                opinions = agent_data['opinion'].values
                if len(set(opinions)) != 1:
                    stable = False
                    break
            if stable:
                break
            steady_round += 1

        opinions = final_round['opinion'].unique()
        if len(opinions) == 1:
            convergence_round = max_iter
        else:
            convergence_round = 11

        if_base = 0 if exp_idx > 0 else 1
        llm_name = model_map.get(llm, f"Unknown({llm})")

        agree_state = "None"
        if convergence_round != 11:
            agree_state = "consistent"
        else:
            if len(opinions) == 3:
                agree_state = "-1_0_1"
            else:
                if -1 in opinions and 0 in opinions:
                    agree_state = "-1_0"
                elif -1 in opinions and 1 in opinions:
                    agree_state = "polar"
                else:
                    agree_state = "0_1"

        if agree_state == "consistent":
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

    return pd.DataFrame(convs)


# ----------------------------- 2. 超参数合法性判断 -----------------------------
def is_valid_hyperparam(id_val, yd_val):
    """判断超参数(id, yd)是否满足 c12 > c11_c21 > c22 > c11_c31"""

    def c(i, op):
        return id_val ** (-i) * (op ** yd_val)

    c12 = c(1, 2)
    c11_c21 = c(1, 1) + c(2, 1)
    c22 = c(2, 2)
    c11_c31 = c(1, 1) + c(3, 1)

    return c12 > c11_c21 > c22 > c11_c31


# ----------------------------- 3. 影响力计算（封装，不写磁盘） -----------------------------
def compute_influence(data_df, id_val, yd_val):
    """
    根据超参数计算 influence 和 stubbornness 矩阵
    返回: (pos_influence, neg_influence, pos_stubborn, neg_stubborn)
    每个都是 7x7x36x7 的 list (1-based索引)
    """

    def c(i, op):
        return id_val ** (-i) * (op ** yd_val)

    def sex_race_to_agent(sex, race):
        return (sex - 1) * 3 + (race - 1) + 1

    # 初始化存储结构
    pos_influence = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    neg_influence = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    pos_stubborn = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    neg_stubborn = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]

    pos_influencet = [[[[[[] for _ in range(7)] for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in
                      range(7)]
    neg_influencet = [[[[[[] for _ in range(7)] for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in
                      range(7)]

    grouped = data_df.groupby(['llm', 'topic', 'experiment_index'])

    for (llm, topic, exp_idx), group in grouped:
        group = group.sort_values(['debate_iter', 'sex', 'race'])
        agents = list(group[['sex', 'race']].drop_duplicates().itertuples(index=False, name=None))

        # 按时间步组织观点
        opinions_by_iter = {}
        for t in range(0, 11):
            sub = group[group['debate_iter'] == t]
            opinions_by_iter[t] = {(row.sex, row.race): row.opinion for row in sub.itertuples()}

        for i in range(1, 11):
            prev = opinions_by_iter[i - 1]
            curr = opinions_by_iter[i]

            for agent in agents:
                nx = sex_race_to_agent(agent[0], agent[1])
                old_op = prev.get(agent)
                new_op = curr.get(agent)
                if old_op is None or new_op is None:
                    continue
                delta = new_op - old_op
                if delta == 0:
                    continue

                c_val = c(i, abs(delta))
                # stubbornness 更新（注意：减 c_val 使负值越大表示越不固执）
                if delta > 0:
                    pos_stubborn[llm][topic][exp_idx + 5][nx] -= c_val
                else:
                    neg_stubborn[llm][topic][exp_idx + 5][nx] -= c_val

                # 找出影响的来源
                S = []
                for other in agents:
                    if other == agent:
                        continue
                    prev_op = prev.get(other)
                    if prev_op is None:
                        continue
                    if delta > 0 and prev_op > old_op:
                        S.append(other)
                    elif delta < 0 and prev_op < old_op:
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
                        influence_val = c_val * weights[s] / total_weight
                        s_agent = sex_race_to_agent(s[0], s[1])
                        if delta > 0:
                            pos_influencet[llm][topic][exp_idx + 5][s_agent][nx].append(influence_val)
                        else:
                            neg_influencet[llm][topic][exp_idx + 5][s_agent][nx].append(influence_val)

    # 求平均
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    for llm in range(1, 7):
        for topic in range(1, 7):
            for expr in range(1, 36):
                for source in range(1, 7):
                    for target in range(1, 7):
                        pos_influence[llm][topic][expr][source] += safe_mean(
                            pos_influencet[llm][topic][expr][source][target])
                        neg_influence[llm][topic][expr][source] += safe_mean(
                            neg_influencet[llm][topic][expr][source][target])

    return pos_influence, neg_influence, pos_stubborn, neg_stubborn


# ----------------------------- 4. 组装最终数据（类似 last_op26.py） -----------------------------
def assemble_final_data(conv_df, pos_influence, neg_influence, pos_stubborn, neg_stubborn):
    """
    返回完整的 DataFrame，包含原始收敛结果 + 24个指标 + rumor/truth + 8个聚合统计
    """
    name_to_idx = {
        "DeepSeek-V3.2": 1,
        "GPT-5.1": 2,
        "Llama-3.3-70b-instruct": 3,
        "Gemini-3.1-Flash-Lite-Preview": 4,
        "Qwen3.5-Flash": 5,
        "Qwen3.5-35B-A3B": 6
    }

    # 映射表（0-index）
    neg_map_type1 = [[1, 5], [5, 0], [5, 1], [5, 2], [5, 3]]
    neg_map_type2 = [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4]]
    pos_map = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 0], [2, 1], [2, 3], [2, 4], [2, 5],
        [3, 0], [3, 1], [3, 2], [3, 4], [3, 5],
        [4, 0], [4, 1], [4, 2], [4, 3], [4, 5],
        [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]
    ]

    agent_cols = [f'agent{i}_{stat}' for i in range(1, 7) for stat in ['posi', 'negi', 'poss', 'negs']]
    new_metrics = []
    rumor_truth = []

    for _, row in conv_df.iterrows():
        llm_name = row['LLM']
        topic = row['topic']
        experiment = row['experiment']

        llm_idx = name_to_idx[llm_name]
        topic_idx = int(topic)
        exp_idx = int(experiment) + 5

        row_metrics = []
        for agent in range(1, 7):
            posi = pos_influence[llm_idx][topic_idx][exp_idx][agent]
            negi = neg_influence[llm_idx][topic_idx][exp_idx][agent]
            poss = pos_stubborn[llm_idx][topic_idx][exp_idx][agent]
            negs = neg_stubborn[llm_idx][topic_idx][exp_idx][agent]
            row_metrics.extend([posi, negi, poss, negs])
        new_metrics.append(row_metrics)

        exp_val = int(experiment)
        if exp_val <= 0:
            if llm_name in {"DeepSeek-V3.2", "GPT-5.1", "Llama-3.3-70b-instruct", "Gemini-3.1-Flash-Lite-Preview"}:
                neg_map = neg_map_type1
            else:
                neg_map = neg_map_type2
            idx = exp_val + 4
            rumor, truth = neg_map[idx]
        else:
            idx = exp_val - 1
            rumor, truth = pos_map[idx]
        rumor_truth.append((rumor + 1, truth + 1))

    df_new = conv_df.copy()
    df_new[agent_cols] = new_metrics
    df_new['rumor'] = [rt[0] for rt in rumor_truth]
    df_new['truth'] = [rt[1] for rt in rumor_truth]

    # 添加聚合统计
    posi_cols = [f'agent{i}_posi' for i in range(1, 7)]
    negi_cols = [f'agent{i}_negi' for i in range(1, 7)]
    poss_cols = [f'agent{i}_poss' for i in range(1, 7)]
    negs_cols = [f'agent{i}_negs' for i in range(1, 7)]

    df_new['ave_Ip'] = df_new[posi_cols].mean(axis=1)
    df_new['ave_In'] = df_new[negi_cols].mean(axis=1)
    df_new['ave_Sp'] = df_new[poss_cols].mean(axis=1)
    df_new['ave_Sn'] = df_new[negs_cols].mean(axis=1)
    df_new['max_Ip'] = df_new[posi_cols].max(axis=1)
    df_new['max_In'] = df_new[negi_cols].max(axis=1)
    df_new['max_Sp'] = df_new[poss_cols].max(axis=1)
    df_new['max_Sn'] = df_new[negs_cols].max(axis=1)

    return df_new


# ----------------------------- 5. 绘图函数（四合一热力图，只针对 DeepSeek）-----------------------------
def plot_4in1_heatmap(df_deepseek, id_val, yd_val, save_path):
    """
    df_deepseek: 已经过滤好只包含 DeepSeek-V3.2 的 DataFrame
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")

    plot_configs = [
        ("ave_Ip", "ave_In", "Group: Positive vs Negative Influence"),
        ("ave_Sp", "ave_Sn", "Group: Positive vs Negative Stubbornness"),
        ("max_Ip", "max_In", "Leader: Max Positive vs Max Negative Influence"),
        ("max_Sp", "max_Sn", "Leader: Max Positive vs Max Negative Stubbornness")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (x_col, y_col, title) in enumerate(plot_configs):
        ax = axes[idx]
        heat_df = pd.DataFrame({
            "x": df_deepseek[x_col],
            "y": df_deepseek[y_col],
            "value": df_deepseek["末轮平均观点"]
        })
        # 分箱
        heat_df["x_bin"] = pd.cut(heat_df["x"], bins=15)
        heat_df["y_bin"] = pd.cut(heat_df["y"], bins=15)
        pivot = heat_df.groupby(["x_bin", "y_bin"])["value"].mean().unstack()

        sns.heatmap(pivot.T, cmap="coolwarm", ax=ax, vmin=-1, vmax=1,
                    cbar=False, linewidths=0.3, square=False)

        # 坐标轴真实值
        ax.set_xticks(np.linspace(0, len(pivot.columns) - 1, 15))
        ax.set_yticks(np.linspace(0, len(pivot.index) - 1, 15))
        ax.set_xticklabels(np.round(np.linspace(df_deepseek[x_col].min(), df_deepseek[x_col].max(), 15), 2))
        ax.set_yticklabels(np.round(np.linspace(df_deepseek[y_col].min(), df_deepseek[y_col].max(), 15), 2))

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax).set_label("Final Average Opinion\n(1=correct, -1=wrong)")

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


# ----------------------------- 6. 主流程 -----------------------------
def main():
    # 创建输出文件夹
    output_dir = "heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    print(f"图片将保存到文件夹: {output_dir}/")

    print("Loading original data...")
    original_df = load_original_data()

    print("Computing convergence results (once)...")
    conv_df = compute_convergence_results(original_df)

    # 定义超参数搜索范围
    id_start, id_end, id_step = 1.0, 2.0, 0.01
    yd_start, yd_end, yd_step = 1.0, 1.51, 0.01

    id_vals = np.arange(id_start, id_end, id_step)
    yd_vals = np.arange(yd_start, yd_end, yd_step)

    print(f"Total hyperparam combinations: {len(id_vals) * len(yd_vals)}")

    valid_params = []
    for id_val in id_vals:
        for yd_val in yd_vals:
            if is_valid_hyperparam(id_val, yd_val):
                valid_params.append((id_val, yd_val))

    print(f"Valid parameter pairs: {len(valid_params)}")

    for idx, (id_val, yd_val) in enumerate(valid_params):
        print(f"\nProcessing {idx + 1}/{len(valid_params)}: id={id_val:.4f}, yd={yd_val:.4f}")

        pos_inf, neg_inf, pos_stub, neg_stub = compute_influence(original_df, id_val, yd_val)
        final_df = assemble_final_data(conv_df, pos_inf, neg_inf, pos_stub, neg_stub)
        df_ds = final_df[final_df['LLM'] == "DeepSeek-V3.2"].copy()
        if df_ds.empty:
            print(f"Warning: No DeepSeek data for id={id_val}, yd={yd_val}, skip.")
            continue

        # 图片路径：文件夹 + 文件名
        filename = f"DeepSeek_{id_val:.2f}_{yd_val:.2f}.png"
        save_path = os.path.join(output_dir, filename)
        plot_4in1_heatmap(df_ds, id_val, yd_val, save_path)

    print("\nAll done!")


if __name__ == "__main__":
    main()