import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import os
warnings.filterwarnings('ignore')

# ==================== 1. 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']        # 使用黑体
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题

# ==================== 2. 输出目录 ====================
output_dir = "heatmaps_valid_params"
os.makedirs(output_dir, exist_ok=True)
print(f"图片将保存到文件夹: {output_dir}/")

# 存储所有参数对的得分和详细信息
score_matrix = {}  # {(id_val, yd_val): score}
results_data = []  # 用于保存所有参数对的结果

# ==================== 3. 加载原始数据 ====================
def load_original_data():
    """返回原始观点数据 DataFrame"""
    import get_data
    return get_data.step()

# ==================== 4. 超参数合法性判断 ====================
def is_valid_hyperparam(id_val, yd_val):
    """判断超参数(id, yd)是否满足 c12 > c11_c21 > c22 > c11_c31"""
    def compute_c(i, op):
        return id_val ** (-i) * (op ** yd_val)
    
    c12 = compute_c(1, 2)
    c11_c21 = compute_c(1, 1) + compute_c(2, 1)
    c22 = compute_c(2, 2)
    c11_c31 = compute_c(1, 1) + compute_c(3, 1)
    
    return c12 > c11_c21 > c22 > c11_c31

# ==================== 5. 影响力计算 ====================
def compute_influence(data_df, id_val, yd_val):
    """
    根据超参数计算 influence 和 stubbornness 矩阵
    返回: (pos_influence, neg_influence, pos_stubborn, neg_stubborn)
    每个都是 7x7x36x7 的 list (1-based索引)
    """
    def compute_c(i, op):
        return id_val ** (-i) * (op ** yd_val)
    
    def sex_race_to_agent(sex, race):
        return (sex - 1) * 3 + (race - 1) + 1
    
    # 初始化存储结构
    pos_influence = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    neg_influence = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    pos_stubborn = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    neg_stubborn = [[[[0.0 for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    
    pos_influencet = [[[[[[] for _ in range(7)] for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    neg_influencet = [[[[[[] for _ in range(7)] for _ in range(7)] for _ in range(36)] for _ in range(7)] for _ in range(7)]
    
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
                
                c_val = compute_c(i, abs(delta))
                # stubbornness 更新
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

# ==================== 6. 组装最终数据 ====================
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

# ==================== 6.5. 计算理想矩阵和损失 ====================
def compute_ideal_matrix_and_loss(df_filtered, id_val, yd_val):
    """
    计算10x10矩阵的损失
    返回: (total_loss, valid_count, loss_matrix, value_matrix, count_matrix)
    """
    # 创建10x10的固定网格
    ip_min, ip_max = df_filtered["ave_Ip"].min(), df_filtered["ave_Ip"].max()
    in_min, in_max = df_filtered["ave_In"].min(), df_filtered["ave_In"].max()
    
    # 生成10个bin的边界
    ip_bins = np.linspace(ip_min, ip_max, 11)
    in_bins = np.linspace(in_min, in_max, 11)
    
    # 初始化10x10矩阵
    value_matrix = np.full((10, 10), np.nan)
    count_matrix = np.zeros((10, 10))
    
    # 将数据分配到10x10网格中
    for _, row in df_filtered.iterrows():
        ip_val = row["ave_Ip"]
        in_val = row["ave_In"]
        opinion_val = row["末轮平均观点"]
        
        ip_idx = np.digitize(ip_val, ip_bins) - 1
        in_idx = np.digitize(in_val, in_bins) - 1
        
        ip_idx = max(0, min(9, ip_idx))
        in_idx = max(0, min(9, in_idx))
        
        if np.isnan(value_matrix[in_idx, ip_idx]):
            value_matrix[in_idx, ip_idx] = opinion_val
        else:
            value_matrix[in_idx, ip_idx] += opinion_val
        count_matrix[in_idx, ip_idx] += 1
    
    # 计算每个bin的平均观点值
    for i in range(10):
        for j in range(10):
            if count_matrix[i, j] > 0:
                value_matrix[i, j] /= count_matrix[i, j]
    
    # 获取所有有效格点的坐标和值
    valid_cells = []
    for i in range(10):
        for j in range(10):
            if not np.isnan(value_matrix[i, j]):
                valid_cells.append((i, j, value_matrix[i, j]))
    
    # 按观点值从小到大排序
    valid_cells.sort(key=lambda x: x[2])
    
    # 将相同值的格子分组
    value_groups = {}
    for i, j, val in valid_cells:
        # 使用四舍五入到小数点后10位来避免浮点数精度问题
        val_rounded = round(val, 10)
        if val_rounded not in value_groups:
            value_groups[val_rounded] = []
        value_groups[val_rounded].append((i, j, val))
    
    # 为每个值组分配连续的秩次范围
    value_rank_ranges = {}
    current_rank = 1
    for val_rounded in sorted(value_groups.keys()):
        group = value_groups[val_rounded]
        count = len(group)
        value_rank_ranges[val_rounded] = (current_rank, current_rank + count - 1)
        current_rank += count
    
    # 统计每条对角线上有多少个有效格子
    diagonal_counts = {}
    for i, j, val in valid_cells:
        b = i - j
        diagonal_counts[b] = diagonal_counts.get(b, 0) + 1
    
    # 按b从大到小排序
    sorted_diagonals = sorted(diagonal_counts.keys(), reverse=True)
    
    # 为每条对角线分配理论秩次范围
    diagonal_ranges = {}
    current_rank = 1
    for b in sorted_diagonals:
        count = diagonal_counts[b]
        diagonal_ranges[b] = (current_rank, current_rank + count - 1)
        current_rank += count
    
    # 计算每个格子的损失：对于相同值的格子，选择损失最小的对角线分配
    # 使用0-1损失：如果b_actual == b_ideal则损失为0，否则为1
    loss_matrix = np.full((10, 10), np.nan)
    total_loss = 0
    valid_count = len(valid_cells)
    
    # 对每个值组，找到最优的对角线分配方案
    for val_rounded, group in value_groups.items():
        rank_start, rank_end = value_rank_ranges[val_rounded]
        
        # 找出该秩次范围可能对应的所有对角线
        candidate_diagonals = []
        for b, (r_start, r_end) in diagonal_ranges.items():
            # 检查对角线与值组的秩次范围是否有重叠
            if r_start <= rank_end and r_end >= rank_start:
                candidate_diagonals.append(b)
        
        # 对于组内的每个格子，尝试所有候选对角线，选择损失最小的（0-1损失）
        for i, j, val in group:
            b_actual = i - j
            min_loss = float('inf')
            best_b_ideal = None
            
            for b_candidate in candidate_diagonals:
                # 0-1损失：如果匹配则为0，否则为1
                loss = 0 if b_actual == b_candidate else 1
                if loss < min_loss:
                    min_loss = loss
                    best_b_ideal = b_candidate
                # 如果已经找到损失为0的最优解，提前退出
                if min_loss == 0:
                    break
            
            if best_b_ideal is not None:
                loss_matrix[i, j] = min_loss
                total_loss += min_loss
    
    return total_loss, valid_count, loss_matrix, value_matrix, count_matrix

# ==================== 7. 绘制 ave_IP-ave_In 热力图 ====================
def plot_aveIP_aveIn_heatmap(df_filtered, id_val, yd_val, save_path):
    """
    只绘制 ave_IP vs ave_In 的热力图
    df_filtered: 已经过滤好只包含指定四个模型的 DataFrame
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    heat_df = pd.DataFrame({
        "x": df_filtered["ave_Ip"],
        "y": df_filtered["ave_In"],
        "value": df_filtered["末轮平均观点"]
    })
    
    # 分箱
    heat_df["x_bin"] = pd.cut(heat_df["x"], bins=15)
    heat_df["y_bin"] = pd.cut(heat_df["y"], bins=15)
    pivot = heat_df.groupby(["x_bin", "y_bin"])["value"].mean().unstack()
    
    sns.heatmap(pivot.T, cmap="coolwarm", ax=ax, vmin=-1, vmax=1,
                cbar=True, linewidths=0.3, square=False)
    
    # 坐标轴真实值
    ax.set_xticks(np.linspace(0, len(pivot.columns) - 1, 15))
    ax.set_yticks(np.linspace(0, len(pivot.index) - 1, 15))
    ax.set_xticklabels(np.round(np.linspace(df_filtered["ave_Ip"].min(), df_filtered["ave_Ip"].max(), 15), 2), rotation=45)
    ax.set_yticklabels(np.round(np.linspace(df_filtered["ave_In"].min(), df_filtered["ave_In"].max(), 15), 2), rotation=0)
    
    ax.set_title(f"ave_Ip vs ave_In (id={id_val:.2f}, yd={yd_val:.2f})", fontsize=13, fontweight='bold')
    ax.set_xlabel("ave_Ip (Average Positive Influence)", fontsize=11)
    ax.set_ylabel("ave_In (Average Negative Influence)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

# ==================== 8. 绘制带损失标注的热力图 ====================
def plot_loss_heatmap(loss_matrix, value_matrix, total_loss, valid_count, id_val, yd_val, ip_min, ip_max, in_min, in_max, save_path):
    """
    绘制损失热力图，每个格子显示abs(b-c)的值
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建掩码，隐藏无效格子
    mask = np.isnan(loss_matrix)
    
    # 绘制热力图（注意：需要翻转矩阵使纵轴从下往上增大）
    loss_matrix_flipped = np.flipud(loss_matrix)  # 上下翻转
    mask_flipped = np.flipud(mask)
    
    sns.heatmap(loss_matrix_flipped, mask=mask_flipped, cmap="YlOrRd", ax=ax, 
                cbar=True, linewidths=0.5, linecolor='gray',
                annot=True, fmt='.0f', annot_kws={'size': 8},
                vmin=0, vmax=1)
    
    # 设置标题，包含总损失和有效格点数
    title = f"Loss Matrix (id={id_val:.2f}, yd={yd_val:.2f})\nTotal Loss: {total_loss:.1f}, Valid Cells: {valid_count}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置坐标轴标签
    ax.set_xlabel("ave_Ip (Average Positive Influence)", fontsize=11)
    ax.set_ylabel("ave_In (Average Negative Influence)", fontsize=11)
    
    # 设置刻度位置和真实值（注意：因为矩阵翻转了，所以in_tick_values也要翻转）
    tick_positions = np.arange(0.5, 10.5, 1)
    ip_tick_values = np.linspace(ip_min, ip_max, 10)
    in_tick_values = np.linspace(in_min, in_max, 10)[::-1]  # 翻转以匹配翻转后的矩阵
    
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(np.round(ip_tick_values, 2), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(np.round(in_tick_values, 2), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot: {save_path}")

# ==================== 8.5. 保存原始矩阵图 ====================
def save_value_matrix_plot(value_matrix, count_matrix, id_val, yd_val, ip_min, ip_max, in_min, in_max, output_dir):
    """
    保存原始观点值矩阵的热力图，每个格子显示"观点值+样本数"
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 创建掩码，隐藏无效格子
    mask = np.isnan(value_matrix)
    
    # 翻转矩阵使纵轴从下往上增大
    value_matrix_flipped = np.flipud(value_matrix)
    mask_flipped = np.flipud(mask)
    count_matrix_flipped = np.flipud(count_matrix)
    
    # 创建自定义标注：观点值+样本数
    annot_matrix = np.empty_like(value_matrix_flipped, dtype=object)
    for i in range(10):
        for j in range(10):
            if not mask_flipped[i, j]:
                val = value_matrix_flipped[i, j]
                count = int(count_matrix_flipped[i, j])
                annot_matrix[i, j] = f"{val:.2f}+{count}"
            else:
                annot_matrix[i, j] = ""
    
    # 绘制热力图
    sns.heatmap(value_matrix_flipped, mask=mask_flipped, cmap="coolwarm", ax=ax, 
                cbar=True, linewidths=0.5, linecolor='gray',
                annot=annot_matrix, fmt='', annot_kws={'size': 5},
                vmin=-1, vmax=1)
    
    # 设置标题
    title = f"Value Matrix (id={id_val:.2f}, yd={yd_val:.2f})"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置坐标轴标签
    ax.set_xlabel("ave_Ip (Average Positive Influence)", fontsize=11)
    ax.set_ylabel("ave_In (Average Negative Influence)", fontsize=11)
    
    # 设置刻度位置和真实值（注意：因为矩阵翻转了，所以in_tick_values也要翻转）
    tick_positions = np.arange(0.5, 10.5, 1)
    ip_tick_values = np.linspace(ip_min, ip_max, 10)
    in_tick_values = np.linspace(in_min, in_max, 10)[::-1]  # 翻转以匹配翻转后的矩阵
    
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(np.round(ip_tick_values, 2), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(np.round(in_tick_values, 2), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    filename = f"value_matrix_id{id_val:.2f}_yd{yd_val:.2f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved value matrix plot: {filepath}")
    
    return filepath

# ==================== 9. 主流程 ====================
def main():
    print("Loading original data...")
    original_df = load_original_data()
    
    # 保留所有6个模型（使用数字索引过滤）
    allowed_models = ["DeepSeek-V3.2", "GPT-5.1", "Llama-3.3-70b-instruct", 
                      "Gemini-3.1-Flash-Lite-Preview", "Qwen3.5-Flash", "Qwen3.5-35B-A3B"]
    model_map_reverse = {
        1: "DeepSeek-V3.2",
        2: "GPT-5.1", 
        3: "Llama-3.3-70b-instruct",
        4: "Gemini-3.1-Flash-Lite-Preview",
        5: "Qwen3.5-Flash",
        6: "Qwen3.5-35B-A3B"
    }
    allowed_llm_indices = [idx for idx, name in model_map_reverse.items() if name in allowed_models]
    original_df = original_df[original_df['llm'].isin(allowed_llm_indices)].copy()
    
    print(f"Filtered data to only include: {allowed_models}")
    print(f"Data shape after filtering: {original_df.shape}")
    
    # 定义超参数搜索范围
    id_start, id_end, id_step = 1.0, 1.51, 0.01
    yd_start, yd_end, yd_step = 1.0, 1.51, 0.01
    
    id_vals = np.arange(id_start, id_end, id_step)
    yd_vals = np.arange(yd_start, yd_end, yd_step)
    
    print(f"Total hyperparam combinations: {len(id_vals) * len(yd_vals)}")
    
    # 遍历所有参数组合
    count_processed = 0
    count_valid = 0
    count_invalid = 0
    
    for id_val in id_vals:
        for yd_val in yd_vals:
            count_processed += 1
            
            # 检查是否合法
            if not is_valid_hyperparam(id_val, yd_val):
                # 不合法，跳过（涂白 = 不画图）
                count_invalid += 1
                
                # 记录非法参数的信息
                results_data.append({
                    'id_val': round(id_val, 4),
                    'yd_val': round(yd_val, 4),
                    'is_valid': False,
                    'total_loss': None,
                    'valid_count': None,
                    'normalized_loss': None,
                    'score': None
                })
                
                if count_processed % 1000 == 0:
                    print(f"Processed {count_processed}/{len(id_vals) * len(yd_vals)} combinations, "
                          f"Valid: {count_valid}, Invalid: {count_invalid}")
                continue
            
            # 合法参数，计算并绘图
            count_valid += 1
            print(f"\nProcessing valid param {count_valid}: id={id_val:.4f}, yd={yd_val:.4f} "
                  f"(#{count_processed}/{len(id_vals) * len(yd_vals)})")
            
            # 计算影响力
            pos_inf, neg_inf, pos_stub, neg_stub = compute_influence(original_df, id_val, yd_val)
            
            # 计算收敛结果
            from itertools import product
            model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct",
                         4: "Gemini-3.1-Flash-Lite-Preview", 5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}
            
            convs = []
            grouped = original_df.groupby(['llm', 'topic', 'experiment_index'])
            
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
            
            conv_df = pd.DataFrame(convs)
            
            # 组装最终数据
            final_df = assemble_final_data(conv_df, pos_inf, neg_inf, pos_stub, neg_stub)
            
            # 计算损失矩阵
            total_loss, valid_count, loss_matrix, value_matrix, count_matrix = compute_ideal_matrix_and_loss(final_df, id_val, yd_val)
            
            # 计算归一化损失和得分
            if valid_count > 0:
                normalized_loss = total_loss / valid_count
                score = 1.0 - normalized_loss
            else:
                normalized_loss = 0.0
                score = 1.0
            
            # 存储得分
            score_matrix[(id_val, yd_val)] = score
            
            # 记录合法参数的详细信息
            results_data.append({
                'id_val': round(id_val, 4),
                'yd_val': round(yd_val, 4),
                'is_valid': True,
                'total_loss': total_loss,
                'valid_count': valid_count,
                'normalized_loss': normalized_loss,
                'score': score
            })
            
            print(f"  Total Loss: {total_loss:.1f}, Valid Cells: {valid_count}, Normalized Loss: {normalized_loss:.4f}, Score: {score:.4f}")
    
    print(f"\n{'='*60}")
    print(f"All computations done!")
    print(f"Total combinations processed: {count_processed}")
    print(f"Valid parameters: {count_valid}")
    print(f"Invalid parameters (skipped): {count_invalid}")
    print(f"{'='*60}")
    
    # 保存结果到CSV文件
    print(f"\nSaving results to CSV...")
    results_df = pd.DataFrame(results_data)
    csv_filepath = os.path.join(output_dir, "parameter_space_results.csv")
    results_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f"Saved results to: {csv_filepath}")
    print(f"Total records: {len(results_df)}")
    print(f"Valid params: {results_df['is_valid'].sum()}")
    print(f"Invalid params: {(~results_df['is_valid']).sum()}")
    
    # 绘制总参数空间热力图
    print(f"\nPlotting overall parameter space heatmap...")
    plot_overall_score_heatmap(score_matrix, id_vals, yd_vals, output_dir)

def plot_overall_score_heatmap(score_matrix, id_vals, yd_vals, output_dir):
    """
    绘制总参数空间的热力图，显示每个合法参数对的得分
    """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    
    # 创建得分矩阵
    n_id = len(id_vals)
    n_yd = len(yd_vals)
    score_grid = np.full((n_yd, n_id), np.nan)  # yd为行，id为列
    
    for (id_val, yd_val), score in score_matrix.items():
        # 找到对应的索引
        id_idx = np.argmin(np.abs(id_vals - id_val))
        yd_idx = np.argmin(np.abs(yd_vals - yd_val))
        score_grid[yd_idx, id_idx] = score
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    mask = np.isnan(score_grid)
    
    # 翻转矩阵使纵轴从下往上增大
    score_grid_flipped = np.flipud(score_grid)
    mask_flipped = np.flipud(mask)
    
    # 使用黄-红配色
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    sns.heatmap(score_grid_flipped, mask=mask_flipped, cmap=cmap, ax=ax,
                cbar=True, linewidths=0.3, linecolor='gray',
                annot=False, vmin=0, vmax=1)
    
    # 设置标题
    ax.set_title("Parameter Space Score Heatmap\n(Yellow=High Score, Red=Low Score, White=Invalid)", 
                 fontsize=14, fontweight='bold')
    
    # 设置坐标轴标签
    ax.set_xlabel("id_val (Influence Decay)", fontsize=12)
    ax.set_ylabel("yd_val (Opinion Sensitivity)", fontsize=12)
    
    # 设置刻度位置和真实值
    # 每隔一定间隔显示刻度，避免过于拥挤
    step = max(1, n_id // 10)
    tick_positions_id = np.arange(0, n_id, step)
    tick_positions_yd = np.arange(0, n_yd, step)
    
    id_tick_values = id_vals[::step]
    yd_tick_values = yd_vals[::step][::-1]  # 翻转以匹配翻转后的矩阵
    
    ax.set_xticks(tick_positions_id + 0.5)
    ax.set_yticks(tick_positions_yd + 0.5)
    ax.set_xticklabels(np.round(id_tick_values, 2), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(np.round(yd_tick_values, 2), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "overall_score_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall score heatmap: {filepath}")
    
    # 统计信息
    valid_scores = [s for s in score_matrix.values()]
    if valid_scores:
        print(f"\nScore Statistics:")
        print(f"  Mean Score: {np.mean(valid_scores):.4f}")
        print(f"  Max Score: {np.max(valid_scores):.4f}")
        print(f"  Min Score: {np.min(valid_scores):.4f}")
        print(f"  Std Dev: {np.std(valid_scores):.4f}")

if __name__ == "__main__":
    main()
