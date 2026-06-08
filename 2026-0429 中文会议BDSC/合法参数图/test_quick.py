"""
快速测试脚本 - 用于验证合法参数图的计算逻辑是否正确
测试少量参数组合，包括完整的损失矩阵计算功能
"""
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def is_valid_hyperparam(id_val, yd_val):
    """判断超参数(id, yd)是否满足 c12 > c11_c21 > c22 > c11_c31"""
    def compute_c(i, op):
        return id_val ** (-i) * (op ** yd_val)
    
    c12 = compute_c(1, 2)
    c11_c21 = compute_c(1, 1) + compute_c(2, 1)
    c22 = compute_c(2, 2)
    c11_c31 = compute_c(1, 1) + compute_c(3, 1)
    
    return c12 > c11_c21 > c22 > c11_c31

def test_validity_check():
    """测试合法性检查函数"""
    print("测试合法性检查函数...")
    
    # 测试几个已知的参数点
    test_cases = [
        (1.04, 1.02, True),
        (1.20, 1.08, True),
        (1.40, 1.10, True),
        (1.40, 1.25, True),
        (1.00, 1.00, False),
        (1.50, 1.50, False),
    ]
    
    for id_val, yd_val, expected in test_cases:
        result = is_valid_hyperparam(id_val, yd_val)
        status = "✓" if result == expected else "✗"
        print(f"  {status} id={id_val:.2f}, yd={yd_val:.2f}: valid={result} (expected={expected})")
    
    print()

def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    
    try:
        import get_data
        df = get_data.step()
        print(f"  ✓ 数据加载成功，形状: {df.shape}")
        print(f"  ✓ 列名: {list(df.columns)}")
        
        # 检查模型分布
        if 'llm' in df.columns:
            llm_counts = df['llm'].value_counts()
            print(f"\n  模型分布:")
            for llm, count in llm_counts.items():
                print(f"    - {llm}: {count} 条记录")
        
        print()
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        print()

def compute_ideal_matrix_and_loss(df_filtered, id_val, yd_val):
    """
    计算10x10矩阵的损失（简化版，用于测试）
    返回: (total_loss, valid_count, loss_matrix, value_matrix)
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
    print(f"  ✓ Saved value matrix plot: {filepath}")
    
    return filepath


def plot_loss_heatmap(loss_matrix, value_matrix, total_loss, valid_count, id_val, yd_val, ip_min, ip_max, in_min, in_max, save_path):
    """
    绘制损失热力图（测试用）
    """
    plt.rcParams['font.family'] = 'Arial'
    sns.set_style("white")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.isnan(loss_matrix)
    
    # 翻转矩阵使纵轴从下往上增大
    loss_matrix_flipped = np.flipud(loss_matrix)
    mask_flipped = np.flipud(mask)
    
    sns.heatmap(loss_matrix_flipped, mask=mask_flipped, cmap="YlOrRd", ax=ax, 
                cbar=True, linewidths=0.5, linecolor='gray',
                annot=True, fmt='.0f', annot_kws={'size': 8},
                vmin=0, vmax=1)
    
    title = f"Loss Matrix (id={id_val:.2f}, yd={yd_val:.2f})\nTotal Loss: {total_loss:.1f}, Valid Cells: {valid_count}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
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
    print(f"  ✓ Saved loss plot: {save_path}")


def test_small_grid_with_loss():
    """测试小范围网格搜索并生成损失矩阵图（使用真实数据）"""
    print("测试小范围网格搜索（含损失矩阵计算）...")
    print("="*60)
    
    # 创建输出目录
    output_dir = "test_loss_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"图片将保存到: {output_dir}/\n")
    
    # 加载数据
    try:
        import get_data
        original_df = get_data.step()
        print(f"✓ 数据加载成功: {original_df.shape}\n")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}\n")
        return
    
    # 保留所有6个模型
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
    print(f"✓ 过滤后数据: {original_df.shape} (包含全部6个模型)\n")
    
    # 导入主程序的函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compute_valid_params import (
        is_valid_hyperparam,
        compute_influence,
        assemble_final_data
    )
    
    # 测试几个合法参数
    test_params = [
        (1.04, 1.02),
        (1.06, 1.02),
        (1.08, 1.04),
        (1.20,1.08),
    ]
    
    print(f"将测试 {len(test_params)} 个合法参数对:\n")
    
    for id_val, yd_val in test_params:
        if not is_valid_hyperparam(id_val, yd_val):
            print(f"  ✗ 跳过非法参数: id={id_val:.2f}, yd={yd_val:.2f}")
            continue
        
        print(f"处理参数: id={id_val:.2f}, yd={yd_val:.2f}")
        
        # 使用真实的影响力计算
        print("  - 计算影响力矩阵...")
        pos_inf, neg_inf, pos_stub, neg_stub = compute_influence(original_df, id_val, yd_val)
        
        # 计算收敛结果
        from itertools import product
        model_map = {1: "DeepSeek-V3.2", 2: "GPT-5.1", 3: "Llama-3.3-70b-instruct",
                     4: "Gemini-3.1-Flash-Lite-Preview", 5: "Qwen3.5-Flash", 6: "Qwen3.5-35B-A3B"}
        
        convs = []
        grouped = original_df.groupby(['llm', 'topic', 'experiment_index'])
        
        print("  - 计算收敛结果...")
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
        print(f"  - 收敛结果: {len(conv_df)} 条记录")
        
        # 组装最终数据
        print("  - 组装最终数据...")
        final_df = assemble_final_data(conv_df, pos_inf, neg_inf, pos_stub, neg_stub)
        
        # 获取数据范围
        ip_min, ip_max = final_df["ave_Ip"].min(), final_df["ave_Ip"].max()
        in_min, in_max = final_df["ave_In"].min(), final_df["ave_In"].max()
        
        print(f"  - ave_Ip 范围: [{ip_min:.4f}, {ip_max:.4f}]")
        print(f"  - ave_In 范围: [{in_min:.4f}, {in_max:.4f}]")
        
        # 计算损失矩阵
        print("  - 计算损失矩阵...")
        total_loss, valid_count, loss_matrix, value_matrix, count_matrix = compute_ideal_matrix_and_loss(
            final_df, id_val, yd_val
        )
        
        print(f"  - 总损失: {total_loss:.1f}")
        print(f"  - 有效格点数: {valid_count}")
        
        # 绘制损失热力图
        filename = f"test_loss_id{id_val:.2f}_yd{yd_val:.2f}.png"
        save_path = os.path.join(output_dir, filename)
        plot_loss_heatmap(loss_matrix, value_matrix, total_loss, valid_count, id_val, yd_val,
                         ip_min, ip_max, in_min, in_max, save_path)
        
        # 保存原始观点值矩阵图
        save_value_matrix_plot(value_matrix, count_matrix, id_val, yd_val, ip_min, ip_max, in_min, in_max, output_dir)
        print()
    
    print("="*60)
    print(f"测试完成！查看 {output_dir}/ 文件夹中的图片")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("合法参数图 - 快速测试（含损失矩阵功能）")
    print("="*60)
    print()
    
    test_validity_check()
    test_data_loading()
    test_small_grid_with_loss()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
