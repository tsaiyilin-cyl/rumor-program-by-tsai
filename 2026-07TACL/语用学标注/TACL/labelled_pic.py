import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import os
from pathlib import Path

def extract_title_from_folder(folder_name):
    """从文件夹名称中提取图表标题
    例如: speeches_DeepSeek-V3.2-WIFI_round7 -> DeepSeek-V3.2-WIFI
          speeches_DeepSeek-V3.2-精英人士 -> DeepSeek-V3.2-精英人士
    """
    # 移除 'speeches_' 前缀
    title = folder_name.replace('speeches_', '', 1)
    # 移除 '_round' 及后面的数字
    if '_round' in title:
        title = title.split('_round')[0]
    return title

def process_annotated_file(folder_path):
    """处理单个文件夹中的 annotated_speeches.xlsx 文件"""
    
    excel_file = folder_path / 'annotated_speeches.xlsx'
    
    if not excel_file.exists():
        print(f"跳过: {folder_path} (未找到 annotated_speeches.xlsx)")
        return
    
    # 提取图表标题
    folder_name = folder_path.name
    chart_title = extract_title_from_folder(folder_name)
    
    print(f"\n{'='*60}")
    print(f"正在处理: {folder_name}")
    print(f"图表标题: {chart_title}")
    print(f"{'='*60}")
    
    try:
        # ---------------------- 1. 数据读取与清洗 ----------------------
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
        
        target_columns = [
            '情感维度得分',
            '来源可信度得分',
            '逻辑性维度得分',
            '总说服力得分',
            'influence',
            '观点'
        ]
        
        df_target = df[target_columns].copy()
        df_target.rename(columns={'influence': 'agent影响力'}, inplace=True)
        
        # 清洗空值
        df_target = df_target.dropna()
        for col in df_target.columns:
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
        df_target = df_target.dropna()
        
        # 只保留观点为 1 或 -1 的样本
        df_target = df_target[df_target['观点'].isin([1, -1])]
        
        if len(df_target) == 0:
            print(f"警告: {folder_name} 清洗后无有效数据")
            return
        
        # 用于相关性计算的变量（不含"观点"）
        corr_cols = ['情感维度得分', '来源可信度得分', '逻辑性维度得分', '总说服力得分', 'agent影响力']
        df_corr = df_target[corr_cols].copy()
        
        # ---------------------- 2. 相关性计算 ----------------------
        print("===== Pearson相关系数矩阵 =====")
        corr_matrix = df_corr.corr(method='pearson')
        print(corr_matrix.round(4))
        print("\n===== 各变量对显著性检验 =====")
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                corr, p = pearsonr(df_corr[cols[i]], df_corr[cols[j]])
                sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                print(f"{cols[i]} ↔ {cols[j]} | r={corr:.4f} | p={p:.6f} {sig}")
        
        # 设置全局绘图风格
        sns.set_theme(style='whitegrid', palette='pastel', font='SimHei')
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        
        # ---------------------- 3. 热力图 ----------------------
        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11}
        )
        plt.title('Pearson 相关性热力矩阵图', fontsize=15, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.tight_layout()
        heatmap_path = folder_path / 'correlation_heatmap.png'
        plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"已保存热力图: {heatmap_path}")
        
        # ---------------------- 4. 配对图（分组 + 总回归线） ----------------------
        palette = {1: '#1f77b4', -1: '#d62728'}
        
        pair_grid = sns.pairplot(
            df_target,
            hue='观点',
            palette=palette,
            kind='reg',                    # 分组回归线（蓝、红）
            diag_kind='kde',
            plot_kws={
                'scatter_kws': {'alpha': 0.6, 's': 40, 'edgecolor': 'w'},
                'line_kws': {'lw': 1.5}
            },
            diag_kws={'fill': True, 'alpha': 0.5},
            corner=False,
            height=2.2,
            aspect=1.2
        )
        
        # 添加总回归线（黑色，不显示散点）
        # 获取变量列表（顺序与 pairplot 中一致）
        vars_list = df_target[corr_cols].columns.tolist()  # 注意不含 '观点'
        
        # 遍历所有非对角线的子图
        for i in range(len(vars_list)):
            for j in range(len(vars_list)):
                ax = pair_grid.axes[i, j]
                if ax is None or i == j:   # 跳过对角线（密度图）
                    continue
                # 获取当前子图的 x 和 y 变量名（通过子图标签）
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                if not xlabel or not ylabel:
                    continue
                # 提取整体数据（不分组）
                x_data = df_target[xlabel]
                y_data = df_target[ylabel]
                # 用 sns.regplot 只画回归线（scatter=False），加到当前子图上
                sns.regplot(
                    x=x_data, y=y_data,
                    scatter=False,
                    color='black',
                    line_kws={'linewidth': 2, 'linestyle': '--'},   # 黑色虚线，粗一点
                    ci=None,          # 不显示置信区间，避免杂乱
                    truncate=False,   # 回归线延伸至数据两端
                    ax=ax
                )
        
        # 调整标题 - 使用从文件夹名提取的标题
        pair_grid.fig.suptitle(f'{chart_title}两两变量关系图（蓝红=分组回归，黑色虚线=总回归线）', y=1.02, fontsize=16)
        pair_grid.fig.subplots_adjust(top=0.93)
        
        # 调整子图标签字体
        for ax in pair_grid.axes.flat:
            if ax is not None:
                ax.set_xlabel(ax.get_xlabel(), fontsize=9)
                ax.set_ylabel(ax.get_ylabel(), fontsize=9)
                ax.tick_params(labelsize=8)
        
        # 处理图例：分组图例会默认显示，手动修改位置避免遮挡
        # 获取已有的图例句柄并调整位置
        leg = pair_grid._legend
        if leg is not None:
            leg.set_bbox_to_anchor((0.85, 0.97))
            leg.set_title('观点')
        
        pairplot_path = folder_path / 'pairplot_variables_negpos.png'
        plt.savefig(pairplot_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"已保存配对图: {pairplot_path}")
        
    except Exception as e:
        print(f"错误: 处理 {folder_name} 时出错 - {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数：遍历所有子文件夹并处理"""
    base_path = Path(r'E:\A\TACL')
    
    print(f"开始处理 TACL 文件夹下的所有子文件夹...")
    print(f"基础路径: {base_path}")
    
    # 查找所有包含 annotated_speeches.xlsx 的子文件夹
    processed_count = 0
    for folder in base_path.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            excel_file = folder / 'annotated_speeches.xlsx'
            if excel_file.exists():
                process_annotated_file(folder)
                processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成！共处理了 {processed_count} 个文件夹")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
