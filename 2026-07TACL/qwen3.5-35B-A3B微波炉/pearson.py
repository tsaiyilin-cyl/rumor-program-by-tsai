import pandas as pd
from scipy.stats import pearsonr

# 读取Excel文件
df = pd.read_excel('annotated_speeches.xlsx', sheet_name='Sheet1')

# 提取需要的列
target_columns = [
    '情感维度得分',
    '来源可信度得分',
    '逻辑性维度得分',
    '总说服力得分',
    'influence'
]
df_target = df[target_columns].copy()

# 重命名influence列为"agent影响力"，方便结果查看
df_target.rename(columns={'influence': 'agent影响力'}, inplace=True)

# 去除空值行
df_target = df_target.dropna()

# 转换为数值类型，确保计算正确
for col in df_target.columns:
    df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
df_target = df_target.dropna()

# 1. 计算Pearson相关系数矩阵
corr_matrix = df_target.corr(method='pearson')
print("===== Pearson相关系数矩阵 =====")
print(corr_matrix.round(4))
print("\n")

# 2. 计算每对变量的相关系数和p值（显著性）
print("===== 各变量对的Pearson相关系数与显著性p值 =====")
columns = df_target.columns
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        col1 = columns[i]
        col2 = columns[j]
        corr, p_value = pearsonr(df_target[col1], df_target[col2])
        # 标记显著性
        significance = ""
        if p_value < 0.001:
            significance = "*** (p<0.001，极显著)"
        elif p_value < 0.01:
            significance = "** (p<0.01，高度显著)"
        elif p_value < 0.05:
            significance = "* (p<0.05，显著)"
        else:
            significance = "(p≥0.05，不显著)"
        print(f"{col1} 与 {col2}：相关系数 = {corr:.4f}，p值 = {p_value:.6f} {significance}")