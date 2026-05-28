import pandas as pd

# 读取文件（注意：若文件有多个 sheet，默认读取第一个）
df = pd.read_excel("llmasjudge.xlsx", sheet_name=0)

# 统一处理 ans 列：去除空格、转为大写字符串
df['ans'] = df['ans'].astype(str).str.strip().str.upper()

# 分割数据
false_rows = df[df['ans'] == 'FALSE'].copy()   # 全部 FALSE 行
true_rows = df[df['ans'] == 'TRUE'].copy()     # 全部 TRUE 行

# 对 TRUE 行按 1% 比例抽取（至少 1 行，如果总数大于 0）
sample_ratio = 0.01
n_true = len(true_rows)
if n_true > 0:
    n_sample = max(1, int(n_true * sample_ratio))  # 至少抽 1 行
    sampled_true = true_rows.sample(n=n_sample, random_state=42)  # random_state 保证可复现
else:
    sampled_true = pd.DataFrame()

# 合并结果
result = pd.concat([false_rows, sampled_true], ignore_index=True)

# 保存到新 Excel 文件
result.to_excel("sampled_result.xlsx", index=False)

print(f"处理完成：")
print(f"  - FALSE 行数：{len(false_rows)}，全部保留")
print(f"  - TRUE 行数：{n_true}，按 {sample_ratio*100:.0f}% 抽取 {len(sampled_true)} 行")
print(f"  - 最终抽取结果共 {len(result)} 行，已保存至 sampled_result.xlsx")