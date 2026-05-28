import json
import time
import pandas as pd
from openai import OpenAI
import os

API_KEY = "sk-UIZbzdwsHCUlLYXvf6oHBDBoXAlrzhK8zAjg3Q1J4iBKhUjE"
BASE_URL = "https://hk.n1n.ai/v1"
MODEL_NAME = "qwen3.5-35b-a3b"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============ System Prompt（你的完整要求）============
SYSTEM_PROMPT = """
# Role
你是一位精通语用学（Pragmatics）、社会语言学（Sociolinguistics）和心理学（Psychology）的学术分析专家。你需要帮我分析一批人类的发言数据。

# Task Context
发言者分为“掌握正确观点（讲真话）”和“持有错误观点（但自身不知情，盲目自信）”两种情况。这两种情况下的发言，都会因为发言者潜意识中的“社会地位高低”而展现出不同的语用策略。

# Goal
我将为你提供【场景类型】（A.正确观点 或 B.错误观点）以及【发言文本】。请你根据提供的《编码手册》，分析该发言主要使用了哪些“语用策略”。

# 《编码手册》（Codebook）
（请严格从以下策略中选择，可多选。策略编号及定义如下：）

【场景A：讲真话时的策略】
- S1-绝对化断言：频繁使用极度确定的词汇（必须、绝对、毫无疑问）。
- S2-模糊与弱化：使用缓冲词（可能、也许、我个人觉得）降低锋芒。
- S3-借用外部言据：引用权威或他人（专家说、书上写）增加可信度。
- S4-祈使控制：使用命令语气，直接要求对方接受。
- S5-自我贬低/消极礼貌：陈述前先贬低自己或道歉（献丑了、我可能不懂）。
- S6-语用傲慢/反问：面对质疑不屑解释，通过反问句式压制对方。
- S7-过度解释：面对质疑急于自证，使用大量逻辑连词长篇大论。

【场景B：持有错误观点时的策略】
- S8-跨界类比：用自己熟悉领域的经验强行解释不懂的领域（跟我们做XX一样）。
- S9-居高临下的降维：将反对视为对方眼界不够（你站得不够高、你以后就懂了）。
- S10-诉诸民间/反智：用民间常识或反权威情绪辩护错误（老百姓都知道、专家扯淡）。
- S11-伪深刻/学术黑话：使用宏大词汇或行业黑话包装逻辑漏洞（底层逻辑、格局）。
- S12-轶事谬误：依赖个别具体经历（我亲戚、我见过）代替统计规律。
- S13-黑白二元逻辑：使用极端的量词（全部、根本没）将复杂问题简单化。
- S14-发誓/情绪化承诺：用强烈情绪或誓言（我敢打赌、拿命担保）弥补逻辑虚弱。

# Output Format (JSON 格式)
请严格按以下 JSON 格式输出，不要输出任何额外的废话：
{
  "Scenario": "A 或 B",
  "Detected_Strategies": ["填入策略编号及名称，如 S12-轶事谬误", "S14-发誓/情绪化承诺"],
  "Reasoning": "简要解释为什么选择这些策略，结合发言中的具体词汇和句式分析（限100字以内）。",
  "Estimated_Status": "High 或 Low 或 Unclear (根据策略倾向推断其社会地位)"
}"""

# ============ 标注函数（含重试）============
def annotate_text(scenario, text, max_retries=3):
    user_message = f"【场景类型】：{scenario}\n【发言文本】：{text}"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,          # 低温度保证输出稳定
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
            content = response.choices[0].message.content.strip()
            # 解析 JSON
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"第{attempt+1}次尝试失败: {e}")
            time.sleep(2)  # 避免频繁请求
    return {"Scenario": scenario, "Detected_Strategies": [], "Reasoning": "标注失败", "Estimated_Status": "Unclear"}

# ============ doing a thing ============
SAVE_EVERY = 10               # 每处理多少条保存一次
OUTPUT_FILE = "pragmatic_ans3.xlsx"
batch_rows = []
df = pd.read_excel("pragmatics.xlsx", sheet_name="Sheet1")
# 存储最终结果的列表
output_rows = []

num = 1
print(f"共需处理 {len(df)} 条数据，开始标注...")
for i, (idx, row) in enumerate(df.iterrows()):
    if i <=35384:continue##skip
    scenario = row["opinion"]
    text = row["content"]
    print(time.localtime())

    annotation = annotate_text(scenario, text)

    # 拼接策略
    strategies_str = ", ".join(annotation.get("Detected_Strategies", []))
    reasoning = annotation.get("Reasoning", "")
    est_status = annotation.get("Estimated_Status", "Unclear")

    # 将原始行数据转为字典，并追加标注结果
    row_dict = row.to_dict()
    row_dict["检测到的策略"] = strategies_str
    row_dict["推理说明"] = reasoning
    row_dict["推断社会地位"] = est_status
    batch_rows.append(row_dict)

    if len(batch_rows) >= SAVE_EVERY:
        # 如果文件已存在，读取后追加；否则新建
        if os.path.exists(OUTPUT_FILE):
            existing = pd.read_excel(OUTPUT_FILE)
            updated = pd.concat([existing, pd.DataFrame(batch_rows)], ignore_index=True)
        else:
            updated = pd.DataFrame(batch_rows)
        updated.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
        batch_rows = []  # 清空批次缓存
        print(f"writing till {i}")
if batch_rows:
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_excel(OUTPUT_FILE)
        updated = pd.concat([existing, pd.DataFrame(batch_rows)], ignore_index=True)
    else:
        updated = pd.DataFrame(batch_rows)
    updated.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

print("全部处理完毕！")