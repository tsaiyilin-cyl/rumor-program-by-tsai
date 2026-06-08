import os
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time

client = OpenAI(
    api_key="sk-af67dbec0fa84d018a1057482090c2e5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL_NAME = "qwen3.6-plus"

SYSTEM_PROMPT = """请你作为专业的文本分析专家，基于亚里士多德说服理论（ethos、pathos、logos），对提供的文本进行精准标注和评分，严格遵循以下要求，输出规范、可直接用于数据整理的结果，无需额外冗余解释，仅按照指定格式返回标注内容。

一、标注核心维度及评分标准（总分均为10分，0分表示完全无相关体现，10分表示体现极其充分）
1. 情感维度（pathos）：评估文本通过语言、情感表达，引发受众情绪共鸣（如共情、感动、愤怒、认同等）的能力。评分依据：情感表达的自然度、针对性，以及对受众情绪的调动强度，无情感倾向则得0-2分，有明显情感导向且能引发共鸣得6-10分，介于两者之间得3-5分。
2. 来源可信度维度（ethos）：评估文本作者/信息来源的权威性、可信度，以及文本传递的真诚度、专业性。评分依据：是否有权威背书（如专业资质、真实案例、权威数据）、语言表达的严谨度、作者立场的客观性，无任何可信度支撑得0-2分，有明确权威支撑且真诚专业得6-10分，介于两者之间得3-5分。
3. 逻辑性维度（logos）：评估文本的论证逻辑、论据合理性、推理严谨性。评分依据：观点是否明确、论据是否充分且贴合观点、推理过程是否连贯无漏洞、是否符合逻辑规律（如因果、递进、对比等），无明确逻辑或逻辑混乱得0-2分，逻辑严谨、论据扎实得6-10分，介于两者之间得3-5分。
4. 总说服力得分：取上述三个维度得分的平均值（保留1位小数），综合反映文本的整体说服效果，评分范围0-10分。

二、输出格式（严格遵循，不得修改格式）：
pathos（情感）得分：X.X，评分说明：[简要说明得分理由，1-2句话，聚焦情感调动能力，不超过50字]
ethos（来源可信度）得分：X.X，评分说明：[简要说明得分理由，1-2句话，聚焦可信度支撑，不超过50字]
logos（逻辑性）得分：X.X，评分说明：[简要说明得分理由，1-2句话，聚焦逻辑严谨性，不超过50字]
本段总说服力得分：X.X

三、补充要求
1. 评分客观中立，严格对照评分标准，不主观臆断，避免过高或过低偏差；
2. 评分说明简洁精准，直击核心，不冗余、不偏离对应维度；
3. 若文本中某一维度无任何体现（如无情感表达、无可信度支撑），评分说明需明确说明“无相关体现”；
4. 严格按照上述格式输出，不得添加任何额外标题、解释性文字，确保API调用后可直接提取有效数据。

四、待标注文本
{text}
"""


def parse_model_output(output_text: str) -> dict:
    """从模型输出文本中提取各维度得分和说明"""
    patterns = {
        "pathos_score": r"pathos（情感）得分：(\d+(?:\.\d+)?)",
        "pathos_reason": r"pathos（情感）得分：\d+(?:\.\d+)?，评分说明：([^\n]+)",
        "ethos_score": r"ethos（来源可信度）得分：(\d+(?:\.\d+)?)",
        "ethos_reason": r"ethos（来源可信度）得分：\d+(?:\.\d+)?，评分说明：([^\n]+)",
        "logos_score": r"logos（逻辑性）得分：(\d+(?:\.\d+)?)",
        "logos_reason": r"logos（逻辑性）得分：\d+(?:\.\d+)?，评分说明：([^\n]+)",
        "total_score": r"本段总说服力得分：(\d+(?:\.\d+)?)"
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = ""

    if not result.get("total_score"):
        try:
            s1 = float(result.get("pathos_score", 0))
            s2 = float(result.get("ethos_score", 0))
            s3 = float(result.get("logos_score", 0))
            result["total_score"] = str(round((s1 + s2 + s3) / 3, 1))
        except:
            result["total_score"] = "0.0"
    return result


def annotate_text(text: str, retry=3) -> dict:
    user_prompt = SYSTEM_PROMPT.format(text=text)
    for attempt in range(retry):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            output = completion.choices[0].message.content.strip()
            print(f"\n[DEBUG] 模型原始输出:\n{output}\n{'-'*50}")
            parsed = parse_model_output(output)

            # 拼接三个理由作为总说服力理由
            reasons = []
            if parsed.get("pathos_reason"):
                reasons.append(f"情感：{parsed['pathos_reason']}")
            if parsed.get("ethos_reason"):
                reasons.append(f"可信度：{parsed['ethos_reason']}")
            if parsed.get("logos_reason"):
                reasons.append(f"逻辑：{parsed['logos_reason']}")
            total_reason = "；".join(reasons) if reasons else ""

            return {
                "情感维度得分": parsed.get("pathos_score", "5.0"),
                "情感维度得分理由": parsed.get("pathos_reason", ""),
                "来源可信度得分": parsed.get("ethos_score", "5.0"),
                "来源可信度得分理由": parsed.get("ethos_reason", ""),
                "逻辑性维度得分": parsed.get("logos_score", "5.0"),
                "逻辑性维度得分理由": parsed.get("logos_reason", ""),
                "总说服力得分": parsed.get("total_score", "5.0"),
                "总说服力得分理由": total_reason
            }
        except Exception as e:
            print(f"错误 (尝试 {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(2)
            else:
                return {
                    "情感维度得分": "5.0",
                    "情感维度得分理由": f"解析失败: {str(e)}",
                    "来源可信度得分": "5.0",
                    "来源可信度得分理由": f"解析失败: {str(e)}",
                    "逻辑性维度得分": "5.0",
                    "逻辑性维度得分理由": f"解析失败: {str(e)}",
                    "总说服力得分": "5.0",
                    "总说服力得分理由": "解析失败"
                }


def main():
    input_file = "speeches_GPT-5.1-精英人士.xlsx"
    output_file = "annotated_speeches.xlsx"

    df = pd.read_excel(input_file, sheet_name="Sheet1")
    print(f"共读取 {len(df)} 条记录")

    new_columns = [
        "情感维度得分", "情感维度得分理由",
        "来源可信度得分", "来源可信度得分理由",
        "逻辑性维度得分", "逻辑性维度得分理由",
        "总说服力得分", "总说服力得分理由"
    ]
    for col in new_columns:
        df[col] = None

    # 逐条处理（移除了break，会处理全部）
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="标注进度"):
        text = row["发言文本"]
        if pd.isna(text) or not str(text).strip():
            continue

        result = annotate_text(str(text))
        for col in new_columns:
            df.at[idx, col] = result.get(col, "")


    df.to_excel(output_file, index=False)
    print(f"标注完成，结果保存至 {output_file}")


if __name__ == "__main__":
    main()