import sys
import os
import json
from llm import DeepseekLLM          # 改为导入 DeepseekLLM
from model import RumorModel
from casevo.util.tot_log_stream import TotLogStream

def run_single_experiment(config_file, topic, rumor_id, max_iter, log_base_dir, llm):
    safe_topic = topic.replace(' ', '_').replace('?', '').replace('，', '').replace('？', '')
    log_dir = os.path.join(log_base_dir, safe_topic)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_file, 'r', encoding='utf-8') as f:
        agent_configs = json.load(f)
    agent_num = len(agent_configs)
    TotLogStream.init_log(agent_num, log_dir, if_event=True, buffer_size=10)

    model = RumorModel(config_file, llm, topic)

    for agent in model.agent_list:
        if agent.unique_id == rumor_id:
            agent.opinion = False
            agent.is_rumor_spreader = True
            print(f"智能体 {rumor_id} 被设置为谣言者（观点=False）")
        else:
            agent.opinion = True
            agent.is_rumor_spreader = False

    for i in range(max_iter):
        print(f"=== 话题 '{topic}' - 第 {i+1} 轮 ===")
        model.step()
        opinions = {a.unique_id: a.opinion for a in model.agent_list}
        print("Opinions:", opinions)

    TotLogStream.write_log()
    print(f"话题 '{topic}' 模拟结束，日志保存在 {log_dir}\n")

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    rumor_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    # 从环境变量读取 API Key（更安全）
    #####
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key = "sk-c6fac6331d7f4d03ba5408050daa136b"
    if not api_key:
        # error info
        print("请设置环境变量 DEEPSEEK_API_KEY 或在代码中直接填入 API Key")
        sys.exit(1)

    llm = DeepseekLLM(api_key=api_key)

    topics = [
        "我们应该相信地圆说而不是地平说",
    ]

    log_base_dir = "./logs"
    os.makedirs(log_base_dir, exist_ok=True)

    for topic in topics:
        run_single_experiment(config_file, topic, rumor_id, max_iter, log_base_dir, llm)

if __name__ == "__main__":
    main()
    ##运行前需要清理memory