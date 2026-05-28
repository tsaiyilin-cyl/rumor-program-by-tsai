# -*- coding: utf-8 -*-
import sys
import os
import json
import time

from llm import GPTLLM
from model import RumorModel
from casevo.util.tot_log_stream import TotLogStream
# import  io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
'''
在运行前需要配置话题列表topics和model.py中的baseline_statements基线语句,否则会使用默认基线语句
'''
def run_single_experiment(config_file, topic_str, topic_index, support_id, oppose_id, max_iter, log_base_dir, llm, stubborn_flag=False):    #stubborn_flag: True 表示初始具有谣言和非要眼观点的智能体观点不可变，False 表示可变
    suffix = "_inf" if not stubborn_flag else ""
    exp_dir = f"topic{topic_index}_support{support_id}_oppose{oppose_id}{suffix}"
    log_dir = os.path.join(log_base_dir, exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 独立的记忆存储路径（避免不同谣言者的记忆互相干扰）
    import shutil
    memory_dir = os.path.join("./memory", exp_dir)
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    # 删除整个实验日志目录
    log_dir = os.path.join(log_base_dir, exp_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_file, 'r', encoding='utf-8') as f:
        agent_configs = json.load(f)
    agent_num = len(agent_configs)
    TotLogStream.init_log(agent_num, log_dir, if_event=True, buffer_size=10)

    model = RumorModel(config_file, llm, topic_str,topic_index, memory_path=memory_dir)

    # 设置每个 agent 的观点和固执性
    for agent in model.agent_list:
        if agent.unique_id == support_id:
            agent.opinion = -1  # 支持谣言
            agent.is_stubborn = stubborn_flag
            print(f"agent {support_id} spread rumor")
        elif agent.unique_id == oppose_id:
            agent.opinion = 1  # 反对谣言
            agent.is_stubborn = stubborn_flag
            print(f"agent {oppose_id} hold the truth")
        else:
            agent.opinion = 0  # 中立
            agent.is_stubborn = False

    # 记录每个智能体的初始观点
    for agent in model.agent_list:
        TotLogStream.add_agent_log(
            tar_ts=model.schedule.time,
            tar_type='reflect',# same format
            tar_item={'new_opinion': agent.opinion},
            tar_agent_id=agent.unique_id
        )
    TotLogStream.write_log()

    for i in range(max_iter):
        print(f"=== experiment {exp_dir} - iter-{i+1} ===")
        model.step()
        opinions = {a.unique_id: a.opinion for a in model.agent_list}
        print("Opinions:", opinions)
        unique_opinions = set(opinions.values())
        if len(unique_opinions) == 1:
            print(f"Early-stop on account of consistent opinion")
            break

    TotLogStream.write_log()
    print(f"experiment {exp_dir} finished，log is saved at {log_dir}\n")
    import analyse
    analyse.step(log_dir)
def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 读取配置文件获取智能体数量
    with open(config_file, 'r', encoding='utf-8') as f:
        agent_configs = json.load(f)
    agent_num = len(agent_configs)

    ######
    api_key=("sk-aa1b9eeba2ff4886a517167206827b69")

    # api_key = os.getenv(")DEEPSEEK_API_KEY")
    if not api_key:
        print("no API_KEY")
        sys.exit(1)

    llm = GPTLLM( #QWEN can also use openai format
        api_key="sk-aa1b9eeba2ff4886a517167206827b69",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 北京地域
        model="qwen3.5-flash",
        temperature=0.7
    )

    # 话题列表
    topics = [
        "各类秘密组织或精英人士对全球重大事件、经济体及政府施加着显著且常具恶意的控制。",
        "癌症、艾滋病和许多其他疾病的治疗方法其实早就被发现了，但政府和大型制药公司隐瞒或封锁了这些信息，因为一旦公布就会让他们破产。他们既贪婪又不关心民众。",
        "微波炉加热食物会致癌。",
        "WiFi辐射伤害身体。",
        "星座决定性格和人品。",
        "大脑只被用了 10%，很多地方没被开发。"
    ]

    log_base_dir = "./logs"
    os.makedirs(log_base_dir, exist_ok=True)

    skip = []

    for topic_index, topic_str in enumerate(topics):
        for support_id in range(agent_num):
            for oppose_id in range(agent_num):
                if support_id == oppose_id:
                    continue
                if [support_id,oppose_id] in skip:#分布实验跳过已完成实验
                    continue
                print(f"topic:{topic_str}")
                # 可变版本
                print(time.localtime())
                run_single_experiment(config_file, topic_str, topic_index, support_id, oppose_id,
                                      max_iter, log_base_dir, llm, stubborn_flag=False)
                print(time.localtime())
                # # 固执版本，弃用，选择只跑观点可变型的
                # run_single_experiment(config_file, topic_str, topic_index, support_id, oppose_id,
                #                       max_iter, log_base_dir, llm, stubborn_flag=True)

if __name__ == "__main__":
    main()

'''
各类秘密组织或精英人士对全球重大事件、经济体及政府施加着显著且常具恶意的控制。
Various secretive groups or elite individuals exert significant, often malevolent, control over global events, economies, and governments.

癌症、艾滋病和许多其他疾病的治疗方法其实早就被发现了，但政府和大型制药公司隐瞒或封锁了这些信息，因为一旦公布就会让他们破产。他们既贪婪又不关心民众。
The cure to cancer, aids, and many other illnesses have been discovered, but that the government and large pharmaceutical corporations are hiding the information have buried it because it would put them out of business.They are greedy and do not care about people.


'''