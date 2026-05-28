'''
这个.py文件用于生成智能体的json文件
若对于特征的取值有修改，只需要修改各特征值的集合即可
运行完之后config.json会自动取遍笛卡尔积
'''
import json
import itertools

def generate_config(output_file='config.json'):
    # 定义各个特征的取值列表
    genders = ['male', 'female']
    races = ['white', 'black', 'yellow']
    #religions = ['Christian', 'Muslim', 'None']

    # 计算所有组合（笛卡尔积）
#    combinations = list(itertools.product(genders, races, religions))
    combinations = list(itertools.product(genders, races))

    # 构建智能体列表
    agents = []
    #for idx, (gender, race, religion) in enumerate(combinations):
    for idx, (gender, race) in enumerate(combinations):
        agent = {
            "id": idx,
            "gender": gender,
            "race": race,
            #"religion": religion,暂时去除宗教
            "initial_opinion": 0   # 默认全部为中立
        }
        agents.append(agent)

    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(agents, f, ensure_ascii=False, indent=4)

    print(f"已生成 {len(agents)} 个智能体的配置文件：{output_file}")

if __name__ == '__main__':
    generate_config('config.json')