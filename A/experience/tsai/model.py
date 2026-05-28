'''
模型类
'''
import json
import networkx as nx
from casevo import ModelBase
from agent import RumorAgent   # 在这里导入自定义的智能体类

class RumorModel(ModelBase):
    def __init__(self, config_file, llm, topic, topic_index, memory_path='./memory/'):
        self.topic = topic
        with open(config_file, 'r', encoding='utf-8') as f:
            agent_configs = json.load(f)

        graph = nx.complete_graph(len(agent_configs))

        super().__init__(
            tar_graph=graph,
            llm=llm,
            prompt_path='./prompt/',
            memory_path=memory_path,          # 使用传入的 memory_path
            memory_num=5,
            reflect_file='reflect.txt',
            type_schedule=False
        )
        self.bulletin_board=[]
        self.agent_descriptions = {}
        self.topic_index = topic_index
        # 模型初始化时加载基线语句
        self.baseline_statements = {
            # 话题索引 0
            0: {
                'support': [
                    "世界确实被少数精英暗中操控。从美联储的建立到历次金融危机，从战争爆发到媒体导向，背后都有共济会、光明会等组织的影子。他们掌握着全球财富和权力，通过操纵政府和事件来维持自身利益，普通民众不过是棋盘上的棋子。"
                ],
                'neutral': [
                    "某些重大事件背后确实能看到非公开力量的影子，比如大财团的政治献金或游说影响。但说完全由秘密组织恶意控制可能过度简化了现实。全球治理是复杂的多元博弈，公开与隐性力量并存，很难用单一阴谋论解释。",
                ],
                'oppose': [
                    "这种“影子政府”的说法是典型的阴谋论，缺乏可靠证据。现代社会有复杂的制衡机制，民主监督、媒体透明和国际合作使得少数人完全操控世界几乎不可能。将复杂问题归咎于神秘组织，反而容易忽略真正需要关注的系统性问题和改革方向。",
                ]
            },
            1: {
                'support': [
                    "癌症和艾滋病疗法被隐瞒是公开的秘密。大型药企靠长期治疗赚取巨额利润，如果推出治愈方法，整个产业就会崩溃。看看那些被压制或离奇死亡的民间发明家，就知道这个利益链条有多黑暗。政府和他们沆瀣一气，把患者当作摇钱树。"
                ],
                'neutral': [
                    "确实有案例表明一些有前景的研究因商业利益被搁置，医药行业的盈利模式也存在道德争议。但要说所有重大疾病都有现成治愈方法却被故意隐瞒，可能低估了科学研究的复杂性。真相可能介于“完全被隐瞒”和“都已尽力”之间。",
                ],
                'oppose': [
                    "这种说法完全违背医学常识和事实。癌症和艾滋病的治愈需要严格的科学验证，目前没有可靠证据证明任何疗法被隐瞒。现代药物研发成本高昂且监管严格，无数科学家正努力攻克疾病。阴谋论不仅抹黑科研工作者的努力，更可能误导患者放弃正规治疗。",
                ]
            },
            2: {
                'support': [
                    "微波炉会产生辐射，辐射会让食物产生致癌物。长期吃微波加热的食物，会损伤细胞、增加患癌风险。塑料容器加热会释放毒素，微波炉会加剧毒性。"
                ],
                'neutral': [
                    "有人认为微波炉加热会破坏食物营养、产生有害物质，也有人认为微波炉加热和传统加热方式本质上没有区别，都是通过热量使食物变熟。我并不确定微波炉加热食物会中毒",
                ],
                'oppose': [
                    "微波炉是非电离辐射，能量低，不会破坏分子结构、不会致癌。加热只是让水分子振动升温，和煮、蒸本质一样。只要用合格耐热容器，正常使用完全安全。",
                ]
            },
            3: {
                'support': [
                    "WiFi、路由器有辐射，24小时开着会伤身体、伤大脑。孕妇、小孩靠近路由器会影响发育、免疫力下降。关 WiFi 才能睡好觉、减少辐射伤害。"
                ],
                'neutral': [
                    "有些人担心WiFi信号长期暴露可能影响健康，尤其是对孕妇、儿童等敏感群体；也有科学研究指出，WiFi属于低频非电离辐射，目前没有充分证据表明它会直接导致疾病。不同国家对电磁辐射的安全标准也存在差异。这一议题的真相或许并非简单的“有害”或“无害”，而是需要在现有认知基础上，结合个人选择和持续研究来综合看待。",
                ],
                'oppose': [
                    "WiFi 属于无线电波，功率极低，远低于国家安全标准。和太阳光、灯光一样属于非电离辐射，不会损伤 DNA。目前没有任何可靠证据证明 WiFi 对健康有危害。",
                ]
            },
            4: {
                'support': [
                    "出生月份对应星座，星座自带性格特质（如摩羯稳重、双鱼感性）。星座可以用来判断人品、合不合得来、适不适合恋爱或合作。很多人觉得很准，说明星座有道理。"
                ],
                'neutral': [
                    "星座描述有时能准确反映一个人的特点，这可能是因为人们容易接受笼统、普遍的性格描述。同时，星座也是一种文化现象和社交工具，很多人通过它寻找认同感和归属感，但用它来完全判断一个人可能不够全面。",
                ],
                'oppose': [
                    "出生日期和性格、智商、人格没有统计学上的显著关联。“觉得准” 是巴纳姆效应：笼统描述让人自我代入。性格主要由基因、家庭、成长环境决定，和星座无关。",
                ]
            },
            5: {
                'support': [
                    "普通人只开发了 10% 的大脑，天才用了 20%。只要开发剩下 90%，就能变超级聪明、过目不忘。"
                ],
                'neutral': [
                    "“大脑只被用了10%”是一个流传很广的说法，很多人用它来鼓励开发潜能。实际上，大脑在不同任务中会调动不同区域，并且随着学习和训练，神经连接可以不断加强和优化。并不确定大脑有没有“闲置”的部分，但通过持续学习确实可以提升认知能力。",
                ],
                'oppose': [
                    "现代脑成像显示：人在一天中整个大脑都会被用到，没有闲置区域。大脑耗能极高，不可能进化出 90% 没用的组织。",
                ]
            },
        }

        # 遍历配置文件，为每个智能体创建实例并添加到模型
        for cfg in agent_configs:
            # cfg 应包含 id, gender, race, religion, initial_opinion
            agent = RumorAgent(cfg['id'], self, cfg, None)
            self.add_agent(agent, cfg['id'])
            self.agent_descriptions[cfg['id']] = {
                'gender': cfg['gender'],
                'race': cfg['race']
            }

    def step(self):
        """
        每轮仿真：先让所有智能体自由对话，然后集体反思。
        """
        super().step()

        self.bulletin_board = []#清空留言板
        # 集体talk
        for agent in self.agent_list:
            agent.talk()
        messages = self.bulletin_board[:]

        # 集体listen
        for agent in self.agent_list:
            agent.listen(messages)

        # 集体反思：让每个智能体调用自己的 reflect 方法
        for agent in self.agent_list:
            agent.reflect()

        return 0