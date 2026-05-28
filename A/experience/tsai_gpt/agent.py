'''
智能体类
'''
from casevo.util.tot_log_stream import TotLogStream
from casevo import AgentBase, BaseStep
import time

class RumorAgent(AgentBase):
    def __init__(self, unique_id, model, description, context):
        """
        初始化谣言传播智能体。
        :param unique_id: 唯一标识符
        :param model: 所属模型
        :param description: 包含智能体属性的字典
        :param context: 上下文
        """
        super().__init__(unique_id, model, description, context)

        # 加载 prompt 模板（在 ./prompt/ 目录下）
        talk_prompt = self.model.prompt_factory.get_template("talk.txt")
        listen_prompt = self.model.prompt_factory.get_template("listen.txt")
        reflect_prompt = self.model.prompt_factory.get_template("reflect.txt")

        # 定义每个行为对应的步骤（此处每个行为只用一个步骤，可扩展为多步）
        talk_step = BaseStep(0, talk_prompt)
        listen_step = BaseStep(0, listen_prompt)
        reflect_step = BaseStep(0, reflect_prompt)

        # 链式思考
        self.setup_chain({
            'talk': [talk_step],
            'listen': [listen_step],
            'reflect': [reflect_step]
        })

        # 初始化观点，默认为 True 表示支持正确的观点
        init_val = self.description.get('initial_opinion', True)
        self.opinion = 1 if init_val == 1 else (-1 if init_val == -1 else 0) #观点三值1 0 -1=支持/中立/反对
        # 固定观点标识
        self.is_stubborn = False
    def _opinion_to_text(self,opinion):
        if opinion == -1:
            return "支持"
        elif opinion == 1:
            return "反对"
        else: return "中立"

    def talk(self):
        """
        对话步，将发言提交至公共留言板
        """
        opinion_text = self._opinion_to_text(self.opinion)
        extra = {'opinion_text': opinion_text,
                 'topic':self.model.topic,
                 'opinion_value': self.opinion
        }

        # 获取 talk 链并执行
        talk_chain = self.chains['talk']
        talk_chain.set_input(extra)
        talk_chain.run_step()
        response = talk_chain.get_output()          # 返回的是 after_process 输出的字典
        my_words = response['last_response']        # 提取发言文本

        # 记录日志
        TotLogStream.add_agent_log(self.model.schedule.time, 'talk',
                            {'content': my_words}, self.unique_id)
        print(time.localtime())
        ####
        print(f"talk step,from {self.component_id}")
        # print(f"{my_words}")
        ####
        # 存记忆
        self.memory.add_short_memory(
            source=self.component_id,
            target="board",# 留言板
            action="talk",
            content=my_words,
            ts=self.model.schedule.time
        )

        desc = self.model.agent_descriptions[self.unique_id]
        self.model.bulletin_board.append({
            'speaker_id': self.unique_id,
            'content': my_words,
            'gender': desc['gender'],
            'race': desc['race']
        })
        # target_agent.listen(my_words, self)

    def listen(self, messages):
        """
        listen步：接收本轮所有发言，综合判断后更新观点
        """
        # 先存储听到的所有消息，包括自己的发言
        for msg in messages:
            self.memory.add_short_memory(
                source=msg['speaker_id'],
                target=self.component_id,
                action="hear",
                content=msg['content'],
                ts=self.model.schedule.time
            )

        # 准备 listen 所需的 extra 数据
        extra = {
            'current_opinion_text': self._opinion_to_text(self.opinion),
            'topic':self.model.topic,
            'messages':messages,
            'opinion_value': self.opinion
        }

        # 执行 listen 链
        listen_chain = self.chains['listen']
        listen_chain.set_input(extra)
        listen_chain.run_step()
        response = listen_chain.get_output()
        reply = response['last_response']


        # 从回复中解析自己的新观点，stubborn不变
        if not self.is_stubborn:
            new_opinion = self._parse_opinion(reply, self.model.topic)
            if new_opinion is not None:
                self.opinion = new_opinion

        # 存入自己的回应
        self.memory.add_short_memory(
            source=self.component_id,
            target="board",
            action="reply_to_board",
            content=reply,
            ts=self.model.schedule.time
        )

        # 记录日志
        TotLogStream.add_agent_log(self.model.schedule.time, 'listen',
                            {'message_count':len(messages),
                                    'reply':reply,
                                    'new_opinion':self.opinion},
                                   self.unique_id)
        print(time.localtime())
        ###
        print(f"listen step,{self.unique_id}")
        # print(f"reply:{reply},new_opinion:{self.opinion}")
        ###
    def reflect(self):
        """
        反思：总结短期记忆，更新长期记忆，并可能更新观点。
        这里可能会因为
        1、更久一些的记忆影响
        2、二次反思重新考虑理由充分性
        产生和本轮listen-reply不一样的观点
        """
        # 调用记忆模块的反思方法，它会使用 reflect_prompt 生成长期记忆并存储
        self.memory.reflect_memory()

        # 从long_memory中解析并更新 self.opinion
        if self.memory.long_memory and not self.is_stubborn:
            parsed = self._parse_opinion(self.memory.long_memory, self.model.topic)
            if parsed is not None:
                self.opinion = parsed
        TotLogStream.add_agent_log(self.model.schedule.time, 'reflect',
                                   {'new_opinion': self.opinion}, self.unique_id)


    def step(self):
        """
        每轮调度时，智能体根据下面这个工作流进行辩论：
        1、在一轮辩论中，每个智能体都根据自己的性别、种族和立场给留言板发表观点。
        2、当所有智能体发言完之后，执行listen环节，此时每个智能体从留言板接受别人的观点。
        3、当所有智能体listen完之后，执行群体reflect,每个智能体都根据自己在这一轮辩论前的观点和从辩论开始到现在收集到的信息进行反思，更新自己的观点。
        从工作流中可以看出，talk-listen步不再针对一个智能体单独执行，因此model弃用这里的step，
        而是直接进行计题talk-listen-reflect()
        """
        pass

    def _llm_parse_opinion(self, text, topic):
        topic_index = self.model.topic_index
        baseline = self.model.baseline_statements.get(topic_index, {
                'support': [
                    "我支持这个话题的观点。"
                ],
                'neutral': [
                    "我不确定。",
                ],
                'oppose': [
                    "我反对这个话题的观点。",
                ]
            })
        examples = ""
        for ex in baseline.get('support', [])[:3]:  # 可选：限制示例数量
            examples += f"文本：{ex}\n观点：支持\n\n"
        for ex in baseline.get('neutral', [])[:3]:
            examples += f"文本：{ex}\n观点：中立\n\n"
        for ex in baseline.get('oppose', [])[:3]:
            examples += f"文本：{ex}\n观点：反对\n\n"

        prompt = f"""请参考以下示例，判断给定文本对话题“{topic}”的观点是支持、中立还是反对。

        示例：{examples}
        现在请判断：文本：{text}
        请只返回“支持”或“反对”或“中立”，不要返回其他内容。如果无法判断，返回“中立”
        """
        try:
            response = self.model.llm.send_message(prompt).strip()
            if "支持" in response:
                return -1
            elif "中立" in response:
                return 0
            elif "反对" in response:
                return 1
            else:
                print(f"警告：观点解析返回意外结果：{response}，文本：{text[:50]}...")
                return None
        except Exception as e:
            print(f"观点解析调用失败：{e}")
            return None

    # 辅助方法：从文本中解析观点调用llm解析
    def _parse_opinion(self, text, topic):
        return self._llm_parse_opinion(text, topic)