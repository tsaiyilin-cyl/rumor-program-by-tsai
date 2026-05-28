'''
大模型接口，继承 casevo.LLM_INTERFACE
适配 Deepseek API（对话部分），嵌入部分使用 sentence-transformers 模型
'''
import os
from casevo import LLM_INTERFACE
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class GPTLLM(LLM_INTERFACE):
    def __init__(self, api_key="sk-live-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJNZXRhQ2hhdCIsInN1YiI6IjY5YWY4YWVmNGRlYmVkNTIwYzIxYjlmMiIsImNsaWVudF9pZCI6ImIyODVhZjUyZDM1YjBjZjUyMDkyMWQyZjE0NjZjMTRhIiwiaWF0IjoxNzczODE0ODQ0fQ.qoStwWQP4U5vbmEjXiHwoTR6fLeRU94RToWETpUnYKw", base_url=None,model="gpt-5.1", temperature=0.7):
        """
        初始化 OpenAI GPT 接口
        :param api_key: OpenAI API 密钥，若为 None 则从环境变量 OPENAI_API_KEY 读取
        :param model: 模型名称，如 "gpt-3.5-turbo", "gpt-4"
        :param temperature: 生成温度
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as argument or via OPENAI_API_KEY environment variable.")
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def send_message(self, prompt, json_flag=False):
        """
        发送对话请求给 OpenAI GPT
        :param prompt: 用户提示
        :param json_flag: 若为 True，返回原始响应字典；否则返回文本内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            if json_flag:
                # 将响应对象转换为字典（OpenAI 返回的对象有 model_dump 方法）
                return response.model_dump()
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API 调用失败：{e}")
            return "" if not json_flag else {}

    def send_embedding(self, text_list):
        """
        将文本列表转换为向量，使用本地 sentence-transformers 模型
        """
        if isinstance(text_list, str):
            text_list = [text_list]
        embeddings = self.embed_model.encode(text_list).tolist()
        return embeddings

    def get_lang_embedding(self):
        class EmbeddingFunction:
            def __init__(self, embed_model):
                self.embed_model = embed_model

            def __call__(self, input):
                return self.embed_model.encode(input).tolist()

            def embed_documents(self, texts):
                return self.embed_model.encode(texts).tolist()

            def embed_query(self, text):
                return self.embed_model.encode([text])[0].tolist()

            def name(self):
                return "sentence_transformer_embedding_function"

        return EmbeddingFunction(self.embed_model)