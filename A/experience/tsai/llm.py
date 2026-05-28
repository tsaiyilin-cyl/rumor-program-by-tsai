'''
大模型接口，继承 casevo.LLM_INTERFACE
适配 Deepseek API（对话部分），嵌入部分使用 sentence-transformers 模型
'''
import requests
import os
from casevo import LLM_INTERFACE
from sentence_transformers import SentenceTransformer

class DeepseekLLM(LLM_INTERFACE):
    def __init__(self, api_key=None):
        """
        初始化 Deepseek 接口
        :param api_key: API 密钥，如果为 None 则从环境变量 DEEPSEEK_API_KEY 读取
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Deepseek API key is required. Set it as argument or via DEEPSEEK_API_KEY environment variable.")

        self.chat_url = "https://api.deepseek.com/v1/chat/completions"
        # 使用本地嵌入模型，不再需要嵌入 API
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量且效果不错的模型

    def send_message(self, prompt, json_flag=False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False
        }
        response = requests.post(self.chat_url, headers=headers, json=data)
        result = response.json()
        if json_flag:
            return result
        # 解析返回格式（兼容 OpenAI 风格）
        return result['choices'][0]['message']['content']

    def send_embedding(self, text_list):
        """
        将文本列表转换为向量，使用本地 sentence-transformers 模型。
        """
        # 确保输入是列表
        if isinstance(text_list, str):
            text_list = [text_list]
        embeddings = self.embed_model.encode(text_list).tolist()
        return embeddings

    def get_lang_embedding(self):
        """
        返回一个可调用对象，用于 ChromaDB 自动计算嵌入。
        参数名必须为 input，与 ChromaDB 要求一致。
        """
        class EmbeddingFunction:
            def __init__(self, embed_model):
                self.embed_model = embed_model
            def __call__(self, input):
                # input 是文本列表，直接调用 encode
                return self.embed_model.encode(input).tolist()
            def embed_documents(self, texts):
                return self.embed_model.encode(texts).tolist()
            def embed_query(self, text):
                return self.embed_model.encode([text])[0].tolist()
            def name(self):
                return "sentence_transformer_embedding_function"
        return EmbeddingFunction(self.embed_model)