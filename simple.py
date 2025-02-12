import os
from datetime import datetime
from langchain_ollama import OllamaLLM  # 使用新的 OllamaLLM 类

# Initialize Ollama with local deepseek-r1:7b model
llm = OllamaLLM(
    model="deepseek-r1:7b",  # 本地模型名称
    ollama_base_url="http://localhost:11434",  # Ollama 的本地监听地址
    num_ctx=2**14
)

prompt = f"""你是一个剑桥大学网络空间安全领域的专家,
当前的时间是: "{datetime.now()}",

之前的对话历史: "",

我的问题是: "什么是SQL注入攻击",

考虑上述对话历史以及我的问题，你可以向搜索引擎提出一些问题来补充所需信息，请以列表的形式返回你的疑问。
如果历史对话中已经包含足够信息，你可以返回空列表。"""

print(f"Your prompt is: {prompt}")
response = llm.invoke(prompt)
print(response)