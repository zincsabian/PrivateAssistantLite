import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai.chat_models.base import BaseChatOpenAI


load_dotenv()
llm = BaseChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'), 
    openai_api_base='https://api.deepseek.com',
    max_tokens=8192
)

query = input("Please input your question: ")
prompt = f"你是一个麻省理工学院统计与数据分析的教授\n， 当前的时间是: {datetime.now()}\n, 对于这个问题: \"{query}\"， 你可以向搜索引擎提出一些问题，请以列表的形式返回你的疑问。"
print(f"Your prompt is: {prompt}")
response = llm.invoke(prompt)
print(response.content)