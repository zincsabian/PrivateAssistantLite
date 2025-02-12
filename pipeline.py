## standard lib

import logging
from typing import List, Dict, Tuple
import os
import torch
import json
import re
from datetime import datetime
from dotenv import load_dotenv

## 1st lib
from search import WebSearcher

## 3rd lib
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


class ChatMessage:
    def __init__(self, role: str, content: str, context: str = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.context = context  # 存储回答这个问题时使用的上下文


class QAPipeline:
    def __init__(self, log_level: str = "INFO", history_limit: int = 5):
        self.logger = self._get_logger(log_level)
        self.logger.info("Initializing QA Pipeline")
        self.history: List[ChatMessage] = []
        self.history_limit = history_limit

        try:
            load_dotenv()
            self.logger.info("Loaded environment variables")

            # Initialize LLM
            self.logger.info("Initializing LLM")
            # self.llm = BaseChatOpenAI(
            #     model='deepseek-chat',
            #     openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
            #     openai_api_base='https://api.deepseek.com',
            #     max_tokens=8192
            # )
            self.llm = OllamaLLM(
                model="deepseek-r1:7b",  # 本地模型名称
                ollama_base_url="http://localhost:11434",  # Ollama 的本地监听地址
                num_ctx=8192,
            )

            # Initialize web searcher
            self.logger.info("Initializing web searcher")
            self.searcher = WebSearcher()

            # Initialize embeddings model
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.logger.info(f"Using {device} to inference")
            self.logger.info("Initializing embeddings model")

            self.embeddings = HuggingFaceEmbeddings(
                model_name="moka-ai/m3e-base",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Initialize text splitter
            self.logger.info("Initializing text splitter")
            self.text_splitter = TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                disallowed_special=(),  # 禁用所有特殊标记检查
            )

            # Initialize prompt templates
            self.logger.info("Setting up prompt templates")
            self.search_prompt = PromptTemplate(
                input_variables=["query", "time", "history"],
                template="""你是一个剑桥大学网络空间安全领域的专家,
当前的时间是: "{time}",
之前的对话历史: "{history}",
我的问题是: "{query}",
考虑上述对话历史以及我的问题，你可以向搜索引擎提出一些问题来补充所需信息，请以列表的形式返回你的疑问。
""",
            )

            self.qa_prompt = PromptTemplate(
                input_variables=["current_context", "query", "time", "history"],
                template="""你是一个剑桥大学网络空间安全领域的专家

当前的时间是: "{time}"

之前的对话历史: "{history}"

当前问题的相关资料: "{current_context}"

我的问题是: "{query}"

请基于上述所有信息提供一个连贯的回答。如果历史对话中的信息与当前问题相关，请一并考虑。
""",
            )

            self.vectorstore = None
            self.logger.info("QA Pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize QA Pipeline: {str(e)}")
            raise

    def format_prompt_for_log(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """Format prompt template with given arguments for logging"""
        try:
            return prompt_template.template.format(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to format prompt for logging: {str(e)}")
            return f"Error formatting prompt: {str(e)}"

    def _log_pretty_json(self, data: dict, message: str):
        """Helper method to log dictionaries in a pretty format"""
        self.logger.debug(
            f"{message}:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _get_logger(log_level: str) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicate logs
        if logger.handlers:
            logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def extract_search_queries(self, text: str) -> List[str]:
        """
        Extract questions from numbered sections and their bullet points.
        Each numbered section is treated as a chunk of related questions.
        """
        self.logger.debug(f"Extracting search queries from text: {text}")
        chunks = []
        current_chunk = []
        current_section = None

        try:
            # 将文本按行分割
            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                question = re.sub(r"^\s*-\s*", "", line)
                question = re.sub(r"\*\*", "", question)

                section_match = re.match(r"^\d+\..*", line)
                if section_match:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                    current_chunk.append(question)
                elif current_chunk:
                    current_chunk.append(question)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            self.logger.debug(f"Extracted questions from LLM: {chunks}")
            return chunks

        except Exception as e:
            self.logger.error(f"Failed to extract search queries: {str(e)}")
            raise

    def search_and_process(self, search_queries: List[str]) -> None:
        """Execute searches and process results into vector store"""
        self.logger.info(f"Processing {len(search_queries)} search queries")
        all_texts = []

        if len(search_queries) == 0:
            return None

        try:
            # Search and collect results
            for i, query in enumerate(search_queries, 1):
                self.logger.debug(f"Searching query {i}/{len(search_queries)}: {query}")
                results = self.searcher.search(query)
                self.logger.debug(
                    f"Search results for query {i}:\n"
                    "----------------------------------------"
                )

                for url, content in results.items():
                    self.logger.debug(f"URL: {url}")
                    self.logger.debug(f"Content preview: {content[:500]}...\n")

                for url, content in results.items():
                    chunks = self.text_splitter.split_text(content)
                    chunks = [f"Source: {url}\n\n{chunk}" for chunk in chunks]
                    all_texts.extend(chunks)

            # Create or update vector store
            self.logger.debug(f"Adding {len(all_texts)} text chunks to vector store")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_texts(all_texts, self.embeddings)
            else:
                self.vectorstore.add_texts(all_texts)

        except Exception as e:
            self.logger.error(f"Failed in search and process step: {str(e)}")
            raise

    def _format_history(self, max_tokens: int = 2000) -> str:
        """格式化对话历史和相关上下文"""
        formatted_history = []
        # history_contexts = []

        for msg in self.history[-self.history_limit :]:
            if msg.role == "user":
                formatted_history.append(f"Human: {msg.content}")
            else:
                formatted_history.append(f"Assistant: {msg.content}")
                # if msg.context:
                #     history_contexts.append(msg.context)

        # 使用text splitter确保不超出token限制
        text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
        history_text = "\n".join(formatted_history)
        # contexts_text = "\n---\n".join(history_contexts)

        if len(history_text) > 0:
            history_text = text_splitter.split_text(history_text)[0]
        # if len(contexts_text) > 0:
        #     contexts_text = text_splitter.split_text(contexts_text)[0]

        return history_text

    def answer_question(self, query: str) -> str:
        """Enhanced answer_question method with chat history support"""
        self.logger.info(f"Processing question in chat context: {query}")

        try:
            # 获取格式化的历史对话
            history_text = self._format_history()

            self.logger.debug(f"history text: {history_text}")
            # self.logger.debug(f"history_contexts: {history_contexts}")

            # 生成搜索查询
            self.logger.info("=== Search Phase ===")
            chain = self.search_prompt | self.llm | StrOutputParser()
            search_response = chain.invoke(
                {"query": query, "time": datetime.now(), "history": history_text}
            )

            search_queries = self.extract_search_queries(search_response)

            # 搜索和处理结果
            self.search_and_process(search_queries)

            # 检索相关上下文
            if self.vectorstore is None:
                current_context = "No information available."
            else:
                relevant_docs = self.vectorstore.similarity_search(query, k=5)
                current_context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )

            self.logger.info(current_context)

            # 生成最终答案
            chain = self.qa_prompt | self.llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "current_context": current_context,
                    "query": query,
                    "time": datetime.now(),
                    "history": history_text,
                    # "history_contexts": history_contexts
                }
            )

            # 更新对话历史
            self.history.append(ChatMessage("user", query))
            self.history.append(ChatMessage("assistant", answer, current_context))

            # 如果历史记录过长，删除最早的记录
            while (
                len(self.history) > self.history_limit * 2
            ):  # *2是因为每轮对话有两条消息
                self.history.pop(0)

            return answer

        except Exception as e:
            self.logger.error(f"Failed to answer question: {str(e)}")
            raise


if __name__ == "__main__":
    # 初始化pipeline
    pipeline = QAPipeline(log_level="INFO", history_limit=5)

    # 第一轮对话
    answer1 = pipeline.answer_question("什么是SQL注入攻击？")

    print(answer1)

    # 第二轮对话
    answer2 = pipeline.answer_question("如何预防这种攻击？")

    print(answer2)
