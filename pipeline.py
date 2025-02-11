import logging
from typing import List, Dict
import os
import torch
import json
import re
from dotenv import load_dotenv
from search import WebSearcher
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class QAPipeline:
    def __init__(self, log_level: str = "INFO"):
        self.logger = self._get_logger(log_level)
        self.logger.info("Initializing QA Pipeline")
        
        try:
            load_dotenv()
            self.logger.debug("Loaded environment variables")
            
            # Initialize LLM
            self.logger.debug("Initializing LLM")
            self.llm = BaseChatOpenAI(
                model='deepseek-chat',
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                openai_api_base='https://api.deepseek.com',
                max_tokens=1024
            )
            
            # Initialize web searcher
            self.logger.debug("Initializing web searcher")
            self.searcher = WebSearcher()
            
            # Initialize embeddings model
            self.logger.debug("Initializing embeddings model")
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            
            self.logger.info(f"Using device: {device}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="moka-ai/m3e-base",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter
            self.logger.debug("Initializing text splitter")
            self.text_splitter = TokenTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Initialize prompt templates
            self.logger.debug("Setting up prompt templates")
            self.search_prompt = PromptTemplate(
                input_variables=["query"],
                template="对于这个问题: {query}， 如果你可以联网搜索， 你会考虑搜索哪些问题，不要超过5个问题，请以列表的形式返回你的问题"
            )
            
            self.qa_prompt = PromptTemplate(
                input_variables=["context", "query"],
                template="基于以下资料回答我的问题：{context}\n\n我的问题是: {query}"
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
        self.logger.debug(f"{message}:\n{json.dumps(data, ensure_ascii=False, indent=2)}")

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
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue        

                question = re.sub(r'^\s*-\s*', '', line)
                question = re.sub(r'\*\*', '', question)

                # 检查是否是新的编号部分
                section_match = re.match(r'^\d+\..*', line)
                if section_match:
                    if current_chunk:
                        # 如果已有收集的问题，保存当前chunk
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                    current_chunk.append(question)
                elif current_chunk:
                    current_chunk.append(question)
            
            # 添加最后一个chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            self.logger.debug(f"Extracted question chunks: {chunks}")
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

    def answer_question(self, query: str) -> str:
        """Complete pipeline to answer a question"""
        self.logger.info(f"Processing question: {query}")
        
        try:
            # Generate search queries
            self.logger.info("=== Search Phase ===")
            search_prompt_content = self.format_prompt_for_log(
                self.search_prompt, 
                query=query
            )
            self.logger.debug(
                "Sending search prompt to model:\n"
                "----------------------------------------\n"
                f"{search_prompt_content}\n"
                "----------------------------------------"
            )
            
            chain = self.search_prompt | self.llm | StrOutputParser()
            search_response = chain.invoke({"query": query})
            
            self.logger.debug(
                "Received search response from model:\n"
                "----------------------------------------\n"
                f"{search_response}\n"
                "----------------------------------------"
            )
            
            search_queries = self.extract_search_queries(search_response)
            self.logger.info(
                "Extracted search queries:\n"
                "----------------------------------------\n" +
                "\n".join(f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(search_queries)) +
                "\n----------------------------------------"
            )
            
            # Search and process results
            self.logger.info("=== Web Search Phase ===")
            self.search_and_process(search_queries)
            
            # Retrieve relevant contexts
            if self.vectorstore is None:
                self.logger.warning("No vector store available")
                return "No information available to answer the question."
                
            self.logger.info("=== Document Retrieval Phase ===")
            relevant_docs = self.vectorstore.similarity_search(query, k=10)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            self.logger.debug(
                "Retrieved documents context:\n"
                "----------------------------------------\n"
                f"{context[:1000]}...\n"  # Only show first 1000 chars to avoid huge logs
                "----------------------------------------"
            )
            
            # Generate final answer
            self.logger.info("=== Answer Generation Phase ===")
            qa_prompt_content = self.format_prompt_for_log(
                self.qa_prompt,
                context=context,
                query=query
            )
            self.logger.info(
                "Sending QA prompt to model:\n"
                "----------------------------------------\n"
                f"{qa_prompt_content}\n"
                "----------------------------------------"
            )
            
            chain = self.qa_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "query": query})
            
            self.logger.debug(
                "Received final answer from model:\n"
                "----------------------------------------\n"
                f"{answer}\n"
                "----------------------------------------"
            )
            
            self.logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to answer question: {str(e)}")
            raise

