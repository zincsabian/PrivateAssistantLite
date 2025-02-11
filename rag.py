import logging
from typing import List, Dict
import os
import torch
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
            
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            self.logger.info(f"Using {device} to embedding...")

            # Initialize embeddings model
            self.logger.debug("Initializing embeddings model")
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
                template="对于这个问题: {query}， 如果你可以联网搜索， 你会考虑搜索哪些问题， 请以列表的形式返回你的问题"
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
        """Extract numbered list items from LLM response"""
        self.logger.debug(f"Extracting search queries from text: {text}")
        queries = []
        try:
            for line in text.split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    query = line.split('.', 1)[1].strip()
                    queries.append(query)
            self.logger.debug(f"Extracted queries: {queries}")
            return queries
        except Exception as e:
            self.logger.error(f"Failed to extract search queries: {str(e)}")
            raise

    def search_and_process(self, search_queries: List[str]) -> None:
        """Execute searches and process results into vector store"""
        self.logger.info(f"Processing {len(search_queries)} search queries")
        all_texts = []
        
        try:
            # Search and collect results
            for query in search_queries:
                self.logger.debug(f"Searching for query: {query}")
                results = self.searcher.search(query)
                self.logger.debug(f"Got {len(results)} results")
                
                for url, content in results.items():
                    self.logger.debug(f"Processing content from URL: {url}")
                    chunks = self.text_splitter.split_text(content)
                    chunks = [f"Source: {url}\n\n{chunk}" for chunk in chunks]
                    all_texts.extend(chunks)
            
            # Create or update vector store
            self.logger.info(f"Adding {len(all_texts)} text chunks to vector store")
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
            self.logger.debug("Generating search queries")
            chain = self.search_prompt | self.llm | StrOutputParser()
            search_response = chain.invoke({"query": query})
            search_queries = self.extract_search_queries(search_response)
            
            # Search and process results
            self.logger.debug("Searching and processing results")
            self.search_and_process(search_queries)
            
            # Retrieve relevant contexts
            if self.vectorstore is None:
                self.logger.warning("No vector store available")
                return "No information available to answer the question."
                
            self.logger.debug("Retrieving relevant documents")
            relevant_docs = self.vectorstore.similarity_search(query, k=10)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate final answer
            self.logger.debug("Generating final answer")
            chain = self.qa_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "query": query})
            
            self.logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to answer question: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        pipeline = QAPipeline(log_level="DEBUG")
        
        # Example question
        question = "什么是向量数据库？它有什么优势？"
        
        print("Processing question:", question)
        answer = pipeline.answer_question(question)
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")