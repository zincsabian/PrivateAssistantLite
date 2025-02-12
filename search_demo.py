from search import WebSearcher
from dotenv import load_dotenv

load_dotenv()
searcher = WebSearcher()
query = "如何识别一个可能的 SQL 注入漏洞？"
scraped = searcher.search(query)
for url, content in scraped.items():
    print(f"URL: {url}:\ncontent: {content[:500]}\n")
