from search import WebSearcher
from dotenv import load_dotenv

load_dotenv()
searcher = WebSearcher()
query = "Python编程"
date_restrict = 3  # 3天内结果
scraped = searcher.search_and_scrape(query, date_restrict)
for url, content in scraped.items():
    print(f"{url}:\n{content}\n")