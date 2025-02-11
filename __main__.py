from search import WebSearcher
from dotenv import load_dotenv

load_dotenv()
searcher = WebSearcher()
query = "Python编程"
scraped = searcher.search_and_scrape(query)
for url, content in scraped.items():
    print(f"{url}:\n")
