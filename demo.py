from search import WebSearcher
from dotenv import load_dotenv

load_dotenv()
searcher = WebSearcher()
query = "fix memory leak in cpp"
scraped = searcher.search_and_scrape(query)
for url, content in scraped.items():
    print(f"URL: {url}:\ncontent: {content[:500]}\n")
