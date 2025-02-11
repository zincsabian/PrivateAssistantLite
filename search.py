import os
import requests
import urllib.parse
from bs4 import BeautifulSoup
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeVar

class WebSearcher:
    def __init__(self):
        self.logger = self._get_logger('INFO')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        })

        self.search_api_url = os.environ.get("SEARCH_API_URL")
        self.search_api_key = os.environ.get("SEARCH_API_KEY")
        self.search_project_id = os.environ.get("SEARCH_PROJECT_KEY")

        # 如果没有配置环境变量，使用默认值
        if not self.search_api_url:
            self.search_api_url = "https://www.googleapis.com/customsearch/v1"
        if not self.search_api_key:
            self.search_api_key = "dummy-search-api-key"
        if not self.search_project_id:
            self.search_project_id = "dummy-search-project-id"

    @staticmethod
    def _get_logger(log_level: str) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def search_web(self, query: str, date_restrict: int = 0, target_site: str = "") -> List[str]:
        escaped_query = urllib.parse.quote(query)
        url_base = f"{self.search_api_url}?key={self.search_api_key}&cx={self.search_project_id}&q={escaped_query}"
        url_paras = f"&safe=active"
        if date_restrict > 0:
            url_paras += f"&dateRestrict={date_restrict}"
        if target_site:
            url_paras += f"&siteSearch={target_site}&siteSearchFilter=i"
        url = f"{url_base}{url_paras}"

        self.logger.debug(f"Searching for query: {query}")

        response = requests.get(url)
        if not response:
            raise Exception("No response from search API")

        search_results = response.json()
        if "error" in search_results:
            raise Exception(search_results["error"])

        total_results = search_results["searchInformation"].get("totalResults", 0)
        if total_results == 0:
            self.logger.warning(f"No results found for query: {query}")
            return []

        results = search_results.get("items", [])
        if not results:
            self.logger.warning(f"No result items in the response for query: {query}")
            return []

        found_links = []
        for result in results:
            link = result.get("link", "")
            if link:
                found_links.append(link)
        return found_links

    def scrape_url_content(self, url: str) -> Optional[str]:
        self.logger.info(f"Scraping {url} ...")
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "lxml", from_encoding="utf-8")
            body_tag = soup.body
            if body_tag:
                body_text = " ".join(body_tag.get_text().split()).strip()
                if len(body_text) > 100:
                    self.logger.info(f"Successfully scraped {url} with length: {len(body_text)}")
                    return body_text
                else:
                    self.logger.warning(f"Body text too short for URL: {url}, length: {len(body_text)}")
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
        return None

    def search_and_scrape(self, query: str, date_restrict: int = 0, target_site: str = "") -> Dict[str, str]:
        search_results = self.search_web(query, date_restrict, target_site)
        scraped_content = {}
        for url in search_results:
            content = self.scrape_url_content(url)
            if content:
                scraped_content[url] = content
        return scraped_content