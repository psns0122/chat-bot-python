from __future__ import annotations
from typing import List
import requests
from bs4 import BeautifulSoup

REMOVE_SELECTORS = ["script", "style", "noscript", "svg", "canvas", "iframe",
                    "header", "footer", "nav", "aside", "form", "button"]

class IngestRepository:
    def __init__(self, timeout_sec: int = 10):
        self.timeout_sec = timeout_sec

    def fetch_and_parse_urls(self, urls: List[str]) -> str:
        sb: list[str] = []

        for url in urls:
            try:
                r = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=self.timeout_sec
                )
                r.raise_for_status()

                soup = BeautifulSoup(r.text, "html.parser")

                # remove tags
                for tag in REMOVE_SELECTORS:
                    for node in soup.select(tag):
                        node.decompose()

                # main 우선
                main = soup.select_one("main")
                text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)

                if text:
                    sb.append(text)

            except Exception:
                # Java 코드도 실패 시 그냥 skip 느낌이라 동일하게 무시
                continue

        result = " ".join(sb)
        result = " ".join(result.split())  # normalize whitespace
        if len(result) > 5000:
            result = result[:5000] + "...(이하 생략)"
        return result
