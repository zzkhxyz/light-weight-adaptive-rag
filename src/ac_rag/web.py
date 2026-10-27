from __future__ import annotations

import httpx

from .models import Chunk
from .router import language_of
from .store import point_id


class TavilySearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str) -> list[Chunk]:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "search_depth": "basic", "max_results": 5},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        return [
            Chunk(id=point_id(item["url"], 0), source=item["url"], text=item.get("content", ""), language=language_of(query))
            for item in results if item.get("url") and item.get("content")
        ]
