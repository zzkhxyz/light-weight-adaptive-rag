from __future__ import annotations

import asyncio


class CPUModels:
    def __init__(self, en_name: str, ru_name: str, rerank_name: str):
        self.names = {"en": en_name, "ru": ru_name}
        self.rerank_name = rerank_name
        self._embedders = {}
        self._reranker = None
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str], language: str) -> list[list[float]]:
        async with self._lock:
            if language not in self._embedders:
                from sentence_transformers import SentenceTransformer
                self._embedders[language] = await asyncio.to_thread(SentenceTransformer, self.names[language], device="cpu")
            model = self._embedders[language]
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        async with self._lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = await asyncio.to_thread(CrossEncoder, self.rerank_name, device="cpu")
            model = self._reranker
        scores = await asyncio.to_thread(model.predict, [(query, text) for text in texts], show_progress_bar=False)
        # MS MARCO scores are logits; sigmoid gives a bounded routing score, not a calibrated probability.
        import math
        return [1 / (1 + math.exp(-float(score))) for score in scores]
