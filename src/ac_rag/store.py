from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .models import Chunk
from .router import language_of, tokenize


def split_text(text: str, size: int = 160, overlap: int = 30) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Expected size > overlap >= 0")
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return chunks


def point_id(source: str, index: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{source}:{index}".encode(), digest_size=8).digest(), "big") & ((1 << 63) - 1)


def rrf(rankings: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    totals: dict[int, float] = defaultdict(float)
    chunks: dict[int, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            totals[chunk.id] += 1 / (k + rank)
            chunks[chunk.id] = chunk
    return [chunks[cid].model_copy(update={"score": score}) for cid, score in sorted(totals.items(), key=lambda item: item[1], reverse=True)]


class HybridStore:
    """Local Qdrant dense index plus process-local BM25 rebuilt from Qdrant at startup."""

    def __init__(self, path: Path, models_backend):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.models = models_backend
        self._lock = asyncio.Lock()
        self._chunks: dict[str, list[Chunk]] = {"en": [], "ru": []}
        self._bm25: dict[str, BM25Okapi | None] = {"en": None, "ru": None}
        self._load()

    def _load(self) -> None:
        for lang in ("en", "ru"):
            name = f"chunks_{lang}"
            if not self.client.collection_exists(name):
                continue
            offset = None
            while True:
                records, offset = self.client.scroll(name, offset=offset, limit=256, with_payload=True)
                self._chunks[lang].extend(Chunk(id=int(r.id), **r.payload) for r in records)
                if offset is None:
                    break
            self._rebuild(lang)

    def _rebuild(self, lang: str) -> None:
        rows = self._chunks[lang]
        self._bm25[lang] = BM25Okapi([tokenize(row.text) or [""] for row in rows]) if rows else None

    async def ingest(self, source: str, text: str) -> list[int]:
        segments = split_text(text)
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for i, segment in enumerate(segments):
            lang = language_of(segment)
            grouped[lang].append(Chunk(id=point_id(source, i), text=segment, source=source, language=lang))
        vectors_by_language = {
            lang: await self.models.embed([row.text for row in rows], lang)
            for lang, rows in grouped.items()
        }
        async with self._lock:
            # Remove a prior version of this source, including tail chunks after shorter re-ingestion.
            for lang in ("en", "ru"):
                old = [row.id for row in self._chunks[lang] if row.source == source]
                if old:
                    await asyncio.to_thread(self.client.delete, f"chunks_{lang}", old)
                self._chunks[lang] = [row for row in self._chunks[lang] if row.source != source]
            for lang, rows in grouped.items():
                vectors = vectors_by_language[lang]
                name = f"chunks_{lang}"
                if not self.client.collection_exists(name):
                    await asyncio.to_thread(self.client.create_collection, name, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
                points = [models.PointStruct(id=row.id, vector=vector, payload=row.model_dump(exclude={"id", "score"})) for row, vector in zip(rows, vectors)]
                await asyncio.to_thread(self.client.upsert, name, points)
                self._chunks[lang].extend(rows)
            for lang in ("en", "ru"):
                self._rebuild(lang)
        return [row.id for rows in grouped.values() for row in rows]

    async def search(self, query: str, limit: int = 20, vector_query: str | None = None, source: str | None = None) -> list[Chunk]:
        lang = language_of(query)
        async with self._lock:
            if not self._chunks[lang]:
                return []
            vector = (await self.models.embed([vector_query or query], lang))[0]
            name = f"chunks_{lang}"
            source_filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]) if source else None
            result = await asyncio.to_thread(self.client.query_points, name, query=vector, query_filter=source_filter, limit=limit, with_payload=True)
            dense = [Chunk(id=int(r.id), score=float(r.score), **r.payload) for r in result.points]
            bm = self._bm25[lang]
            tokens = tokenize(query)
            sparse = []
            if bm and tokens:
                scores = bm.get_scores(tokens)
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                sparse = [self._chunks[lang][i].model_copy(update={"score": float(scores[i])}) for i in order if scores[i] > 0 and (source is None or self._chunks[lang][i].source == source)][:limit]
            return rrf([dense, sparse])[:limit]

    async def search_dense(self, query: str, limit: int = 5) -> list[Chunk]:
        lang = language_of(query)
        async with self._lock:
            if not self._chunks[lang]:
                return []
            vector = (await self.models.embed([query], lang))[0]
            result = await asyncio.to_thread(self.client.query_points, f"chunks_{lang}", query=vector, limit=limit, with_payload=True)
            return [Chunk(id=int(r.id), score=float(r.score), **r.payload) for r in result.points]

    def count(self) -> int:
        return sum(len(rows) for rows in self._chunks.values())

    def close(self) -> None:
        self.client.close()
