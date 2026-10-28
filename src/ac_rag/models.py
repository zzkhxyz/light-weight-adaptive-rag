from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Route(StrEnum):
    DIRECT = "direct"
    STANDARD_RAG = "standard_rag"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"


class Chunk(BaseModel):
    id: int
    text: str
    source: str
    language: Literal["en", "ru"]
    score: float = 0.0
    page: int | None = None


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[int]
    confidence_score: float = Field(ge=0, le=1)
    reasoning_trace: list[str] = Field(description="Audit events, never hidden model reasoning")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    route: Route | None = None
    document_only: bool = False
    source: str | None = Field(default=None, max_length=500)


class SentenceEvidence(BaseModel):
    sentence: str
    citations: list[int]


class QueryResult(RAGResponse):
    route: Route
    sources: list[Chunk]
    latency_ms: float
    llm_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    evidence: list[SentenceEvidence] = Field(default_factory=list)


class CompareResult(BaseModel):
    baseline: QueryResult
    adaptive: QueryResult


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=2_000_000)


class IngestResult(BaseModel):
    source: str
    chunk_ids: list[int]


class EvidenceChoice(BaseModel):
    sentence_index: int = Field(ge=0)
    citations: list[int]


class GuardVerdict(BaseModel):
    supported: bool
    unsupported_claims: list[str]
    evidence: list[EvidenceChoice] = Field(default_factory=list)
