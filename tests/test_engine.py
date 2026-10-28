import asyncio

from ac_rag.config import Settings
from ac_rag.engine import Engine
from ac_rag.models import Chunk, GuardVerdict, RAGResponse, Route
from ac_rag.router import route_query
from ac_rag.store import rrf, split_text


class FakeStore:
    async def search(self, query, limit=20, vector_query=None, source=None):
        return [Chunk(id=7, text="Paris is the capital of France.", source="guide", language="en")]

    async def source_chunks(self, source):
        return [Chunk(id=7, text="Paris is the capital of France.", source=source, language="en")]


class FakeModels:
    async def rerank(self, query, texts):
        return [0.92 for _ in texts]


class LowScoreModels:
    async def rerank(self, query, texts):
        return [0.05 for _ in texts]


class FakeLLM:
    def __init__(self, supported=True):
        self.supported = supported

    async def answer(self, query, context, direct=False, strict=False):
        return RAGResponse(answer="Paris is the capital of France.", citations=[7, 999], confidence_score=0.9, reasoning_trace=[])

    async def guard(self, query, answer, context):
        return GuardVerdict(supported=self.supported, unsupported_claims=[] if self.supported else ["claim"])

    async def rewrite(self, query):
        return [query]


def test_supported_answer_has_only_valid_citations():
    async def run():
        engine = Engine(FakeStore(), FakeModels(), FakeLLM(), Settings(max_fallbacks=0))
        return await engine.query("What is the capital of France in the guide?", Route.STANDARD_RAG)
    result = asyncio.run(run())
    assert result.citations == [7]
    assert [source.id for source in result.sources] == [7]
    assert result.llm_calls == 2


def test_failed_guard_abstains_after_retry():
    async def run():
        engine = Engine(FakeStore(), FakeModels(), FakeLLM(supported=False), Settings(max_fallbacks=0))
        return await engine.query("Question", Route.STANDARD_RAG)
    result = asyncio.run(run())
    assert result.citations == []
    assert result.confidence_score == 0
    assert result.llm_calls == 4


def test_selected_document_reaches_guard_despite_low_reranker_score():
    async def run():
        engine = Engine(FakeStore(), LowScoreModels(), FakeLLM(), Settings(max_fallbacks=0))
        return await engine.query("Что написано в документе?", Route.STANDARD_RAG, document_only=True, source="guide")
    result = asyncio.run(run())
    assert "Недостаточно релевантных данных" not in result.answer
    assert result.citations == [7]
    assert "guard:pass" in result.reasoning_trace


def test_document_overview_uses_source_without_reranking():
    async def run():
        engine = Engine(FakeStore(), LowScoreModels(), FakeLLM(), Settings(max_fallbacks=0))
        return await engine.query("О чем документ?", document_only=True, source="guide")
    result = asyncio.run(run())
    assert result.citations == [7]
    assert "retrieval:document_overview" in result.reasoning_trace


def test_rrf_rewards_overlap():
    a = Chunk(id=1, text="a", source="s", language="en")
    b = Chunk(id=2, text="b", source="s", language="en")
    assert rrf([[a, b], [b]])[0].id == 2


def test_router_and_chunker():
    assert route_query("Привет!") == Route.DIRECT
    assert route_query("Compare the approaches") == Route.HYDE
    assert split_text("one two three four", size=3, overlap=1) == ["one two three", "three four"]
