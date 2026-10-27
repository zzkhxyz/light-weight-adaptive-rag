from __future__ import annotations

import asyncio
import time

from .config import Settings
from .models import Chunk, QueryResult, Route
from .router import route_query
from .store import rrf


class Engine:
    def __init__(self, store, models, llm, settings: Settings, web=None):
        self.store = store
        self.models = models
        self.llm = llm
        self.settings = settings
        self.web = web

    async def query(self, question: str, forced_route: Route | None = None, document_only: bool = False, source: str | None = None) -> QueryResult:
        started = time.perf_counter()
        route = forced_route or route_query(question)
        if document_only and route == Route.DIRECT:
            route = Route.STANDARD_RAG
        trace = [f"route:{route.value}"]
        calls = 0

        if route == Route.DIRECT:
            if question.strip().lower().rstrip("!?. ") in {"hi", "hello", "hey", "привет", "здравствуй", "здравствуйте"}:
                answer = "Здравствуйте!" if any("а" <= c.lower() <= "я" for c in question) else "Hello!"
                return self._result(answer, [], 1.0, route, [], trace + ["direct:greeting"], calls, started)
            response = await self.llm.answer(question, [], direct=True)
            calls += 1
            return self._result(response.answer, [], response.confidence_score, route, [], trace + ["direct:llm"], calls, started)

        if route == Route.HYDE:
            hypothesis = await self.llm.hyde(question)
            calls += 1
            candidates = await self.store.search(question, self.settings.retrieval_limit, vector_query=hypothesis, source=source)
            trace.append("retrieval:hyde")
        elif route == Route.MULTI_QUERY:
            queries = await self.llm.rewrite(question)
            calls += 1
            candidates = rrf(await asyncio.gather(*(self.store.search(q, self.settings.retrieval_limit, source=source) for q in [question, *queries])))[:self.settings.retrieval_limit]
            trace.append("retrieval:multi_query")
        else:
            candidates = await self.store.search(question, self.settings.retrieval_limit, source=source)
            trace.append("retrieval:hybrid")

        selected = await self._rerank(question, candidates)
        trace.append(f"rerank:{len(selected)}")
        if not selected or selected[0].score < self.settings.rerank_threshold:
            trace.append("crag:low_confidence")
            for _ in range(self.settings.max_fallbacks):
                if self.web is not None and not document_only:
                    try:
                        extra = await self.web.search(question)
                        trace.append("fallback:web")
                    except Exception:
                        extra = []
                        trace.append("fallback:web_error")
                else:
                    extra_queries = await self.llm.rewrite(question)
                    calls += 1
                    searches = await asyncio.gather(*(self.store.search(q, self.settings.retrieval_limit, source=source) for q in extra_queries))
                    extra = rrf(searches)[:self.settings.retrieval_limit]
                    trace.append("fallback:rewrite")
                selected = await self._rerank(question, rrf([selected, extra])[:self.settings.retrieval_limit])
                if selected and selected[0].score >= self.settings.rerank_threshold:
                    break

        selected = [c for c in selected if c.score >= self.settings.rerank_threshold][:self.settings.final_limit]
        if not selected:
            text = "Недостаточно релевантных данных для ответа." if any("а" <= c.lower() <= "я" for c in question) else "There is not enough relevant evidence to answer."
            return self._result(text, [], 0.0, route, [], trace + ["crag:abstain"], calls, started)

        response = await self.llm.answer(question, selected)
        calls += 1
        response = self._restrict_citations(response, selected)
        verdict = await self.llm.guard(question, response.answer, selected)
        calls += 1
        trace.append("guard:pass" if verdict.supported else "guard:fail")
        if not verdict.supported:
            response = await self.llm.answer(question, selected, strict=True)
            calls += 1
            response = self._restrict_citations(response, selected)
            verdict = await self.llm.guard(question, response.answer, selected)
            calls += 1
            trace.append("guard:retry_pass" if verdict.supported else "guard:retry_fail")
        if not verdict.supported or not response.citations:
            text = "Не удалось подтвердить ответ по источникам." if any("а" <= c.lower() <= "я" for c in question) else "The answer could not be verified against the sources."
            return self._result(text, [], 0.0, route, [], trace + ["guard:abstain"], calls, started)
        return self._result(response.answer, response.citations, response.confidence_score, route, selected, trace, calls, started)

    async def _rerank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        scores = await self.models.rerank(question, [c.text for c in chunks])
        return sorted([c.model_copy(update={"score": score}) for c, score in zip(chunks, scores)], key=lambda c: c.score, reverse=True)

    @staticmethod
    def _restrict_citations(response, chunks: list[Chunk]):
        valid = {c.id for c in chunks}
        return response.model_copy(update={"citations": list(dict.fromkeys(c for c in response.citations if c in valid))})

    @staticmethod
    def _result(answer, citations, confidence, route, sources, trace, calls, started) -> QueryResult:
        cited = set(citations)
        return QueryResult(
            answer=answer, citations=citations, confidence_score=confidence,
            reasoning_trace=trace, route=route,
            sources=[c for c in sources if c.id in cited],
            latency_ms=round((time.perf_counter() - started) * 1000, 2), llm_calls=calls,
        )
