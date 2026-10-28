from __future__ import annotations

import asyncio
import time

from .config import Settings
from .models import Chunk, GuardVerdict, QueryResult, Route
from .router import asks_for_overview, route_query
from .store import rrf


class Engine:
    def __init__(self, store, models, llm, settings: Settings, web=None):
        self.store = store
        self.models = models
        self.llm = llm
        self.settings = settings
        self.web = web

    async def baseline_query(self, question: str, source: str) -> QueryResult:
        """Plain dense top-k RAG: one retrieval and one generation, without verification."""
        started = time.perf_counter()
        if hasattr(self.llm, "reset_usage"):
            self.llm.reset_usage()
        chunks = await self.store.search_dense(question, self.settings.final_limit, source=source)
        if not chunks:
            return self._result("Выбранный документ не найден. Загрузите его снова.", [], 0.0,
                                Route.STANDARD_RAG, [], ["baseline:dense", "baseline:empty"], 0, started)
        response = self._restrict_citations(await self.llm.answer(question, chunks), chunks)
        return self._result(response.answer, response.citations, response.confidence_score,
                            Route.STANDARD_RAG, chunks, ["baseline:dense", "baseline:unverified"], 1, started)

    async def query(self, question: str, forced_route: Route | None = None, document_only: bool = False, source: str | None = None) -> QueryResult:
        started = time.perf_counter()
        if hasattr(self.llm, "reset_usage"):
            self.llm.reset_usage()
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

        overview = bool(document_only and source and asks_for_overview(question))
        if overview:
            candidates = await self.store.source_chunks(source)
            trace.append("retrieval:document_overview")
        elif route == Route.HYDE:
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

        selected = candidates if overview else await self._rerank(question, candidates)
        if not overview:
            trace.append(f"rerank:{len(selected)}")
        if document_only and source:
            # For a chosen document, reranking orders passages; the guard decides whether they support an answer.
            if overview:
                selected = selected[:12]
            else:
                # Keep retrieval leaders as well: the default cross encoder is English trained
                # and may demote a decisive passage in Russian or a cross-language query.
                selected = list({chunk.id: chunk for chunk in
                                 [*selected[:self.settings.final_limit], *candidates[:self.settings.final_limit]]}.values())
                trace.append("context:rerank_and_retrieval")
            trace.append("crag:document_scope")
        elif not selected or selected[0].score < self.settings.rerank_threshold:
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

        if not (document_only and source):
            selected = [c for c in selected if c.score >= self.settings.rerank_threshold][:self.settings.final_limit]
        if not selected:
            if document_only and source:
                text = "Выбранный документ не найден. Загрузите его снова." if any("а" <= c.lower() <= "я" for c in question) else "The selected document was not found. Upload it again."
            else:
                text = "Недостаточно релевантных данных для ответа." if any("а" <= c.lower() <= "я" for c in question) else "There is not enough relevant evidence to answer."
            return self._result(text, [], 0.0, route, [], trace + ["crag:abstain"], calls, started)

        response = await self.llm.answer(question, selected)
        calls += 1
        response = self._restrict_citations(response, selected)
        cited_context = [chunk for chunk in selected if chunk.id in response.citations]
        verdict = await self.llm.guard(question, response.answer, cited_context) if cited_context else GuardVerdict(supported=False, unsupported_claims=["no citations"])
        calls += int(bool(cited_context))
        trace.append("guard:pass" if verdict.supported else "guard:fail")
        if not verdict.supported:
            response = await self.llm.answer(question, selected, strict=True)
            calls += 1
            response = self._restrict_citations(response, selected)
            cited_context = [chunk for chunk in selected if chunk.id in response.citations]
            verdict = await self.llm.guard(question, response.answer, cited_context) if cited_context else GuardVerdict(supported=False, unsupported_claims=["no citations"])
            calls += int(bool(cited_context))
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

    def _result(self, answer, citations, confidence, route, sources, trace, calls, started) -> QueryResult:
        cited = set(citations)
        usage = self.llm.usage() if hasattr(self.llm, "usage") else None
        return QueryResult(
            answer=answer, citations=citations, confidence_score=confidence,
            reasoning_trace=trace, route=route,
            sources=[c for c in sources if c.id in cited],
            latency_ms=round((time.perf_counter() - started) * 1000, 2), llm_calls=calls,
            input_tokens=usage[0] if usage else None, output_tokens=usage[1] if usage else None,
        )
