from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import Settings
from .embedding import CPUModels
from .engine import Engine
from .llm import OpenAIBackend
from .models import IngestRequest, IngestResult, QueryRequest, QueryResult
from .store import HybridStore
from .web import TavilySearch


def make_engine(settings: Settings | None = None) -> Engine:
    settings = settings or Settings()
    models = CPUModels(settings.embed_en, settings.embed_ru, settings.rerank_model)
    store = HybridStore(settings.qdrant_path, models)
    web = TavilySearch(settings.tavily_api_key) if settings.enable_web and settings.tavily_api_key else None
    return Engine(store, models, OpenAIBackend(settings.llm_model), settings, web)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = make_engine()
    yield
    app.state.engine.store.close()


app = FastAPI(title="AC-RAG Engine", version="0.1.0", lifespan=lifespan)


def authorize(request: Request) -> None:
    configured = request.app.state.engine.settings.service_api_key
    if configured is None or not configured.get_secret_value():
        return
    provided = request.headers.get("authorization", "")
    expected = f"Bearer {configured.get_secret_value()}"
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(401, detail="Invalid or missing bearer token")


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "chunks": request.app.state.engine.store.count()}


@app.post("/ingest", response_model=IngestResult)
async def ingest(body: IngestRequest, request: Request):
    authorize(request)
    try:
        ids = await request.app.state.engine.store.ingest(body.source, body.text)
    except Exception as exc:
        raise HTTPException(503, detail=f"Ingestion failed: {type(exc).__name__}") from exc
    return IngestResult(source=body.source, chunk_ids=ids)


@app.post("/query", response_model=QueryResult)
async def query(body: QueryRequest, request: Request):
    authorize(request)
    try:
        return await request.app.state.engine.query(body.query, body.route)
    except Exception as exc:
        raise HTTPException(503, detail=f"Query failed: {type(exc).__name__}") from exc


def sse(event: str, value: object) -> str:
    return f"event: {event}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    authorize(request)
    async def events():
        yield sse("status", {"stage": "processing"})
        try:
            result = await request.app.state.engine.query(body.query, body.route)
            # Buffering until the guard passes avoids streaming claims that might be retracted.
            for token in result.answer.split(" "):
                yield sse("token", token + " ")
                await asyncio.sleep(0)
            yield sse("result", result.model_dump(mode="json"))
        except Exception as exc:
            yield sse("error", {"type": type(exc).__name__})
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
