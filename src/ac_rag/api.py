from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .embedding import CPUModels
from .engine import Engine
from .llm import OpenAIBackend
from .loaders import extract_document
from .models import IngestRequest, IngestResult, QueryRequest, QueryResult
from .store import HybridStore
from .web import TavilySearch


def make_engine(settings: Settings | None = None) -> Engine:
    settings = settings or Settings()
    models = CPUModels(settings.embed_en, settings.embed_ru, settings.rerank_model)
    store = HybridStore(settings.qdrant_path, models)
    web = TavilySearch(settings.tavily_api_key) if settings.enable_web and settings.tavily_api_key else None
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return Engine(store, models, OpenAIBackend(settings.llm_model, api_key=key, base_url=settings.openai_base_url or None), settings, web)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = make_engine()
    yield
    app.state.engine.store.close()


app = FastAPI(title="AC-RAG Engine", version="0.1.0", lifespan=lifespan)
PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_PATH = PACKAGE_DIR / "examples" / "expense-policy.txt"
app.mount("/assets", StaticFiles(directory=PACKAGE_DIR / "webapp"), name="assets")


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


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(PACKAGE_DIR / "webapp" / "index.html")


@app.get("/example/document", include_in_schema=False)
async def example_document():
    return PlainTextResponse(
        EXAMPLE_PATH.read_text(encoding="utf-8"),
        headers={"Content-Disposition": 'attachment; filename="expense-policy.txt"'},
    )


@app.post("/example/load", response_model=IngestResult)
async def load_example(request: Request):
    authorize(request)
    text = EXAMPLE_PATH.read_text(encoding="utf-8")
    try:
        ids = await request.app.state.engine.store.ingest("Пример — возмещение расходов", text)
    except Exception as exc:
        raise HTTPException(503, detail=f"Example ingestion failed: {type(exc).__name__}") from exc
    return IngestResult(source="Пример — возмещение расходов", chunk_ids=ids)


@app.post("/upload", response_model=IngestResult)
async def upload(request: Request, file: UploadFile = File(...)):
    authorize(request)
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(400, detail="Загрузите файл .txt, .md, .pdf или .docx")
    content = await file.read(5_000_001)
    if len(content) > 5_000_000:
        raise HTTPException(413, detail="Размер файла не должен превышать 5 МБ")
    try:
        text = extract_document(name, content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, detail=f"Не удалось прочитать документ: {type(exc).__name__}") from exc
    try:
        ids = await request.app.state.engine.store.ingest(name, text)
    except Exception as exc:
        raise HTTPException(503, detail=f"Document indexing failed: {type(exc).__name__}") from exc
    return IngestResult(source=name, chunk_ids=ids)


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
        return await request.app.state.engine.query(body.query, body.route, body.document_only, body.source)
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
            result = await request.app.state.engine.query(body.query, body.route, body.document_only, body.source)
            # Buffering until the guard passes avoids streaming claims that might be retracted.
            for token in result.answer.split(" "):
                yield sse("token", token + " ")
                await asyncio.sleep(0)
            yield sse("result", result.model_dump(mode="json"))
        except Exception as exc:
            yield sse("error", {"type": type(exc).__name__})
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
