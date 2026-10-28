from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .embedding import CPUModels
from .engine import Engine
from .examples.long_report import SOURCE as LONG_SOURCE, build_long_report
from .llm import OpenAIBackend
from .loaders import extract_document, extract_pdf_pages, render_highlighted_page
from .models import CompareResult, IngestRequest, IngestResult, QueryRequest, QueryResult
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


def stored_pdf_path(settings: Settings, source: str) -> Path:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return settings.qdrant_path.parent / "uploads" / f"{digest}.pdf"


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
    engine = request.app.state.engine
    return {"status": "ok", "chunks": engine.store.count(), "model": engine.settings.llm_model}


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(PACKAGE_DIR / "webapp" / "index.html")


@app.get("/lab", include_in_schema=False)
async def lab():
    return FileResponse(PACKAGE_DIR / "webapp" / "lab.html")


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


@app.post("/example/long/load", response_model=IngestResult)
async def load_long_example(request: Request):
    authorize(request)
    try:
        ids = await request.app.state.engine.store.ingest(LONG_SOURCE, build_long_report())
    except Exception as exc:
        raise HTTPException(503, detail=f"Long example ingestion failed: {type(exc).__name__}") from exc
    return IngestResult(source=LONG_SOURCE, chunk_ids=ids)


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
        pages = extract_pdf_pages(content) if name.lower().endswith(".pdf") else None
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, detail=f"Не удалось прочитать документ: {type(exc).__name__}") from exc
    try:
        ids = await request.app.state.engine.store.ingest(name, text, pages=pages)
        if pages is not None:
            path = stored_pdf_path(request.app.state.engine.settings, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
    except Exception as exc:
        raise HTTPException(503, detail=f"Document indexing failed: {type(exc).__name__}") from exc
    return IngestResult(source=name, chunk_ids=ids)


@app.get("/source-pdf")
async def source_pdf(source: str, request: Request):
    authorize(request)
    path = stored_pdf_path(request.app.state.engine.settings, source)
    if not path.is_file():
        raise HTTPException(404, detail="Исходный PDF не найден. Загрузите документ снова")
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@app.get("/source-preview")
async def source_preview(source: str, chunk_id: int, request: Request):
    authorize(request)
    engine = request.app.state.engine
    chunk = await engine.store.get_chunk(source, chunk_id)
    path = stored_pdf_path(engine.settings, source)
    if chunk is None or chunk.page is None or not path.is_file():
        raise HTTPException(404, detail="Страница PDF не найдена. Загрузите документ снова")
    image, highlighted = await asyncio.to_thread(render_highlighted_page, path.read_bytes(), chunk.page, chunk.text)
    return Response(image, media_type="image/png", headers={"X-Highlighted": str(highlighted).lower(), "Cache-Control": "no-store"})


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


@app.post("/compare", response_model=CompareResult)
async def compare(body: QueryRequest, request: Request):
    authorize(request)
    if not body.source:
        raise HTTPException(400, detail="Выберите документ для сравнения")
    try:
        engine = request.app.state.engine
        baseline = await engine.baseline_query(body.query, body.source)
        adaptive = await engine.query(body.query, document_only=True, source=body.source)
        return CompareResult(baseline=baseline, adaptive=adaptive)
    except Exception as exc:
        raise HTTPException(503, detail=f"Comparison failed: {type(exc).__name__}") from exc


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
