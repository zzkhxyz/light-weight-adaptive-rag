# Ответ из документа — Adaptive & Corrective RAG Engine

Typed Python service for CPU retrieval with explicit routing, hybrid search, reranking, corrective fallback, structured generation, and a second faithfulness check. This is an implementation of the attached design, with performance claims left as targets until measured on a real corpus.

## Architecture

```mermaid
flowchart TD
    Q[Query] --> R{Router}
    R -->|DIRECT| D[Direct response]
    R -->|STANDARD| H[Hybrid retrieval]
    R -->|HYDE| Y[Hypothetical passage] --> H
    R -->|MULTI_QUERY| M[Three rewrites] --> H
    H --> F[BM25 + dense RRF k=60] --> X[CPU cross encoder]
    X --> C{CRAG threshold}
    C -->|low| W[Web or one rewrite fallback] --> X
    C -->|high| G[Structured LLM answer]
    D --> O[Response]
    G --> V{Faithfulness guard}
    V -->|fail| G2[Conservative retry] --> V2{Guard}
    V -->|pass| O
    V2 -->|pass| O
    V2 -->|fail| A[Abstain] --> O
```

## Простая демонстрация

После запуска откройте `http://127.0.0.1:8000/`. Нажмите **Выбрать пример**, выберите вопрос о такси после рабочей встречи и нажмите **Найти ответ**. Под ответом появится выдержка из одностраничной памятки. Вместо примера можно загрузить свой TXT, Markdown, PDF или DOCX до 5 МБ. Вопросы на странице ищут только в выбранном документе; если подтверждения нет, сервис сообщает об этом обычным текстом. Технический API остаётся доступен в `/docs`.

Кнопка **Сравнить методы** задаёт один вопрос по одному документу двум пайплайнам с одной LLM: простому dense top-k RAG с одной генерацией и текущему adaptive RAG с гибридным поиском и проверкой ответа. Карточки показывают ответ, цитируемый фрагмент, задержку, вызовы LLM и число токенов, если провайдер его сообщает. Время включает сетевой вызов модели; сравнение выполняется последовательно. Простой baseline не проверяет фактическую достоверность ответа.

Для теста длинного контекста нажмите **Длинный пример**. Учебный документ содержит 96 разделов с ключевыми фактами в начале, середине и конце. Три подготовленных вопроса проверяют отдельные факты и связь между разделами. [Воспроизводимый прогон и результаты](docs/benchmarks/long-demo.md) показывают также ошибки, задержку и расход токенов без заявления о превосходстве одного метода.

В опубликованном прогоне оба метода ответили на 3 из 3 учебных вопросов с цитатами. Adaptive RAG занял в среднем 6.23 с и потратил 12 973 токена против 1.80 с и 5 223 токенов у baseline. На этом маленьком наборе преимущество в качестве не показано.

Для новых PDF источник можно открыть на странице с подсветкой найденного фрагмента. Исходный PDF хранится локально в `data/uploads/`; файлы, загруженные до этой версии, нужно загрузить снова для просмотра страниц.

Пример вопроса по загруженному PDF и ответа с указанием источника:

![Пример вопроса и ответа по документу](docs/images/example-question-answer.png)

Шрифт интерфейса — системный `ui-rounded` с локальным запасным Nunito. Лицензия Nunito находится в `src/ac_rag/webapp/Nunito-OFL.txt`.

## Run locally

Requires Python 3.11+ and an OpenAI-compatible API key. First startup downloads CPU models; subsequent starts use the model cache. The local Qdrant directory holds indexed vectors and payloads. No Docker or GPU is required.

For OpenRouter, set `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, put the `sk-or-...` key in `OPENAI_API_KEY`, and set `AC_RAG_LLM_MODEL` to a provider/model slug such as `openai/gpt-4o-mini`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test]"
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
uvicorn ac_rag.api:app --host 127.0.0.1 --port 8000
```

```powershell
ac-rag ingest .\example.pdf --source example.pdf
ac-rag query "What does the document say about the rollout?"
```

Or use the REST API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ingest -Method Post -ContentType application/json -Body '{"source":"memo","text":"The rollout starts in November."}'
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType application/json -Body '{"query":"When does the rollout start?"}'
```

For a smoke test, `GET /health` should return `status: ok` and a chunk count. Index a short passage with `/ingest`, then ask a question whose answer appears in that passage using `/query`. Verify that `citations` contains an ID from `chunk_ids`, `sources` contains the same passage, and `reasoning_trace` ends with `guard:pass` or `guard:retry_pass`. Ask an unrelated document question to check that the service abstains when evidence is weak. With a bearer token configured, add `Authorization: Bearer <token>` to ingestion and query requests.

`POST /query/stream` emits SSE `status`, `token`, and `result` events. Tokens are emitted after validation so unsupported text cannot leak before the guard completes. This is incremental answer delivery, not provider token streaming. The `reasoning_trace` field records pipeline events only and does not expose hidden model reasoning.

## Design notes

- Embeddings use `BAAI/bge-small-en-v1.5` for English and `cointegrated/rubert-tiny2` for Russian. Qdrant collections are separated by language because vector dimensions can differ. When a document is selected, search checks every language collection containing that document.
- The configured cross encoder is English trained. Its sigmoid score is a routing heuristic, not a calibrated probability. For a selected document, it orders passages but does not block generation; the answer guard makes the final evidence check. For unscoped search, tune the `0.65` threshold per corpus and language. For Russian production traffic, evaluate and configure a multilingual reranker.
- The LLM sees short passage labels, which are mapped back to stable chunk IDs. The guard checks only the passages actually cited by the answer; uncited retrieved text cannot validate a claim.
- BM25 is rebuilt from Qdrant payloads on process startup. One process owns a local Qdrant path; use a Qdrant server and a shared sparse index for multiworker deployment.
- Reingesting a source replaces its prior chunks. Chunk IDs are stable 63 bit hashes of source and chunk index. Source content is treated as data, never as instructions.
- CLI ingestion accepts UTF-8 `.txt`/`.md`, text-based `.pdf`, and `.docx`. Scanned PDFs need OCR before ingestion.
- Web fallback is opt in with `AC_RAG_ENABLE_WEB=true` and `AC_RAG_TAVILY_API_KEY`. Otherwise the fallback reformulates queries once. If evidence stays weak or the guard fails twice, the service abstains.
- The example binds to localhost. Set `AC_RAG_SERVICE_API_KEY` to require a bearer token on ingestion and queries; the simple browser page assumes this token is unset. Put request limits and TLS at the reverse proxy before exposing the API publicly. API keys belong in environment variables, never in Git.

## Evaluation

Targets from the design: CPU routing/retrieval under 150 ms, Precision@K improvement of 25–30%, and fewer generation calls. These are **unverified**. The latency target excludes first model download and LLM network time. The generator and guard themselves add LLM calls; measure end to end costs instead of assuming a 40% reduction.

Create an annotated JSONL file with `query` and `relevant_chunk_ids` for documents already ingested; `eval/example.jsonl` only illustrates the format and is not a valid benchmark until the corresponding ID exists. Then run:

```powershell
python scripts/benchmark.py eval/my_corpus.jsonl --output results/retrieval.json -k 5
pytest -q
```

Демонстрационный прогон двух пайплайнов на открытом синтетическом документе (требует запущенного сервера и ключа LLM):

```powershell
python scripts/evaluate_demo.py
```

Скрипт обновляет `docs/benchmarks/long-demo.json` и `.md`. Проверка `expected_found` означает совпадение с ожидаемой строкой, а не независимое доказательство верности ответа. Цена в деньгах не вычисляется без тарифа конкретной модели; фактическое потребление токенов берётся из ответа провайдера, когда оно доступно.

## Эксперимент с локальной памятью Mamba

`scripts/probe_ssm.py` последовательно подаёт синтетический текст в локальную Mamba-130M, измеряет объём рекуррентного cache и проверяет восстановление одного факта, поставленного в начале, середине или конце. Это нулевая проба предобученной модели, а не обученный компрессор документов или замена RAG. Для точного ответа по произвольным документам ещё нужен обученный механизм извлечения и проверка ссылок на исходные страницы.

Экспериментальная страница доступна на локальном сервере по адресу `http://127.0.0.1:8000/lab`. Ползунок переключает только реально измеренные длины. [Сырые результаты](docs/benchmarks/ssm-probe.json) показывают, что в одном коротком опыте cache оставался размером 2.81 МиБ, а факт в начале не восстанавливался ни на одной из трёх длин. Этот опыт не оценивает 350M-модель и не доказывает качество на документах.

```powershell
python scripts/probe_ssm.py --repeats 4 16 32 --output docs/benchmarks/ssm-probe.json --web-output src/ac_rag/webapp/ssm-report.json
```

Первый запуск скачивает открытую модель `state-spaces/mamba-130m-hf` (около 520 МБ). На CPU используется медленная реализация PyTorch; результаты этого опыта нельзя переносить на оптимизированный GPU Mamba или на 350M-модель без отдельного измерения.

| Metric | Naive dense | Adaptive hybrid | Status |
|---|---:|---:|---|
| Precision@5 | — | — | Run annotated benchmark |
| Median retrieval latency | — | — | Run annotated benchmark |
| Faithfulness | — | — | Requires independently labeled answer set |
| Cost per query | — | — | Requires provider token usage and price snapshot |

The benchmark compares the same indexed corpus using dense top K against BM25+dense RRF and reranking. It records retrieval latency and precision/recall only. It does not manufacture RAGAS, trace screenshots, or provider cost. Production observability can be added through OpenTelemetry/Phoenix; the current response exposes per-request route, timing, and LLM call count.

## Test

`pytest` covers routing, RRF, citation allowlisting, and abstention after failed verification. It uses fake model adapters and needs no downloaded weights or API calls.
