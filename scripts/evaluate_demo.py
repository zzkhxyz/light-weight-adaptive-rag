"""Reproducible live comparison on the fictional long-document fixture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from ac_rag.examples.long_report import EXPECTED, POSITIONS, QUESTIONS


def request_json(base: str, route: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(f"{base.rstrip('/')}{route}", data=data,
                      headers={"Content-Type": "application/json"} if data else {},
                      method="POST" if data else "GET")
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def summarize(rows: list[dict], method: str) -> dict:
    results = [row[method] for row in rows]
    tokens_known = all(result["input_tokens"] is not None for result in results)
    return {
        "correct_answers": sum(result["expected_found"] for result in results),
        "questions": len(results),
        "cited_answers": sum(result["cited"] for result in results),
        "mean_latency_ms": round(sum(result["latency_ms"] for result in results) / len(results), 2),
        "llm_calls": sum(result["llm_calls"] for result in results),
        "input_tokens": sum(result["input_tokens"] for result in results) if tokens_known else None,
        "output_tokens": sum(result["output_tokens"] for result in results) if tokens_known else None,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Длинный пример: воспроизводимое сравнение",
        "",
        f"Запуск: {report['timestamp_utc']} UTC. Модель: `{report['model']}`. Оба метода получили одинаковые вопросы и документ.",
        "",
        "Проверка правильности здесь означает наличие ожидаемого факта в ответе, а не независимую оценку его достоверности. "
        "Наличие ссылки и внутренний guard также не доказывают фактическую точность.",
        "",
        "| Вопрос | Позиция | Метод | Ожидаемый факт | Факт в ответе | Ссылка | Время, с | Вызовы LLM | Токены | Цена, USD |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        for method, label in (("baseline", "Обычный"), ("adaptive", "Adaptive")):
            item = row[method]
            tokens = "—" if item["input_tokens"] is None else str(item["input_tokens"] + item["output_tokens"])
            cost = "—" if item["estimated_cost_usd"] is None else f"{item['estimated_cost_usd']:.6f}"
            lines.append(f"| {row['question']} | {row['position']} | {label} | {row['expected']} | "
                         f"{'да' if item['expected_found'] else 'нет'} | {'да' if item['cited'] else 'нет'} | "
                         f"{item['latency_ms'] / 1000:.2f} | {item['llm_calls']} | {tokens} | {cost} |")
    lines.extend(["", "Цена показана только если при запуске заданы тарифы входных и выходных токенов. "
                  "Результаты относятся только к этим трём учебным вопросам. Для оценки качества на реальных документах нужен отдельный размеченный набор.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/long-demo.json"))
    parser.add_argument("--input-usd-per-million", type=float, default=None)
    parser.add_argument("--output-usd-per-million", type=float, default=None)
    args = parser.parse_args()
    if (args.input_usd_per_million is None) != (args.output_usd_per_million is None):
        parser.error("Set both input and output prices, or neither")
    if args.input_usd_per_million is not None and min(args.input_usd_per_million, args.output_usd_per_million) < 0:
        parser.error("Prices must be nonnegative")
    health = request_json(args.base_url, "/health")
    source = request_json(args.base_url, "/example/long/load", {})["source"]
    rows = []
    for question, expected, position in zip(QUESTIONS, EXPECTED, POSITIONS):
        comparison = request_json(args.base_url, "/compare", {"query": question, "source": source, "document_only": True})
        case = {"question": question, "expected": expected, "position": position}
        for method in ("baseline", "adaptive"):
            result = comparison[method]
            case[method] = {
                "answer": result["answer"], "citations": result["citations"],
                "expected_found": expected.casefold() in result["answer"].casefold(),
                "cited": bool(result["citations"]),
                "latency_ms": result["latency_ms"], "llm_calls": result["llm_calls"],
                "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                "route": result["route"], "reasoning_trace": result["reasoning_trace"],
                "estimated_cost_usd": (
                    round((result["input_tokens"] * args.input_usd_per_million +
                           result["output_tokens"] * args.output_usd_per_million) / 1_000_000, 8)
                    if args.input_usd_per_million is not None and result["input_tokens"] is not None else None
                ),
            }
        rows.append(case)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": health["model"], "source": source, "cases": rows,
        "prices_usd_per_million": {"input": args.input_usd_per_million,
                                   "output": args.output_usd_per_million},
        "summary": {method: summarize(rows, method) for method in ("baseline", "adaptive")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
