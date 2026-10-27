"""Run retrieval A/B on an annotated JSONL set; no fabricated scores."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from ac_rag.api import make_engine


async def main(path: Path, output: Path, k: int):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Dataset is empty")
    engine = make_engine()
    reports = {}
    try:
        for name in ("naive_dense", "hybrid_rerank"):
            precisions, recalls, latencies = [], [], []
            for case in cases:
                relevant = set(case["relevant_chunk_ids"])
                started = time.perf_counter()
                if name == "naive_dense":
                    chunks = await engine.store.search_dense(case["query"], k)
                else:
                    candidates = await engine.store.search(case["query"], engine.settings.retrieval_limit)
                    chunks = (await engine._rerank(case["query"], candidates))[:k]
                latencies.append((time.perf_counter() - started) * 1000)
                found = {chunk.id for chunk in chunks}
                precisions.append(len(found & relevant) / k)
                recalls.append(len(found & relevant) / len(relevant) if relevant else 1.0)
            reports[name] = {
                f"precision_at_{k}": statistics.mean(precisions),
                f"recall_at_{k}": statistics.mean(recalls),
                "latency_ms_median": statistics.median(latencies),
                "latency_ms_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
                "n": len(cases),
            }
    finally:
        engine.store.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/retrieval.json"))
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.output, args.k))
