from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .api import make_engine
from .loaders import load_document


async def _run(args):
    engine = make_engine()
    try:
        if args.command == "ingest":
            path = Path(args.file)
            ids = await engine.store.ingest(args.source or path.name, load_document(path))
            print(f"Indexed {len(ids)} chunks from {path}")
        elif args.command == "query":
            result = await engine.query(args.question)
            print(result.model_dump_json(indent=2))
    finally:
        engine.store.close()


def main():
    parser = argparse.ArgumentParser(prog="ac-rag")
    subs = parser.add_subparsers(dest="command", required=True)
    ingest = subs.add_parser("ingest")
    ingest.add_argument("file")
    ingest.add_argument("--source")
    query = subs.add_parser("query")
    query.add_argument("question")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
