import asyncio

from ac_rag.store import HybridStore


class FakeVectors:
    async def embed(self, texts, language):
        return [[1.0, 0.0] if "paris" in text.lower() else [0.0, 1.0] for text in texts]


def test_local_store_round_trip_and_source_replacement(tmp_path):
    async def run():
        store = HybridStore(tmp_path / "qdrant", FakeVectors())
        try:
            original = await store.ingest("guide", "Paris is the capital of France.")
            assert len(original) == 1
            assert (await store.search("Paris capital"))[0].id == original[0]
            replacement = await store.ingest("guide", "Berlin is the capital of Germany.")
            assert store.count() == 1
            assert replacement == original  # stable IDs for the same source and chunk index
            assert (await store.search("Germany"))[0].text.startswith("Berlin")
            await store.ingest("other", "Rome is the capital of Italy.")
            assert [row.source for row in await store.search("capital", source="guide")] == ["guide"]
        finally:
            store.close()
        reopened = HybridStore(tmp_path / "qdrant", FakeVectors())
        try:
            assert reopened.count() == 2
            assert (await reopened.search("Germany"))[0].text.startswith("Berlin")
        finally:
            reopened.close()
    asyncio.run(run())


def test_source_search_works_when_question_language_differs(tmp_path):
    async def run():
        store = HybridStore(tmp_path / "qdrant", FakeVectors())
        try:
            await store.ingest("english-report", "This report describes KPI monitoring and support programs.")
            matches = await store.search("О чем документ?", source="english-report")
            assert [row.source for row in matches] == ["english-report"]
            dense = await store.search_dense("О чем документ?", source="english-report")
            assert [row.source for row in dense] == ["english-report"]
        finally:
            store.close()
    asyncio.run(run())


def test_pdf_chunks_keep_page_number(tmp_path):
    async def run():
        store = HybridStore(tmp_path / "qdrant", FakeVectors())
        try:
            ids = await store.ingest("report.pdf", "First page. Second page.", pages=["First page.", "Second page."])
            assert (await store.get_chunk("report.pdf", ids[0])).page == 1
            assert (await store.get_chunk("report.pdf", ids[1])).page == 2
        finally:
            store.close()
    asyncio.run(run())
