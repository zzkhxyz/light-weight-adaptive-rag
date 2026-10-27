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
