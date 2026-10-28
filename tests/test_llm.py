import asyncio

from ac_rag.llm import OpenAIBackend
from ac_rag.models import Chunk, EvidenceChoice, GuardVerdict, RAGResponse


def test_short_citation_labels_map_back_to_chunk_ids():
    async def run():
        backend = object.__new__(OpenAIBackend)

        async def fake_parse(system, user, schema):
            assert "[1] First fact" in user
            assert "[2] Second fact" in user
            return RAGResponse(answer="Second fact", citations=[2, 99], confidence_score=1, reasoning_trace=[])

        backend._parse = fake_parse
        chunks = [Chunk(id=1234567890123, text="First fact", source="s", language="en"),
                  Chunk(id=9876543210123, text="Second fact", source="s", language="en")]
        return await backend.answer("Which fact?", chunks)

    assert asyncio.run(run()).citations == [9876543210123]


def test_guard_evidence_labels_map_back_to_chunk_ids():
    async def run():
        backend = object.__new__(OpenAIBackend)

        async def fake_parse(system, user, schema):
            assert "(0) Second fact." in user
            return GuardVerdict(supported=True, unsupported_claims=[],
                                evidence=[EvidenceChoice(sentence_index=0, citations=[2])])

        backend._parse = fake_parse
        chunks = [Chunk(id=11, text="First fact", source="s", language="en"),
                  Chunk(id=22, text="Second fact", source="s", language="en")]
        return await backend.guard("Which fact?", "Second fact.", chunks)

    assert asyncio.run(run()).evidence[0].citations == [22]
