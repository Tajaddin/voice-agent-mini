"""Mock backend tests."""

from __future__ import annotations

import pytest

from voice_agent.audio import generate_speech
from voice_agent.backends import MockLLM, MockSTT, MockTTS


@pytest.mark.asyncio
async def test_mock_stt_calls_responder_with_audio_frames():
    seen = []
    stt = MockSTT(responder=lambda frames: f"got {len(frames)}", latency_ms=0)
    res = await stt.transcribe(generate_speech(3))
    assert res == "got 3"
    assert stt.calls == 1


@pytest.mark.asyncio
async def test_mock_llm_records_call_count_and_returns_tokens():
    llm = MockLLM(responder=lambda p: "reply text", latency_ms=0)
    r = await llm.reply("hello world")
    assert r.text == "reply text"
    assert r.input_tokens >= 1
    assert r.output_tokens >= 1
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_mock_tts_yields_chunks_in_order():
    tts = MockTTS(first_chunk_latency_ms=1, per_chunk_latency_ms=1, chunks_per_word=2)
    chunks = []
    async for c in tts.synthesize("hello world"):
        chunks.append(c)
    # 2 words * 2 chunks_per_word = 4
    assert len(chunks) == 4
    assert tts.synthesised_chunks == 4


@pytest.mark.asyncio
async def test_mock_tts_marks_cancelled_when_cancelled():
    import asyncio

    tts = MockTTS(first_chunk_latency_ms=10, per_chunk_latency_ms=50)

    async def consume():
        async for _ in tts.synthesize("one two three four five"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert tts.cancelled is True
