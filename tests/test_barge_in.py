"""Barge-in edge cases: TTS latency when it is cut off."""

from __future__ import annotations

import asyncio

import pytest

from voice_agent.audio import generate_silence, generate_speech
from voice_agent.backends import MockLLM, MockSTT, MockTTS
from voice_agent.latency import Stage
from voice_agent.pipeline import VoicePipeline

UTTERANCE = generate_speech(4) + generate_silence(15)


async def _stream(chunks, *, pause_before_ms: dict[int, int] | None = None):
    """Yield chunks; optionally sleep before the chunk whose start_ms is a key."""
    pauses = pause_before_ms or {}
    for c in chunks:
        if c.start_ms in pauses:
            await asyncio.sleep(pauses[c.start_ms] / 1000)
        yield c
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_barge_in_before_first_audio_records_no_negative_tts_latency():
    pipeline = VoicePipeline(
        MockSTT(latency_ms=0),
        MockLLM(latency_ms=0),
        MockTTS(first_chunk_latency_ms=500, per_chunk_latency_ms=0),
    )
    audio = UTTERANCE + generate_speech(10, start_ms=1000)
    trace = await pipeline.turn(_stream(audio))

    assert trace.cancelled and trace.first_audio_out_at_ms == 0
    assert trace.latency.actuals[Stage.TTS] >= 0


@pytest.mark.asyncio
async def test_barge_in_records_tts_time_to_first_audio():
    pipeline = VoicePipeline(
        MockSTT(latency_ms=0),
        MockLLM(latency_ms=0),
        MockTTS(first_chunk_latency_ms=0, per_chunk_latency_ms=50),
    )
    # Pause before the barge-in so TTS gets several chunks out first.
    audio = UTTERANCE + generate_speech(10, start_ms=1000)
    trace = await pipeline.turn(_stream(audio, pause_before_ms={1000: 160}))

    assert trace.cancelled and len(trace.audio_out) >= 2
    # TTS budget is time-to-first-audio, measured from when TTS started (after the LLM).
    assert trace.latency.actuals[Stage.TTS] <= trace.first_audio_out_at_ms - trace.llm_done_at_ms
