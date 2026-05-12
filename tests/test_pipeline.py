"""End-to-end pipeline tests."""

from __future__ import annotations

import asyncio

import pytest

from voice_agent.audio import AudioChunk, generate_silence, generate_speech
from voice_agent.backends import MockLLM, MockSTT, MockTTS
from voice_agent.latency import LatencyBudget, Stage
from voice_agent.pipeline import VoicePipeline
from voice_agent.vad import VAD


async def _stream(chunks):
    for c in chunks:
        yield c
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_happy_path_voice_to_voice_returns_trace():
    pipeline = VoicePipeline(
        MockSTT(latency_ms=10),
        MockLLM(latency_ms=20),
        MockTTS(first_chunk_latency_ms=5, per_chunk_latency_ms=2),
    )
    audio = (
        generate_silence(2)
        + generate_speech(5)        # 100 ms of speech
        + generate_silence(15)      # 300 ms hangover -> END
    )
    trace = await pipeline.turn(_stream(audio))
    assert trace.transcript
    assert trace.reply_text
    assert trace.voice_to_voice_ms >= 0
    assert trace.audio_out
    assert trace.utterance_end_at_ms > 0
    assert trace.first_audio_out_at_ms > trace.utterance_end_at_ms


@pytest.mark.asyncio
async def test_stt_decode_latency_recorded_in_budget():
    pipeline = VoicePipeline(
        MockSTT(latency_ms=40),
        MockLLM(latency_ms=0),
        MockTTS(first_chunk_latency_ms=0, per_chunk_latency_ms=0),
        budget=LatencyBudget(stt_ms=20),
    )
    audio = generate_speech(4) + generate_silence(15)
    trace = await pipeline.turn(_stream(audio))
    violations = trace.latency.violations()
    assert any(v.stage == Stage.STT for v in violations)


@pytest.mark.asyncio
async def test_barge_in_cancels_tts_and_records_timestamp():
    """User starts talking mid-TTS. Pipeline must cancel and return early."""
    # Build an input stream that:
    # 1. emits speech + silence (first utterance, triggers a turn)
    # 2. then more silence
    # 3. then speech again (the barge-in)
    audio = (
        generate_speech(4)        # 80 ms of speech => triggers VAD START
        + generate_silence(15)    # 300 ms silence => END (utterance complete)
        + generate_silence(2)     # filler while STT+LLM run
        + generate_speech(20, start_ms=600)  # barge-in!
    )

    pipeline = VoicePipeline(
        MockSTT(latency_ms=5),
        MockLLM(latency_ms=5),
        MockTTS(first_chunk_latency_ms=5, per_chunk_latency_ms=30),
    )
    trace = await pipeline.turn(_stream(audio))
    assert trace.barge_in_at_ms is not None
    assert trace.cancelled is True
    # The TTS produced some audio before cancel, but stopped mid-stream.
    # We don't know exact count due to scheduling, but it should be < total.


@pytest.mark.asyncio
async def test_stream_ending_without_END_still_completes_turn():
    """If the input stream closes mid-utterance, the pipeline must still finish."""
    pipeline = VoicePipeline(
        MockSTT(latency_ms=0),
        MockLLM(latency_ms=0),
        MockTTS(first_chunk_latency_ms=0, per_chunk_latency_ms=0),
    )
    # Only a couple of speech frames, then stream closes — no END event.
    audio = generate_speech(3)
    trace = await pipeline.turn(_stream(audio))
    # Pipeline still ran STT/LLM/TTS.
    assert trace.transcript
    assert trace.reply_text


@pytest.mark.asyncio
async def test_total_latency_under_budget_for_zero_latency_mocks():
    """End-to-end with zero-latency mocks should land under the default budget."""
    pipeline = VoicePipeline(
        MockSTT(latency_ms=0),
        MockLLM(latency_ms=0),
        MockTTS(first_chunk_latency_ms=0, per_chunk_latency_ms=0),
    )
    audio = generate_speech(4) + generate_silence(15)
    trace = await pipeline.turn(_stream(audio))
    # With zero-latency mocks, voice-to-voice should be tiny — well under 100ms.
    assert trace.voice_to_voice_ms < 100
