"""Audio generator + AudioChunk tests."""

from __future__ import annotations

from voice_agent.audio import (
    FRAME_MS,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    AudioChunk,
    frames_to_seconds,
    generate_silence,
    generate_speech,
)


def test_silence_has_correct_frame_count_and_duration():
    frames = generate_silence(5)
    assert len(frames) == 5
    assert all(c.duration_ms == FRAME_MS for c in frames)
    assert frames[0].start_ms == 0
    assert frames[-1].start_ms == 4 * FRAME_MS


def test_speech_pcm_has_nonzero_amplitude():
    chunks = generate_speech(2, amp=8000)
    raw = chunks[0].pcm
    # Every sample is 8000 → bytes alternate as expected.
    assert raw != bytes(len(raw))
    assert chunks[0].is_speech is True


def test_silence_pcm_is_all_zero_bytes():
    chunks = generate_silence(1)
    assert chunks[0].pcm == bytes(len(chunks[0].pcm))


def test_frames_to_seconds_arith():
    assert frames_to_seconds(0) == 0.0
    assert frames_to_seconds(50) == 50 * FRAME_MS / 1000.0
