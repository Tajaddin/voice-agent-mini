"""Minimal audio types.

We deliberately model PCM as plain :class:`bytes` rather than numpy arrays
so the package has zero hard deps. Real backends can build a numpy view
themselves when they need it.
"""

from __future__ import annotations

from dataclasses import dataclass


SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
FRAME_MS = 20            # one frame = 20 ms
SAMPLES_PER_FRAME = SAMPLE_RATE_HZ * FRAME_MS // 1000


@dataclass(slots=True)
class AudioChunk:
    """One audio frame plus its position on the input stream's timeline."""

    pcm: bytes
    sample_rate: int = SAMPLE_RATE_HZ
    sample_width: int = SAMPLE_WIDTH_BYTES
    start_ms: int = 0       # ms from stream start
    is_speech: bool = False  # VAD decision, set later

    @property
    def duration_ms(self) -> float:
        samples = len(self.pcm) // self.sample_width
        return 1000.0 * samples / self.sample_rate


def frames_to_seconds(frames: int) -> float:
    return frames * FRAME_MS / 1000.0


def generate_silence(n_frames: int, *, start_ms: int = 0) -> list[AudioChunk]:
    """``n_frames`` frames of 16-bit zeros."""
    chunk_bytes = bytes(SAMPLES_PER_FRAME * SAMPLE_WIDTH_BYTES)
    return [
        AudioChunk(pcm=chunk_bytes, start_ms=start_ms + i * FRAME_MS, is_speech=False)
        for i in range(n_frames)
    ]


def generate_speech(n_frames: int, *, amp: int = 8000, start_ms: int = 0) -> list[AudioChunk]:
    """``n_frames`` frames of a high-amplitude tone (synthetic "speech")."""
    # 16-bit little-endian samples at constant amplitude => high RMS energy
    sample_le = amp.to_bytes(2, "little", signed=True)
    chunk_bytes = sample_le * SAMPLES_PER_FRAME
    return [
        AudioChunk(pcm=chunk_bytes, start_ms=start_ms + i * FRAME_MS, is_speech=True)
        for i in range(n_frames)
    ]
