"""STT / LLM / TTS backend protocols + mock + real adapters."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Protocol

from voice_agent.audio import AudioChunk


@dataclass
class LLMReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "mock"


# --- STT --------------------------------------------------------------


class STTBackend(Protocol):
    async def transcribe(self, audio: list[AudioChunk]) -> str: ...


class MockSTT:
    """Deterministic STT. ``responder`` is called with the audio frames."""

    def __init__(
        self,
        responder: Callable[[list[AudioChunk]], str] | None = None,
        *,
        latency_ms: int = 50,
    ) -> None:
        self.responder = responder or (lambda frames: f"<{len(frames)} frames>")
        self.latency_ms = latency_ms
        self.calls = 0

    async def transcribe(self, audio: list[AudioChunk]) -> str:
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        return self.responder(audio)


# --- LLM --------------------------------------------------------------


class LLMBackend(Protocol):
    async def reply(self, prompt: str) -> LLMReply: ...


class MockLLM:
    def __init__(
        self,
        responder: Callable[[str], str] | None = None,
        *,
        latency_ms: int = 80,
    ) -> None:
        self.responder = responder or (lambda p: f"mock answer to: {p[:40]}")
        self.latency_ms = latency_ms
        self.calls = 0

    async def reply(self, prompt: str) -> LLMReply:
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        text = self.responder(prompt)
        return LLMReply(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )


class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic  # noqa: F401

        self.model = model or os.environ.get(
            "VOICE_AGENT_MODEL", "claude-haiku-4-5-20251001"
        )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.calls = 0

    async def reply(self, prompt: str) -> LLMReply:
        from anthropic import AsyncAnthropic

        self.calls += 1
        client = AsyncAnthropic(api_key=self._api_key)
        resp = await client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMReply(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
        )


# --- TTS --------------------------------------------------------------


class TTSBackend(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]: ...


class MockTTS:
    """Yields audio frames at a configurable pace; respects cancellation."""

    def __init__(
        self,
        *,
        first_chunk_latency_ms: int = 40,
        per_chunk_latency_ms: int = 20,
        chunks_per_word: int = 2,
    ) -> None:
        self.first_chunk_latency_ms = first_chunk_latency_ms
        self.per_chunk_latency_ms = per_chunk_latency_ms
        self.chunks_per_word = chunks_per_word
        self.cancelled = False
        self.synthesised_chunks = 0

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        from voice_agent.audio import generate_speech

        try:
            await asyncio.sleep(self.first_chunk_latency_ms / 1000.0)
            words = max(1, len(text.split()))
            total = words * self.chunks_per_word
            for i, ch in enumerate(generate_speech(total, start_ms=0)):
                self.synthesised_chunks += 1
                yield ch
                if i < total - 1:
                    await asyncio.sleep(self.per_chunk_latency_ms / 1000.0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
