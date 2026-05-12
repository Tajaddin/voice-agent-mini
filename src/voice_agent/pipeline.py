"""End-to-end voice pipeline with barge-in.

A turn is:

1. VAD on the incoming audio. When ``END`` fires, the utterance is done.
2. STT decodes the captured speech.
3. LLM produces a reply.
4. TTS streams audio out.

During step 4, the input stream is still fed through the VAD. If
``BargeIn.START`` is observed, the orchestrator cancels the TTS task and
returns to listening — that's the barge-in path.

The hero output is :class:`TurnTrace` — every timestamp + every per-stage
latency for one turn. The bench computes voice-to-voice latency as
``audio_end_ms -> first_output_audio_ms``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from voice_agent.audio import AudioChunk
from voice_agent.backends import LLMBackend, STTBackend, TTSBackend
from voice_agent.latency import LatencyBudget, Stage
from voice_agent.vad import VAD, BargeIn


@dataclass
class TurnTrace:
    utterance_end_at_ms: int = 0
    stt_done_at_ms: int = 0
    llm_done_at_ms: int = 0
    first_audio_out_at_ms: int = 0
    last_audio_out_at_ms: int = 0
    transcript: str = ""
    reply_text: str = ""
    barge_in_at_ms: Optional[int] = None
    cancelled: bool = False
    audio_out: list[AudioChunk] = field(default_factory=list)
    latency: LatencyBudget = field(default_factory=LatencyBudget)

    @property
    def voice_to_voice_ms(self) -> int:
        """End-to-end: from last input speech frame to first output audio."""
        if self.first_audio_out_at_ms == 0 or self.utterance_end_at_ms == 0:
            return 0
        return self.first_audio_out_at_ms - self.utterance_end_at_ms


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


class VoicePipeline:
    def __init__(
        self,
        stt: STTBackend,
        llm: LLMBackend,
        tts: TTSBackend,
        *,
        budget: LatencyBudget | None = None,
        vad: VAD | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.budget = budget or LatencyBudget()
        self.vad = vad or VAD()

    async def turn(
        self,
        input_stream: AsyncIterator[AudioChunk],
    ) -> TurnTrace:
        trace = TurnTrace()
        trace.latency = self.budget  # share state so trace + pipeline agree
        captured: list[AudioChunk] = []
        start_time = _now_ms()

        # Step 1: VAD listen until END.
        async for chunk in input_stream:
            event = self.vad.observe(chunk)
            captured.append(chunk)
            if event == BargeIn.END:
                trace.utterance_end_at_ms = _now_ms()
                self.budget.record(Stage.VAD, trace.utterance_end_at_ms - start_time)
                break
        else:
            # Stream ended without explicit END — treat as utterance over.
            trace.utterance_end_at_ms = _now_ms()
            self.budget.record(Stage.VAD, trace.utterance_end_at_ms - start_time)

        # Step 2: STT.
        t0 = _now_ms()
        trace.transcript = await self.stt.transcribe(captured)
        trace.stt_done_at_ms = _now_ms()
        self.budget.record(Stage.STT, trace.stt_done_at_ms - t0)

        # Step 3: LLM.
        t0 = _now_ms()
        reply = await self.llm.reply(trace.transcript)
        trace.reply_text = reply.text
        trace.llm_done_at_ms = _now_ms()
        self.budget.record(Stage.LLM, trace.llm_done_at_ms - t0)

        # Step 4: TTS streaming with barge-in.
        tts_task: asyncio.Task | None = None
        first_chunk_event = asyncio.Event()

        async def _drive_tts():
            try:
                async for out in self.tts.synthesize(trace.reply_text):
                    if not first_chunk_event.is_set():
                        trace.first_audio_out_at_ms = _now_ms()
                        first_chunk_event.set()
                    trace.audio_out.append(out)
                    trace.last_audio_out_at_ms = _now_ms()
            except asyncio.CancelledError:
                trace.cancelled = True
                raise

        tts_task = asyncio.create_task(_drive_tts())

        # While TTS plays, keep watching for barge-in on the input stream.
        t_tts_start = _now_ms()
        try:
            async for chunk in input_stream:
                event = self.vad.observe(chunk)
                if event == BargeIn.START:
                    trace.barge_in_at_ms = _now_ms()
                    tts_task.cancel()
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        pass
                    self.budget.record(Stage.TTS, trace.last_audio_out_at_ms - t_tts_start)
                    return trace
                if tts_task.done():
                    break
            await tts_task
        except asyncio.CancelledError:
            tts_task.cancel()
            raise

        self.budget.record(
            Stage.TTS,
            (trace.first_audio_out_at_ms - t_tts_start) if trace.first_audio_out_at_ms else 0,
        )
        return trace
