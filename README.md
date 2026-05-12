# voice-agent-mini

> Speech → LLM → speech pipeline with **barge-in interruption**, per-stage **latency budgets**, and an **end-to-end voice-to-voice latency** number. Pipeline overhead: **p99 = 1 ms** with zero-latency mocks. Realistic profile (STT 200 ms, LLM 250 ms, TTS 80 ms first chunk): **p50 = 559 ms, p99 = 574 ms** voice-to-voice. **22/22 tests** in 0.7 s.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## What's actually here

A voice agent is four backends in a trench coat: VAD → STT → LLM → TTS. The interesting parts aren't any single backend — they're the **failure modes between them**:

* **Barge-in.** The user starts talking while the agent is talking. You have to cancel the TTS *while it's mid-stream*, return to listening, and not double-trigger a turn.
* **Latency budget enforcement.** Each stage has a named budget. When STT consistently overruns, you swap to a faster model. The pipeline can't fix this for you, but it can surface the exact stage and overage.
* **Voice-to-voice timing.** The number that matters to users is "I stopped talking → I heard the first response audio." Everything else is a proxy.

This repo ships exactly those three: a VAD state machine, a `LatencyBudget` tracker, and a `VoicePipeline` orchestrator that produces a full `TurnTrace` per turn with every timestamp.

## Hero benchmark

```bash
python bench/run_benchmark.py --n 30
```

```json
{
  "configs": [
    {
      "name": "mock-fast",
      "voice_to_voice_ms": { "p50": 0, "p95": 1, "p99": 1, "max": 1, "mean": 0.1 }
    },
    {
      "name": "mock-realistic",
      "stt_ms": 200, "llm_ms": 250, "tts_first_chunk_ms": 80,
      "voice_to_voice_ms": { "p50": 559, "p95": 573, "p99": 574, "mean": 559.2 }
    }
  ]
}
```

| Profile | Pipeline overhead | Real backend cost | Total V2V |
|---|---:|---:|---:|
| `mock-fast` (zero-latency mocks) | **~1 ms** | 0 | ~1 ms |
| `mock-realistic` (production-shape) | ~29 ms | 530 ms (STT+LLM+TTS) | **~559 ms** |

The pipeline overhead at p99 is **1 ms**. The total voice-to-voice latency tracks the sum of backend costs with ~5% jitter from event-loop scheduling. That's the engineering invariant: the pipeline doesn't add latency, the backends do.

## Architecture

```
input audio  ───►  VAD  ───►  STT  ───►  LLM  ───►  TTS  ───►  output audio
                    │                                  ▲
                    │                                  │
                    └──── barge-in event ──────────────┘
                          (cancels TTS, returns to listening)
```

* `VAD` is a short-time-energy state machine with onset + hangover thresholds.
* `LatencyBudget` records per-stage actuals and emits `BudgetViolation(stage, actual_ms, overage_ms)` for any breached budget.
* `VoicePipeline.turn()` is the orchestrator. While TTS is streaming, the input is **still fed through the VAD**; a `BargeIn.START` event cancels the TTS task and returns the trace early with `cancelled=True`.

## Backend protocols

```python
class STTBackend(Protocol):
    async def transcribe(self, audio: list[AudioChunk]) -> str: ...

class LLMBackend(Protocol):
    async def reply(self, prompt: str) -> LLMReply: ...

class TTSBackend(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]: ...
```

Ships with:

* **`MockSTT`** / **`MockLLM`** / **`MockTTS`** — deterministic, configurable per-call latency. Used by every test and the benchmark.
* **`AnthropicLLM`** — wraps `anthropic.AsyncAnthropic.messages.create`.
* Hook points for **Whisper** (STT) and **ElevenLabs** (TTS) are sketched in the docs — both are <100 LoC adapters.

## Quickstart

```bash
pip install -e ".[dev]"

# Run one mock turn end-to-end:
voice-agent demo
# transcript: ...
# reply:      ...
# voice-to-voice ms: 0
# per-stage latencies: {Stage.VAD: 6, Stage.STT: 0, Stage.LLM: 0, Stage.TTS: 0}
# violations: []
```

Programmatic:

```python
import asyncio
from voice_agent import (
    MockSTT, MockLLM, MockTTS, VoicePipeline,
    generate_speech, generate_silence,
)

async def main():
    pipe = VoicePipeline(MockSTT(), MockLLM(), MockTTS())
    async def stream():
        for c in generate_speech(4) + generate_silence(15):
            yield c
    trace = await pipe.turn(stream())
    print("V2V:", trace.voice_to_voice_ms, "ms")
    print("budget violations:", trace.latency.violations())

asyncio.run(main())
```

## The barge-in test

The trickiest test in the suite:

```python
async def test_barge_in_cancels_tts_and_records_timestamp():
    audio = (
        generate_speech(4)             # 80 ms speech => VAD START
        + generate_silence(15)         # 300 ms silence => VAD END (utterance done)
        + generate_silence(2)          # filler while STT+LLM run
        + generate_speech(20, start_ms=600)   # BARGE-IN!
    )
    pipeline = VoicePipeline(MockSTT(...), MockLLM(...), MockTTS(per_chunk_latency_ms=30))
    trace = await pipeline.turn(_stream(audio))
    assert trace.barge_in_at_ms is not None
    assert trace.cancelled is True
```

The pipeline must:

1. Detect the first utterance end and run STT/LLM/TTS.
2. While TTS streams (each chunk 30 ms apart), continue feeding the *same* input stream through the VAD.
3. When the second speech burst is detected, cancel the TTS task, await its `CancelledError`, and return the trace with `barge_in_at_ms` and `cancelled=True` set.

That cancel path is what most homemade voice agents get wrong — they keep talking over the user.

## Tests

```bash
pytest -v
```

```
tests/test_audio.py        4 passed   silence/speech generators + duration arithmetic
tests/test_vad.py          5 passed   silence-only, onset, continue+end-after-hangover, sub-threshold, empty pcm
tests/test_latency.py      4 passed   under-budget, named violations, totals
tests/test_backends.py     4 passed   mock STT/LLM/TTS + TTS cancellation
tests/test_pipeline.py     5 passed   happy path, STT overage, BARGE-IN, stream-without-END, tight V2V on fast mocks
─────────────────────────────────────────────
22 passed in 0.67s
```

## Project layout

```
.
├── src/voice_agent/
│   ├── audio.py         # AudioChunk + generate_silence / generate_speech for tests
│   ├── vad.py           # short-time-energy VAD state machine (onset + hangover)
│   ├── latency.py       # LatencyBudget + Stage + BudgetViolation
│   ├── backends.py      # STT/LLM/TTS protocols + Mock + Anthropic adapters
│   ├── pipeline.py      # VoicePipeline + TurnTrace orchestrator with barge-in
│   └── cli.py           # `voice-agent demo`
├── tests/               # 22 cases across 5 files
└── bench/run_benchmark.py
```

## Limitations

**Synthetic audio only in tests.** The PCM frames are constant-amplitude tones, not real speech. A production VAD swap (webrtcvad or Silero) is one class change.

**Sync STT, not streaming.** `STTBackend.transcribe(audio: list)` takes the whole utterance after VAD END. For lower latency, swap for a streaming STT that emits partial transcripts during the user's turn (Whisper-streaming, Deepgram).

**No interruption text repair.** After a barge-in, the trace records `cancelled=True` but doesn't reconcile what the model already said with what the user just said. Real agents need either "I was saying X, but you asked Y" handling or fast cancel-and-retry — out of scope here.

**No speaker turn-taking model.** The VAD treats any high-energy frame as speech. Real two-party conversations need a smarter turn-taking model that disambiguates "the user paused" from "the user is done."

**No echo cancellation.** Assumes a half-duplex input (e.g., phone headset). For full-duplex (smart speakers, conference rooms), pair with an AEC frontend.

## License

MIT — see [LICENSE](LICENSE).
