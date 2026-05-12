"""Voice-to-voice latency benchmark.

For each backend config, runs N synthetic turns through the pipeline and
reports p50 / p95 / p99 voice-to-voice latency.

Two configs by default:
  * `mock-fast` — zero-latency mocks (measures pipeline overhead floor)
  * `mock-realistic` — STT 200 ms / LLM 250 ms / TTS first-chunk 80 ms
                      (close to production budgets for a voice agent)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_agent.audio import generate_silence, generate_speech
from voice_agent.backends import MockLLM, MockSTT, MockTTS
from voice_agent.pipeline import VoicePipeline


RESULTS = Path(__file__).resolve().parent / "results.json"


async def _stream(chunks):
    for c in chunks:
        yield c
        await asyncio.sleep(0)


async def run_config(name, stt_ms, llm_ms, tts_first_ms, tts_chunk_ms, n_turns) -> dict:
    voice_to_voices: list[int] = []
    for _ in range(n_turns):
        pipeline = VoicePipeline(
            MockSTT(latency_ms=stt_ms),
            MockLLM(latency_ms=llm_ms),
            MockTTS(first_chunk_latency_ms=tts_first_ms, per_chunk_latency_ms=tts_chunk_ms),
        )
        audio = generate_speech(4) + generate_silence(15)
        trace = await pipeline.turn(_stream(audio))
        voice_to_voices.append(trace.voice_to_voice_ms)
    s = sorted(voice_to_voices)
    def pct(p):
        return s[min(len(s) - 1, max(0, int(p * (len(s) - 1))))]
    return {
        "name": name,
        "n_turns": n_turns,
        "stt_ms": stt_ms,
        "llm_ms": llm_ms,
        "tts_first_chunk_ms": tts_first_ms,
        "tts_per_chunk_ms": tts_chunk_ms,
        "voice_to_voice_ms": {
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": max(s),
            "mean": round(statistics.mean(s), 1),
        },
    }


async def main_async(n_turns: int) -> int:
    configs = [
        await run_config("mock-fast", 0, 0, 0, 0, n_turns),
        await run_config("mock-realistic", 200, 250, 80, 30, n_turns),
    ]
    out = {"configs": configs}
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nresults written to {RESULTS}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args.n)))


if __name__ == "__main__":
    main()
