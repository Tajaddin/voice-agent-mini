"""voice-agent-mini — STT → LLM → TTS pipeline with barge-in.

Exports:

* :class:`AudioChunk` — one PCM frame with timing metadata
* :class:`VAD` — energy + silence-duration barge-in detector
* :class:`LatencyBudget` — per-stage budget tracker, returns named violations
* :class:`MockSTT`, :class:`MockLLM`, :class:`MockTTS` — deterministic backends
* :class:`VoicePipeline` — orchestrates a full voice-to-voice turn with cancel
* :class:`TurnTrace` — timestamps + latencies for one turn (the hero metric)
"""

from voice_agent.audio import AudioChunk, frames_to_seconds, generate_silence, generate_speech
from voice_agent.backends import (
    AnthropicLLM,
    LLMBackend,
    LLMReply,
    MockLLM,
    MockSTT,
    MockTTS,
    STTBackend,
    TTSBackend,
)
from voice_agent.latency import LatencyBudget, BudgetViolation, Stage
from voice_agent.pipeline import TurnTrace, VoicePipeline
from voice_agent.vad import VAD, BargeIn

__all__ = [
    "AnthropicLLM",
    "AudioChunk",
    "BargeIn",
    "BudgetViolation",
    "LatencyBudget",
    "LLMBackend",
    "LLMReply",
    "MockLLM",
    "MockSTT",
    "MockTTS",
    "STTBackend",
    "Stage",
    "TTSBackend",
    "TurnTrace",
    "VAD",
    "VoicePipeline",
    "frames_to_seconds",
    "generate_silence",
    "generate_speech",
]
