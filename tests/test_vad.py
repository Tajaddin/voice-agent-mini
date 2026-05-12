"""VAD state-machine tests."""

from __future__ import annotations

from voice_agent.audio import generate_silence, generate_speech
from voice_agent.vad import VAD, BargeIn


def test_silence_only_stays_silence():
    vad = VAD()
    for c in generate_silence(10):
        assert vad.observe(c) == BargeIn.SILENCE


def test_speech_onset_fires_after_threshold():
    vad = VAD(onset_ms=60)
    # 2 frames of speech = 40 ms, below onset.
    pre = [vad.observe(c) for c in generate_speech(2)]
    assert all(e != BargeIn.START for e in pre)
    # 1 more frame = 60 ms, hits threshold.
    next_ev = vad.observe(generate_speech(1)[0])
    assert next_ev == BargeIn.START


def test_speech_continues_then_ends_after_hangover():
    vad = VAD(onset_ms=20, hangover_ms=80)
    vad.observe(generate_speech(1)[0])  # START
    cont = vad.observe(generate_speech(1)[0])
    assert cont == BargeIn.CONTINUE
    # Now silence — needs 4 silent frames (80 ms) before END.
    for _ in range(3):
        assert vad.observe(generate_silence(1)[0]) != BargeIn.END
    assert vad.observe(generate_silence(1)[0]) == BargeIn.END


def test_quiet_speech_below_threshold_treated_as_silence():
    vad = VAD(energy_threshold=10_000)  # higher than amp=8000
    for c in generate_speech(20, amp=8000):
        assert vad.observe(c) == BargeIn.SILENCE


def test_rms_returns_zero_on_empty_pcm():
    assert VAD.rms(b"") == 0.0
