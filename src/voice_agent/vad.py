"""Energy-based voice-activity detector for barge-in.

Production deployments use webrtcvad or Silero VAD; this implementation
uses a simple short-time-energy threshold so the package has no native deps.

Barge-in protocol: while the TTS is speaking, the input stream continues
to be fed through ``VAD.observe(chunk)``. When the detector reports
``BargeIn.START``, the orchestrator must cancel the in-flight TTS and
return to listening mode.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

from voice_agent.audio import AudioChunk


class BargeIn(str, enum.Enum):
    SILENCE = "silence"
    START = "barge_in_start"      # first speech frame after silence
    CONTINUE = "speech_continue"
    END = "speech_end"             # silence resumed after speech


@dataclass
class VAD:
    """State machine driven by a moving short-time-energy estimate.

    * ``energy_threshold`` — RMS above this is considered speech.
    * ``hangover_ms`` — how long silence must persist to declare ``END``.
    * ``onset_ms``    — how long speech must persist before declaring ``START``.

    The defaults (energy 1500, hangover 200 ms, onset 60 ms) work for the
    synthetic test waveforms in ``audio.py`` and are reasonable starting
    points for real 16-bit PCM at 16 kHz.
    """

    energy_threshold: float = 1500.0
    hangover_ms: int = 200
    onset_ms: int = 60

    _in_speech: bool = False
    _speech_ms_run: int = 0     # consecutive speech ms (resets on silence)
    _silence_ms_run: int = 0    # consecutive silence ms (resets on speech)

    def reset(self) -> None:
        self._in_speech = False
        self._speech_ms_run = 0
        self._silence_ms_run = 0

    @staticmethod
    def rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        # 16-bit signed little endian
        n = len(pcm) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"<{n}h", pcm)
        # avoid float-list comprehension overhead — sum of squares
        sq = 0
        for s in samples:
            sq += s * s
        return (sq / n) ** 0.5

    def observe(self, chunk: AudioChunk) -> BargeIn:
        energy = self.rms(chunk.pcm)
        is_speech = energy >= self.energy_threshold
        chunk.is_speech = is_speech
        dur = int(chunk.duration_ms)
        if is_speech:
            self._silence_ms_run = 0
            self._speech_ms_run += dur
            if not self._in_speech and self._speech_ms_run >= self.onset_ms:
                self._in_speech = True
                return BargeIn.START
            if self._in_speech:
                return BargeIn.CONTINUE
            return BargeIn.SILENCE
        # silence frame
        self._speech_ms_run = 0
        self._silence_ms_run += dur
        if self._in_speech and self._silence_ms_run >= self.hangover_ms:
            self._in_speech = False
            return BargeIn.END
        return BargeIn.SILENCE if not self._in_speech else BargeIn.CONTINUE
