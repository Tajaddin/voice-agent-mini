"""Per-stage latency budget tracker.

A voice pipeline has four budgets:

* ``vad_ms``    — time from final speech frame to "user is done talking"
* ``stt_ms``    — STT decode time after the utterance is complete
* ``llm_ms``    — LLM time-to-first-token
* ``tts_ms``    — TTS time-to-first-audio

The tracker records actuals for each stage and produces a list of named
:class:`BudgetViolation` objects. The pipeline can choose to log violations
or take corrective action (e.g., downshift model on persistent LLM overage).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Stage(str, enum.Enum):
    VAD = "vad"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"


@dataclass
class BudgetViolation:
    stage: Stage
    actual_ms: int
    budget_ms: int

    @property
    def overage_ms(self) -> int:
        return self.actual_ms - self.budget_ms


@dataclass
class LatencyBudget:
    vad_ms: int = 250
    stt_ms: int = 300
    llm_ms: int = 350
    tts_ms: int = 100
    actuals: dict[Stage, int] = field(default_factory=dict)

    def record(self, stage: Stage, actual_ms: int) -> None:
        self.actuals[stage] = actual_ms

    def violations(self) -> list[BudgetViolation]:
        budgets = {
            Stage.VAD: self.vad_ms,
            Stage.STT: self.stt_ms,
            Stage.LLM: self.llm_ms,
            Stage.TTS: self.tts_ms,
        }
        out: list[BudgetViolation] = []
        for st, actual in self.actuals.items():
            budget = budgets[st]
            if actual > budget:
                out.append(BudgetViolation(stage=st, actual_ms=actual, budget_ms=budget))
        return out

    @property
    def total_budget_ms(self) -> int:
        return self.vad_ms + self.stt_ms + self.llm_ms + self.tts_ms

    @property
    def total_actual_ms(self) -> int:
        return sum(self.actuals.get(s, 0) for s in Stage)
