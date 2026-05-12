"""LatencyBudget tests."""

from __future__ import annotations

from voice_agent.latency import BudgetViolation, LatencyBudget, Stage


def test_no_violations_when_under_budget():
    b = LatencyBudget(vad_ms=100, stt_ms=200, llm_ms=300, tts_ms=50)
    b.record(Stage.STT, 150)
    b.record(Stage.LLM, 280)
    assert b.violations() == []


def test_records_violations_with_stage_label_and_overage():
    b = LatencyBudget(stt_ms=100)
    b.record(Stage.STT, 240)
    vs = b.violations()
    assert len(vs) == 1
    assert vs[0].stage == Stage.STT
    assert vs[0].actual_ms == 240
    assert vs[0].overage_ms == 140


def test_total_budget_is_sum_of_stages():
    b = LatencyBudget(vad_ms=100, stt_ms=200, llm_ms=300, tts_ms=50)
    assert b.total_budget_ms == 650


def test_total_actual_sums_recorded_stages():
    b = LatencyBudget()
    b.record(Stage.STT, 50)
    b.record(Stage.LLM, 60)
    assert b.total_actual_ms == 110
