"""T-027: Agent初期ウォームアップ契約テスト（詳細仕様書 v1.23 §4.12）。"""
from __future__ import annotations

import json
import logging
import math
import random

import numpy as np
import pytest

from tests.fixtures.t028.scenarios import placeable_case


def _numpy_random_state_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _telemetry_path(tmp_path, run_id: str):
    return tmp_path / "telemetry" / f"{run_id}.jsonl"


def test_warm_001_runs_once_for_each_agent_instance(monkeypatch):
    from agents.heuristic.agent import Agent

    calls = []

    def _spy(self):
        calls.append(self)

    monkeypatch.setattr(Agent, "_run_warmup", _spy)
    first = Agent(module_path="agents/heuristic")
    second = Agent(module_path="agents/heuristic")

    assert calls == [first, second]


def test_warm_004_005_does_not_emit_telemetry_or_advance_first_step(tmp_path, monkeypatch):
    from agents.heuristic.agent import Agent

    run_id = "t027-warm-no-row"
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("TELEMETRY_RUN_ID", run_id)

    agent = Agent(module_path="agents/heuristic")

    assert agent._policy_step == 0
    assert not _telemetry_path(tmp_path, run_id).exists()

    init, observation = placeable_case()
    agent.get_init_states(init)
    agent.policy(observation)

    rows = [json.loads(line) for line in _telemetry_path(tmp_path, run_id).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["step"] == 0
    assert rows[0]["first_step"] is True
    assert math.isfinite(rows[0]["t_total"])
    assert rows[0]["t_total"] >= 0.0


def test_warm_006_policy_completion_count_still_matches_jsonl_rows(tmp_path, monkeypatch):
    from agents.heuristic.agent import Agent

    run_id = "t027-warm-row-count"
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("TELEMETRY_RUN_ID", run_id)
    init, observation = placeable_case()
    agent = Agent(module_path="agents/heuristic")
    agent.get_init_states(init)

    agent.policy(observation)
    agent.policy(observation)

    rows = _telemetry_path(tmp_path, run_id).read_text().splitlines()
    assert agent._policy_step == 2
    assert len(rows) == 2


def test_warm_008_writes_no_files_and_preserves_random_state(tmp_path, monkeypatch):
    from agents.heuristic.agent import Agent

    monkeypatch.delenv("TELEMETRY_DIR", raising=False)
    monkeypatch.delenv("TELEMETRY_RUN_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    py_before = random.getstate()
    np_before = np.random.get_state()

    Agent(module_path="agents/heuristic")

    assert list(tmp_path.iterdir()) == []
    assert random.getstate() == py_before
    assert _numpy_random_state_equal(np.random.get_state(), np_before)


