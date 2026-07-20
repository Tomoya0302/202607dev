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


def test_warm_002_003_exercises_steps_2_through_7(monkeypatch):
    from agents.heuristic import agent as agent_module
    from src.packing_core import watchdog

    names = (
        "build_state",
        "enumerate_candidates",
        "filter_candidates",
        "support_ratio",
        "cg_margin",
        "provisional_p_ng",
        "heuristic_score",
        "safe_decide",
        "make_action",
    )
    calls = {name: 0 for name in names}
    calls["check_l_path"] = 0

    for name in names:
        original = getattr(agent_module, name)

        def _wrapper(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(agent_module, name, _wrapper)

    original_check_l_path = watchdog.check_l_path

    def _check_l_path_wrapper(*args, **kwargs):
        calls["check_l_path"] += 1
        return original_check_l_path(*args, **kwargs)

    monkeypatch.setattr(watchdog, "check_l_path", _check_l_path_wrapper)

    agent_module.Agent(module_path="agents/heuristic")

    assert all(count >= 1 for count in calls.values()), calls


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


def test_warm_007_does_not_leave_dummy_init_state_on_agent():
    from agents.heuristic.agent import Agent

    agent = Agent(module_path="agents/heuristic")

    assert not hasattr(agent, "optimize_enabled")
    assert not hasattr(agent, "lookahead_k")
    assert not hasattr(agent, "container_list")

    init, _ = placeable_case()
    agent.get_init_states(init)
    assert agent.optimize_enabled is True
    assert agent.lookahead_k == 5
    assert agent.container_list is init["container_list"]


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


def test_warm_009_exception_warns_once_and_initialization_continues(monkeypatch, caplog):
    from agents.heuristic import agent as agent_module

    calls = {"n": 0}

    def _raising_build_state(observation, init):
        calls["n"] += 1
        raise RuntimeError("injected T-027 warmup failure")

    monkeypatch.setattr(agent_module, "build_state", _raising_build_state)
    with caplog.at_level(logging.WARNING, logger="packing"):
        agent = agent_module.Agent(module_path="agents/heuristic")

    assert calls["n"] == 1
    assert agent._policy_step == 0
    assert sum("warmup failed" in record.getMessage() for record in caplog.records) == 1


def test_warm_010_does_not_catch_base_exception(monkeypatch):
    from agents.heuristic import agent as agent_module

    def _raising_build_state(observation, init):
        raise KeyboardInterrupt("injected T-027 BaseException")

    monkeypatch.setattr(agent_module, "build_state", _raising_build_state)
    with pytest.raises(KeyboardInterrupt, match="T-027 BaseException"):
        agent_module.Agent(module_path="agents/heuristic")
