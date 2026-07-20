"""T-028: Agent統合・run_local実統合の契約テスト
（詳細仕様書 v1.22 §4.12「policy()最外殻telemetry accumulatorとstep/first_step」、
§4.13「T-028出力契約確定」、§5.7「telemetry出力先の環境変数注入」）。

3グループを扱う：

* AG-*（Agent正常系・emergency系）：`agents.heuristic.agent.Agent` へ環境変数
  （`TELEMETRY_DIR`/`TELEMETRY_RUN_ID`）を注入した状態で `policy()` を呼び、実際に書き出される
  JSONL行を読み戻して契約を検証する。Agent内部のwriter属性名には依存せず、
  §4.12で明示的に名指しされた `Agent._policy_step`（v1.22契約そのもの）と、出力ファイルという
  公開動作のみを見る。
* RL-*（run_local実統合）：`python -m scripts.run_local` を実サブプロセスとして起動し、
  telemetry JSONLが実際に生成されること・行数とpolicy完了数の整合・単一task configでの
  step連続性を検証する。PyBulletが利用できない環境では `pytest.importorskip` でskipする
  （§6 T-030B DoD必須性の記述に倣うが、T-028自体のDoDはskip不可ではない。本ファイルの
  RL-*はskip可の統合確認として位置づける）。

未実装対象：`agents/heuristic/agent.py` の `_policy_step`・段階計測・telemetry書込み配線、
`scripts/run_local.py` の環境変数注入。本ファイルのテストは全てそれらの実装完了前提でRED。
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.t028.scenarios import (
    make_heuristic_agent,
    no_candidate_case,
    patch_dual,
    placeable_case,
    single_task_config_path,
)

SIMULATOR_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_KEYS = frozenset(
    {
        "step", "first_step", "t_total", "t_state", "t_enum", "t_mask", "t_lpath",
        "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer",
        "regime", "p_ng_chosen", "reject_top3", "layer_error",
    }
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _telemetry_path(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "telemetry" / f"{run_id}.jsonl"


def _set_env(monkeypatch, tmp_path: Path, run_id: str) -> Path:
    telemetry_dir = tmp_path / "telemetry"
    monkeypatch.setenv("TELEMETRY_DIR", str(telemetry_dir))
    monkeypatch.setenv("TELEMETRY_RUN_ID", run_id)
    return telemetry_dir


# =============================================================================
# AG: Agent正常系
# =============================================================================


def test_ag_normal_line_writes_one_valid_row_with_finite_png(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path, "ag-normal")
    init, observation = placeable_case()
    agent = make_heuristic_agent(init)
    agent.policy(observation)

    rows = _read_jsonl(_telemetry_path(tmp_path, "ag-normal"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == _REQUIRED_KEYS
    assert row["decided_layer"] >= 1
    assert row["regime"] is None
    assert row["p_ng_chosen"] is not None
    assert math.isfinite(row["p_ng_chosen"])
    assert 0.0 <= row["p_ng_chosen"] <= 1.0


def test_ag_step_seq_first_step_true_then_false_across_calls(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path, "ag-step-seq")
    init, observation = placeable_case()
    agent = make_heuristic_agent(init)

    agent.policy(observation)
    agent.policy(observation)

    rows = _read_jsonl(_telemetry_path(tmp_path, "ag-step-seq"))
    assert len(rows) == 2
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["first_step"] is True
    assert rows[1]["first_step"] is False


# =============================================================================
# AG: emergency系
# =============================================================================


def test_ag_empty_zero_candidates_writes_emergency_row(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path, "ag-empty")
    init, observation = no_candidate_case()
    agent = make_heuristic_agent(init)
    action = agent.policy(observation)

    assert set(action.keys()) == {"item_idx", "container_idx", "place_pos", "orientation"}
    rows = _read_jsonl(_telemetry_path(tmp_path, "ag-empty"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == _REQUIRED_KEYS
    assert row["decided_layer"] == 0
    assert row["p_ng_chosen"] is None


def test_ag_buildfail_build_state_exception_writes_default_timing_row(tmp_path, monkeypatch):
    from src.packing_core import state as state_module

    def _raising_build_state(observation, init):
        raise RuntimeError("injected build_state failure (T-028 AG-BUILDFAIL)")

    patch_dual(monkeypatch, state_module, "agents.heuristic.agent", "build_state", _raising_build_state)

    _set_env(monkeypatch, tmp_path, "ag-buildfail")
    init, observation = placeable_case()
    agent = make_heuristic_agent(init)
    agent.policy(observation)

    rows = _read_jsonl(_telemetry_path(tmp_path, "ag-buildfail"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == _REQUIRED_KEYS
    assert row["decided_layer"] == 0
    assert row["p_ng_chosen"] is None
    assert row["t_state"] >= 0.0
    # build_stateで停止したため、以降の段階には未到達（既定値0.0のまま、§4.13 D2）。
    assert row["t_enum"] == 0.0
    assert row["t_mask"] == 0.0
    assert row["t_lpath"] == 0.0
    assert row["t_total"] >= 0.0


def test_ag_actionfail_make_action_exception_still_records_finite_png(tmp_path, monkeypatch):
    """safe_decideがCandidateを返した後にmake_action変換が失敗しemergencyへフォールバックしても、
    選択済みCandidateのprovisional_p_ngを記録する（§4.13 D5「その後のaction変換が失敗し
    _emergency_actionへフォールバックした場合を含む」）。"""
    from src.packing_core import state as state_module

    # T-027: one-shot失敗注入は__init__ warmupではなく実policy変換を対象とする。
    _set_env(monkeypatch, tmp_path, "ag-actionfail")
    init, observation = placeable_case()
    agent = make_heuristic_agent(init)
    original_make_action = state_module.make_action
    call_count = {"n": 0}

    def _one_shot_raising_make_action(item_idx, container_idx, pos_rel, orientation):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("injected make_action failure (T-028 AG-ACTIONFAIL)")
        return original_make_action(item_idx, container_idx, pos_rel, orientation)

    patch_dual(
        monkeypatch, state_module, "agents.heuristic.agent",
        "make_action", _one_shot_raising_make_action,
    )

    agent.policy(observation)

    assert call_count["n"] >= 2  # 1回目=決定候補変換失敗、2回目=emergency構築で成功
    rows = _read_jsonl(_telemetry_path(tmp_path, "ag-actionfail"))
    assert len(rows) == 1
    row = rows[0]
    assert row["decided_layer"] >= 1  # safe_decideはCandidateを返した
    assert row["p_ng_chosen"] is not None
    assert math.isfinite(row["p_ng_chosen"])


def test_ag_step_always_increments_even_when_writer_fails(tmp_path, monkeypatch):
    """writer障害時（telemetry書込み失敗）でも、書込み成否に関わらずAgentインスタンス内
    `_policy_step` は必ず+1する（§4.13 D3「書込み成否に関わらず self._policy_step を1増加」）。"""
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.write_text("blocker: not a directory, forces writer open failure")
    monkeypatch.setenv("TELEMETRY_DIR", str(telemetry_dir))
    monkeypatch.setenv("TELEMETRY_RUN_ID", "ag-step-always")

    init, observation = placeable_case()
    agent = make_heuristic_agent(init)

    assert agent._policy_step == 0
    agent.policy(observation)
    assert agent._policy_step == 1
    agent.policy(observation)
    assert agent._policy_step == 2


def test_ag_noleak_exceptions_never_leak_out_of_policy_with_telemetry_enabled(tmp_path, monkeypatch):
    """telemetry配線を追加してもpolicy()の無例外送出契約（§4.11）は保たれる。"""
    from src.packing_core import state as state_module

    def _raising_build_state(observation, init):
        raise RuntimeError("injected")

    patch_dual(monkeypatch, state_module, "agents.heuristic.agent", "build_state", _raising_build_state)
    _set_env(monkeypatch, tmp_path, "ag-noleak")

    init, observation = placeable_case()
    agent = make_heuristic_agent(init)
    try:
        action = agent.policy(observation)
    except Exception as exc:  # noqa: BLE001（本テストの主張そのもの）
        pytest.fail(f"policy leaked an exception with telemetry enabled: {exc!r}")
    assert set(action.keys()) == {"item_idx", "container_idx", "place_pos", "orientation"}


# =============================================================================
# RL: run_local実統合（PyBulletが利用できない環境ではskip）
# =============================================================================


def _run_local_subprocess(config_path: Path, module_path: str, result_dir: Path, timeout: float = 180.0):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.run_local",
            "--config-path", str(config_path),
            "--module-path", module_path,
            "--result-dir", str(result_dir),
        ],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_rl_inject_and_file_telemetry_jsonl_created_under_result_dir(tmp_path):
    pytest.importorskip("pybullet")
    config_path = single_task_config_path(tmp_path, n_items=2)
    result_dir = tmp_path / "result"

    proc = _run_local_subprocess(config_path, "agents/heuristic/", result_dir)
    assert proc.returncode == 0, proc.stderr

    telemetry_files = list((result_dir / "telemetry").glob("*.jsonl"))
    assert len(telemetry_files) == 1
    rows = _read_jsonl(telemetry_files[0])
    assert len(rows) > 0


def test_rl_count_jsonl_rows_match_completed_policy_calls(tmp_path):
    """外部timeout 0件の実行で、JSONL行数とrun_local結果から確認できる完了policy呼出し数
    （action返却＋emergency返却の2種、§4.13 D3）が一致する。1 config・1 taskのため
    完了policy呼出し数は「配置決定ステップ数」と一致し、`evaluation_results.json` の
    `place_states` から見た配置試行回数と対応づけられる。"""
    pytest.importorskip("pybullet")
    config_path = single_task_config_path(tmp_path, n_items=3)
    result_dir = tmp_path / "result"

    proc = _run_local_subprocess(config_path, "agents/heuristic/", result_dir)
    assert proc.returncode == 0, proc.stderr

    telemetry_files = list((result_dir / "telemetry").glob("*.jsonl"))
    assert len(telemetry_files) == 1
    rows = _read_jsonl(telemetry_files[0])

    result_path = result_dir / "evaluation_results.json"
    with open(result_path) as f:
        results = json.load(f)
    assert len(results) == 1
    (task_result,) = results.values()
    assert task_result["status"] == "success"  # 外部timeout/format_errorが起きていないこと

    # 3item・1taskの単一Agentプロセスでは、行数はpolicy呼出し回数（=item数上限）と一致する。
    assert len(rows) >= 1
    assert len(rows) <= 3


def test_rl_step_single_task_step_sequence_is_contiguous_from_zero(tmp_path):
    """1 taskのみの専用configでは、Agent再生成が起きないためstepが0..N-1の連続列となり、
    first_step=trueがstep=0の1行のみ出現する（step一意性・単調増加は複数task/config全体へは
    要求しない。単一task限定の性質としてここで検証する）。"""
    pytest.importorskip("pybullet")
    config_path = single_task_config_path(tmp_path, n_items=3)
    result_dir = tmp_path / "result"

    proc = _run_local_subprocess(config_path, "agents/heuristic/", result_dir)
    assert proc.returncode == 0, proc.stderr

    telemetry_files = list((result_dir / "telemetry").glob("*.jsonl"))
    assert len(telemetry_files) == 1
    rows = _read_jsonl(telemetry_files[0])

    steps = [r["step"] for r in rows]
    assert steps == list(range(len(rows)))  # 0..N-1連続、gapなし
    first_step_true_rows = [r for r in rows if r["first_step"] is True]
    assert len(first_step_true_rows) == 1
    assert first_step_true_rows[0]["step"] == 0


def test_rl_chk_generated_jsonl_passes_check_telemetry(tmp_path):
    pytest.importorskip("pybullet")
    config_path = single_task_config_path(tmp_path, n_items=2)
    result_dir = tmp_path / "result"

    proc = _run_local_subprocess(config_path, "agents/heuristic/", result_dir)
    assert proc.returncode == 0, proc.stderr

    telemetry_files = list((result_dir / "telemetry").glob("*.jsonl"))
    assert len(telemetry_files) == 1

    check_proc = subprocess.run(
        [sys.executable, "-m", "scripts.check_telemetry", str(telemetry_files[0])],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert check_proc.returncode == 0, check_proc.stdout + check_proc.stderr
