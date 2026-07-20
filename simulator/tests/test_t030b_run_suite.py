"""T-030B: `scripts/run_suite.py` 契約テスト（詳細仕様書 v1.19/v1.20 §5.5、契約テスト設計計画）。

`scripts/run_suite.py` は本ファイル authoring 時点で未実装のため、`scripts.run_suite` の
import は各テスト関数（またはfixture）内へ遅延する（`test_t024_import_contract.py` と
同方針）。これにより `pytest --collect-only` は実装前でも成功し、実行のみが RED になる。

**方針（黒箱テスト）**：`run_suite`/`main`/CLI/生成物schema/exit code のみを公開契約として
検証し、内部の補助関数名・シグネチャはテストで固定しない（ユーザー承認条件）。
公開契約の検証点：

  - `scripts.run_suite.run_suite(config_dir, module_path, result_dir, baseline=None,
    timeout_sec=600.0) -> int`（`sys.exit` しない中核関数）
  - `scripts.run_suite.main(argv=None) -> None`（`run_suite` の戻り値で `sys.exit`）
  - 生成物：`<result_dir>/suite_results.jsonl`・`<result_dir>/suite_summary.json`・
    `<result_dir>/suite_report.md`
  - `scripts.run_suite.subprocess`（モジュール名前空間の `subprocess` 参照）への spy
    （`run_local.py` 経路の再利用検証。ソース文字列検査は行わない）

契約ID接頭辞 `T030B-`。各セクション見出しに対応する契約IDを記載する。
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.t030b.fakes import (
    ConfigAction,
    FakeRunPlan,
    format_error_result,
    make_baseline_rows,
    make_row,
    multi_task_result,
    nan_inf_result,
    success_result,
    validator_ng_result,
    write_jsonl,
    write_stub_configs,
)

SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
AGENT_MODULE_PATH = "agents/base/"
AGENT_NORMALIZED = "agents/base"  # normalize_agent("agents/base/") 契約（末尾 / 除去）


def _import_run_suite():
    """`scripts.run_suite` を遅延importする（未実装時は各テストがRED、collectionは成功）。"""
    return importlib.import_module("scripts.run_suite")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_by_config(rows: list[dict], config: str) -> list[dict]:
    return [r for r in rows if r["config"] == config]


# =============================================================================
# ARGS: CLI引数・入力検証（T030B-ARGS-001〜007）
# =============================================================================


def test_args_001_required_flags_missing_exit_2():
    """必須3フラグ（--config-dir/--module-path/--result-dir）欠落は argparse により exit 2。"""
    run_suite = _import_run_suite()
    with pytest.raises(SystemExit) as exc_info:
        run_suite.main([])
    assert exc_info.value.code == 2


def test_args_002_result_dir_has_no_default(tmp_path):
    """`--result-dir` は既定値を持たない（省略時は exit 2、追跡済み results/ を汚さない）。"""
    run_suite = _import_run_suite()
    write_stub_configs(tmp_path / "cfgs", ["c01"])
    with pytest.raises(SystemExit) as exc_info:
        run_suite.main(
            ["--config-dir", str(tmp_path / "cfgs"), "--module-path", AGENT_MODULE_PATH]
        )
    assert exc_info.value.code == 2


def test_args_003_timeout_sec_default_600(tmp_path):
    """`--timeout-sec` 省略時、`run_suite` へ渡る既定値は 600.0（spyのtimeout kwargで観測）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})

    import scripts.run_suite as rs_mod  # noqa: F401 (モジュール名前空間を得るため)

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        code = run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
        )
    finally:
        rs_mod.subprocess.run = orig_run

    assert code == 0
    assert len(plan.calls) == 1
    assert plan.calls[0]["kwargs"].get("timeout") == 600.0


@pytest.mark.parametrize(
    "bad_value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
    ids=["zero", "negative", "nan", "posinf", "neginf"],
)
def test_args_004_timeout_sec_must_be_finite_positive(tmp_path, bad_value):
    """`--timeout-sec` が非finiteまたは<=0ならCLI入力エラー（exit 2）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"

    code = run_suite.run_suite(
        config_dir=str(config_dir),
        module_path=AGENT_MODULE_PATH,
        result_dir=str(result_dir),
        timeout_sec=bad_value,
    )
    assert code == 2


def test_args_005_config_dir_not_exist_exit_2(tmp_path):
    """`--config-dir` が存在しない場合 exit 2。"""
    run_suite = _import_run_suite()
    code = run_suite.run_suite(
        config_dir=str(tmp_path / "does-not-exist"),
        module_path=AGENT_MODULE_PATH,
        result_dir=str(tmp_path / "result"),
    )
    assert code == 2


def test_args_006_zero_configs_discovered_exit_2(tmp_path):
    """`--config-dir` は存在するが `c*.json` が0件なら exit 2。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    config_dir.mkdir()
    (config_dir / "readme.txt").write_text("not a config")
    code = run_suite.run_suite(
        config_dir=str(config_dir),
        module_path=AGENT_MODULE_PATH,
        result_dir=str(tmp_path / "result"),
    )
    assert code == 2


def test_args_007_baseline_is_optional(tmp_path):
    """`--baseline` 省略時も正常動作する（必須ではない）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})

    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        code = run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
        )
    finally:
        rs_mod.subprocess.run = orig_run
    assert code == 0


# =============================================================================
# DISC: 発見・決定論順序（T030B-DISC-001〜003）
# =============================================================================


def test_disc_001_002_003_discovery_order_and_filter(tmp_path):
    """`c*.json` をファイル名昇順で発見・実行し、非マッチファイルは対象外（投入順に非依存）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    # 投入順をあえて逆順・非マッチファイル混在にする
    write_stub_configs(config_dir, ["c03", "c01", "c02"])
    (config_dir / "readme.txt").write_text("ignore me")
    (config_dir / "manifest.json").write_text("{}")  # "c*.json" に非マッチ
    result_dir = tmp_path / "result"

    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(task_id="a")),
            "c02": ConfigAction(kind="success", result_json=success_result(task_id="a")),
            "c03": ConfigAction(kind="success", result_json=success_result(task_id="a")),
        }
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        code = run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
        )
    finally:
        rs_mod.subprocess.run = orig_run

    assert code == 0
    stems_called = [Path(c["args"][c["args"].index("--config-path") + 1]).stem for c in plan.calls]
    assert stems_called == ["c01", "c02", "c03"]  # 昇順、投入順(c03,c01,c02)に非依存
    assert "manifest" not in stems_called
    assert len(plan.calls) == 3  # readme.txt/manifest.json は対象外


# =============================================================================
# SUBP: run_local再利用・subprocess spy（T030B-SUBP-001〜005）
# =============================================================================


def test_subp_001_002_003_004_argv_shell_timeout_and_call_count(tmp_path):
    """argv構造・`shell=False`・`timeout=timeout_sec`・呼出し回数と順序を spy で観測する。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result()),
            "c02": ConfigAction(kind="success", result_json=success_result()),
        }
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        code = run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
            timeout_sec=123.0,
        )
    finally:
        rs_mod.subprocess.run = orig_run

    assert code == 0
    assert len(plan.calls) == 2  # SUBP-004: config数分だけ1回ずつ

    for i, stem in enumerate(["c01", "c02"]):  # SUBP-004: 昇順
        call = plan.calls[i]
        args = call["args"]
        assert args[0] == sys.executable
        assert args[1] == "-m"
        assert args[2] == "scripts.run_local"
        assert "--config-path" in args
        assert "--module-path" in args
        assert "--result-dir" in args
        cfg_val = args[args.index("--config-path") + 1]
        assert Path(cfg_val).name == f"{stem}.json"
        mod_val = args[args.index("--module-path") + 1]
        assert mod_val == AGENT_MODULE_PATH
        result_val = args[args.index("--result-dir") + 1]
        assert Path(result_val).name == stem  # <suite-result-dir>/<cN>
        assert call["kwargs"].get("shell", False) is False  # SUBP-002
        assert call["kwargs"].get("timeout") == 123.0  # SUBP-003


def test_subp_005_run_local_cli_exists_guard():
    """GUARD/前提保護：`run_local.py` の CLI フラグが存在する（`--help` 実行で観測、source非依存）。"""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_local", "--help"],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for flag in ("--config-path", "--module-path", "--result-dir"):
        assert flag in result.stdout


# =============================================================================
# CONT: 継続実行（T030B-CONT-001〜003）
# =============================================================================


@pytest.mark.parametrize(
    "failing_kind",
    ["nonzero", "timeout", "missing_json", "broken_json"],
    ids=["nonzero-exit", "timeout-expired", "missing-json", "broken-json"],
)
def test_cont_continues_after_each_failure_mode(tmp_path, failing_kind):
    """あるconfigがnonzero/timeout/JSON欠落/parse不可のいずれでも、残りのconfigを継続する。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02", "c03"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(task_id="a")),
            "c02": ConfigAction(kind=failing_kind, returncode=13),
            "c03": ConfigAction(kind="success", result_json=success_result(task_id="a")),
        }
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
        )
    finally:
        rs_mod.subprocess.run = orig_run

    # 3configすべてが呼ばれた(=継続した)ことをspyで確認
    stems_called = [Path(c["args"][c["args"].index("--config-path") + 1]).stem for c in plan.calls]
    assert stems_called == ["c01", "c02", "c03"]

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    c02_rows = _rows_by_config(rows, "c02")
    assert len(c02_rows) == 1  # ROW-002/003: synthetic 1行のみ
    assert c02_rows[0]["status"] == "exec_exception"
    assert c02_rows[0]["task_id"] is None
    assert c02_rows[0]["exec_exception"] is True

    c01_rows = _rows_by_config(rows, "c01")
    c03_rows = _rows_by_config(rows, "c03")
    assert c01_rows[0]["status"] == "success"
    assert c03_rows[0]["status"] == "success"


# =============================================================================
# CLS: 3層タクソノミ分離（T030B-CLS-001〜004）
# =============================================================================


def test_cls_001_validator_ng_is_success_and_completed(tmp_path):
    """validator NG（status=='success' かつ place_states にFalse）は完走扱い、status行はsuccess。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {"c01": ConfigAction(kind="success", result_json=validator_ng_result(task_id="a"))}
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        run_suite.run_suite(
            config_dir=str(config_dir), module_path=AGENT_MODULE_PATH, result_dir=str(result_dir)
        )
    finally:
        rs_mod.subprocess.run = orig_run

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    row = _rows_by_config(rows, "c01")[0]
    assert row["status"] == "success"
    assert row["n_validator_ng"] >= 1

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["num_completed"] == 1  # COMP-002: validator NGは完走に含む
    assert summary["completion_rate"] == 1.0


def test_cls_002_format_error_is_uncompleted(tmp_path):
    """format_error（status=='format_error'）は未完走扱い。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {"c01": ConfigAction(kind="success", result_json=format_error_result(task_id="a"))}
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        run_suite.run_suite(
            config_dir=str(config_dir), module_path=AGENT_MODULE_PATH, result_dir=str(result_dir)
        )
    finally:
        rs_mod.subprocess.run = orig_run

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    row = _rows_by_config(rows, "c01")[0]
    assert row["status"] == "format_error"
    assert row["n_validator_ng"] is None  # ROW-010（Q3）

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["num_completed"] == 0
    assert summary["n_format_error_total"] == 1


def test_cls_003_exec_exception_is_uncompleted(tmp_path):
    """exec_exception（nonzero/JSON欠落/parse不可/timeout）は未完走扱い。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="missing_json")})
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        run_suite.run_suite(
            config_dir=str(config_dir), module_path=AGENT_MODULE_PATH, result_dir=str(result_dir)
        )
    finally:
        rs_mod.subprocess.run = orig_run

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["num_completed"] == 0
    assert summary["n_exec_exception_total"] == 1


def test_cls_004_three_kinds_mutually_exclusive_in_one_suite(tmp_path):
    """1回のsuite実行内で validator NG／format_error／exec_exception が混同されず分離される。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02", "c03"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=validator_ng_result(task_id="a")),
            "c02": ConfigAction(kind="success", result_json=format_error_result(task_id="a")),
            "c03": ConfigAction(kind="nonzero", returncode=1),
        }
    )
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        run_suite.run_suite(
            config_dir=str(config_dir), module_path=AGENT_MODULE_PATH, result_dir=str(result_dir)
        )
    finally:
        rs_mod.subprocess.run = orig_run

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    assert _rows_by_config(rows, "c01")[0]["status"] == "success"
    assert _rows_by_config(rows, "c02")[0]["status"] == "format_error"
    assert _rows_by_config(rows, "c03")[0]["status"] == "exec_exception"

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["num_completed"] == 1
    assert summary["n_format_error_total"] == 1
    assert summary["n_exec_exception_total"] == 1


# =============================================================================
# ROW/SAN: suite_results.jsonl 行粒度・schema・数値サニタイズ
# (T030B-ROW-001〜011, T030B-SAN-001〜003)
# =============================================================================

_REQUIRED_ROW_KEYS = {
    "config", "agent", "task_id", "status", "fill_score", "num_placed_items",
    "n_validator_ng", "format_error", "exec_exception", "policy_time",
    "optimization_time", "wall_time",
}


def _run_suite_with_plan(run_suite, config_dir, result_dir, plan, **kwargs):
    import scripts.run_suite as rs_mod

    orig_run = subprocess.run
    try:
        rs_mod.subprocess.run = plan
        return run_suite.run_suite(
            config_dir=str(config_dir),
            module_path=AGENT_MODULE_PATH,
            result_dir=str(result_dir),
            **kwargs,
        )
    finally:
        rs_mod.subprocess.run = orig_run


def test_row_001_003_task_rows_one_per_task_id(tmp_path):
    """正常/format_errorのtaskはtask_idごとに1行。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    result_json = multi_task_result(
        success_result(task_id="a"), success_result(task_id="b"),
        format_error_result(task_id="c"),
    )
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=result_json)})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _rows_by_config(_read_jsonl(result_dir / "suite_results.jsonl"), "c01")
    assert {r["task_id"] for r in rows} == {"a", "b", "c"}
    assert len(rows) == 3


def test_row_002_003_exec_exception_synthetic_row_min_one_per_config(tmp_path):
    """exec_exceptionはconfigごとに1 synthetic行（task_id=null）。各configは最低1行を保証。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="missing_json")})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _rows_by_config(_read_jsonl(result_dir / "suite_results.jsonl"), "c01")
    assert len(rows) == 1
    assert rows[0]["task_id"] is None
    assert rows[0]["status"] == "exec_exception"


def test_row_004_status_is_one_of_three_values(tmp_path):
    """status は success/format_error/exec_exception の3値のみ。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02", "c03"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result()),
            "c02": ConfigAction(kind="success", result_json=format_error_result()),
            "c03": ConfigAction(kind="nonzero"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    assert {r["status"] for r in rows} <= {"success", "format_error", "exec_exception"}


def test_row_005_009_row_keys_and_types(tmp_path):
    """行キーが完全一致し、各値の型契約（bool/str-or-null/float-or-null）を満たす。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(task_id="a"))})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    row = rows[0]
    assert set(row.keys()) == _REQUIRED_ROW_KEYS
    assert isinstance(row["task_id"], str)
    assert isinstance(row["format_error"], bool)
    assert isinstance(row["exec_exception"], bool)
    assert isinstance(row["wall_time"], float)
    assert row["wall_time"] is not None


def test_row_006_missing_numbers_are_null(tmp_path):
    """format_error行では欠損する数値（fill_score/num_placed_items/policy_time/optimization_time）がnull。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=format_error_result(task_id="a"))})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    row = _read_jsonl(result_dir / "suite_results.jsonl")[0]
    assert row["fill_score"] is None
    assert row["num_placed_items"] is None
    assert row["policy_time"] is None
    assert row["optimization_time"] is None
    assert row["n_validator_ng"] is None


def test_row_007_task_id_is_str_or_null(tmp_path):
    """task_id は str または null（synthetic行）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(task_id="000")),
            "c02": ConfigAction(kind="missing_json"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)
    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    for r in rows:
        assert r["task_id"] is None or isinstance(r["task_id"], str)


def test_row_008_agent_normalized_across_all_rows(tmp_path):
    """agentは全行で normalize_agent(module_path)（末尾 '/' 除去）と一致する。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result()),
            "c02": ConfigAction(kind="missing_json"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)
    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    assert all(r["agent"] == AGENT_NORMALIZED for r in rows)


def test_row_010_n_validator_ng_null_for_format_and_exec(tmp_path):
    """n_validator_ng は format_error/exec_exception 行で null（Q3確定）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=format_error_result()),
            "c02": ConfigAction(kind="missing_json"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)
    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    assert all(r["n_validator_ng"] is None for r in rows)


def test_row_011_wall_time_replicated_across_config_rows(tmp_path):
    """wall_time はconfig単位subprocess時間を同config全行へ同値で複製する。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    result_json = multi_task_result(success_result(task_id="a"), success_result(task_id="b"))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=result_json)})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _rows_by_config(_read_jsonl(result_dir / "suite_results.jsonl"), "c01")
    wall_times = {r["wall_time"] for r in rows}
    assert len(wall_times) == 1  # 全行同値


def test_san_001_002_003_nan_inf_become_null_and_valid_json(tmp_path):
    """NaN/±Inf は書き出し前にnullへ変換され、出力JSONLは標準JSON（NaN/Infinityトークン非混入）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=nan_inf_result(task_id="a"))})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    jsonl_path = result_dir / "suite_results.jsonl"
    raw_text = jsonl_path.read_text()
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text  # +Infinity/-Infinity 双方の部分文字列を含めて非混入

    row = _read_jsonl(jsonl_path)[0]  # json.loads が例外なく通ることも SAN-003 の一部
    assert row["fill_score"] is None
    assert row["policy_time"] is None
    assert row["optimization_time"] is None


# =============================================================================
# SUM/COMP/SCORE: summary集計・完走定義・score定義
# (T030B-SUM-001〜005, T030B-COMP-001〜004, T030B-SCORE-001〜004)
# =============================================================================

_REQUIRED_SUMMARY_KEYS = {
    "agent", "num_discovered", "num_completed", "completion_rate",
    "n_validator_ng_total", "n_format_error_total", "n_exec_exception_total",
    "mean_fill", "suite_score", "compared_config_stems",
}


def test_sum_001_summary_keys_exact(tmp_path):
    """suite_summary.json のキーは固定10キーと完全一致する（v1.20確定）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert set(summary.keys()) == _REQUIRED_SUMMARY_KEYS


def test_sum_002_agent_recorded(tmp_path):
    """`agent`（正規化識別子）が summary へ記録される。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["agent"] == AGENT_NORMALIZED


def test_sum_003_totals_match_row_aggregation(tmp_path):
    """各totalが行集計（n_validator_ng_totalは非null行の総和）と一致する。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02", "c03"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=validator_ng_result(task_id="a", n_false=2)),
            "c02": ConfigAction(kind="success", result_json=format_error_result(task_id="a")),
            "c03": ConfigAction(kind="nonzero"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    rows = _read_jsonl(result_dir / "suite_results.jsonl")
    expected_ng_total = sum(r["n_validator_ng"] for r in rows if r["n_validator_ng"] is not None)
    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["n_validator_ng_total"] == expected_ng_total
    assert summary["n_format_error_total"] == sum(1 for r in rows if r["status"] == "format_error")
    assert summary["n_exec_exception_total"] == sum(1 for r in rows if r["status"] == "exec_exception")


def test_sum_004_compared_config_stems_sorted_unique(tmp_path):
    """`compared_config_stems` は昇順unique listである。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c03", "c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {s: ConfigAction(kind="success", result_json=success_result()) for s in ("c01", "c02", "c03")}
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    stems = summary["compared_config_stems"]
    assert stems == sorted(set(stems))
    assert stems == ["c01", "c02", "c03"]


def test_sum_005_mean_fill_equals_suite_score(tmp_path):
    """mean_fill と suite_score は同一値。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(fill_score=60.0)),
            "c02": ConfigAction(kind="nonzero"),  # 未完走 -> 0.0 として平均へ算入
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["mean_fill"] == summary["suite_score"]
    assert summary["mean_fill"] == pytest.approx((60.0 + 0.0) / 2)


def test_comp_001_002_003_004_completion_definition_and_rate(tmp_path):
    """完走定義（exit0∧JSON可∧task>=1∧全success）とcompletion_rateの算出。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02", "c03", "c04"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result()),  # 完走
            "c02": ConfigAction(kind="success", result_json=validator_ng_result()),  # 完走(NG含む)
            "c03": ConfigAction(kind="success", result_json=format_error_result()),  # 未完走
            "c04": ConfigAction(kind="nonzero"),  # 未完走
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["num_discovered"] == 4
    assert summary["num_completed"] == 2
    assert summary["completion_rate"] == pytest.approx(0.5)


def test_score_001_002_003_004_fill_score_aggregation(tmp_path):
    """完走configは有限fill_score平均、未完走は0.0、suite_score=全config平均=mean_fill。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    result_json_c01 = multi_task_result(
        success_result(task_id="a", fill_score=80.0), success_result(task_id="b", fill_score=40.0)
    )
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=result_json_c01),  # 平均60.0
            "c02": ConfigAction(kind="missing_json"),  # 未完走 -> 0.0
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    summary = json.loads((result_dir / "suite_summary.json").read_text())
    assert summary["suite_score"] == pytest.approx((60.0 + 0.0) / 2)
    assert summary["mean_fill"] == summary["suite_score"]


# =============================================================================
# REP: レポート生成（T030B-REP-001〜005）
# =============================================================================


def test_rep_001_002_jsonl_and_summary_written(tmp_path):
    """`suite_results.jsonl` と `suite_summary.json` が result_dir へ出力される。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    assert (result_dir / "suite_results.jsonl").exists()
    assert (result_dir / "suite_summary.json").exists()


def test_rep_003_005_report_md_columns_and_row_count(tmp_path, capsys):
    """`suite_report.md` は必須9列・record毎1行・null='-'、stdoutにも出力される。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(task_id="a")),
            "c02": ConfigAction(kind="missing_json"),
        }
    )
    _run_suite_with_plan(run_suite, config_dir, result_dir, plan)

    report_path = result_dir / "suite_report.md"
    assert report_path.exists()
    md_text = report_path.read_text()

    required_cols = [
        "config", "task_id", "status", "n_validator_ng", "num_placed_items",
        "fill_score", "policy_time", "optimization_time", "wall_time",
    ]
    for col in required_cols:
        assert col in md_text

    jsonl_rows = _read_jsonl(result_dir / "suite_results.jsonl")
    md_lines = [ln for ln in md_text.splitlines() if ln.strip()]
    # ヘッダ・区切り行を除いた本体行数がJSONL record数と一致(緩検査：値の網羅を確認)
    body_lines = [ln for ln in md_lines if ln.startswith("|") and "config" not in ln and "---" not in ln]
    assert len(body_lines) == len(jsonl_rows)

    # exec_exceptionのnull値が "-" 表示されている
    exec_line = next(ln for ln in body_lines if "c02" in ln)
    assert "-" in exec_line

    captured = capsys.readouterr()
    assert "c01" in captured.out  # 人間可読レポートはstdoutへも出力される（緩検査：主要値の存在）


def test_rep_004_report_write_failure_exit_2(tmp_path):
    """レポート出力エラー（`suite_results.jsonl` の書込み先が塞がっている）は exit 2。

    per-config subprocess（fake）は正常な `result_dir` 配下で問題なく書き込めるようにし、
    最終レポート書出し段階だけを失敗させて分離する（`result_dir` 自体をファイルにすると
    per-config subprocessの結果JSON書込み自体も失敗し exec_exception と混同するため避ける）。
    """
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "suite_results.jsonl").mkdir()  # 書込み先をディレクトリで塞ぐ

    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})
    code = _run_suite_with_plan(run_suite, config_dir, result_dir, plan)
    assert code == 2


# =============================================================================
# BASE: --baseline 比較契約（T030B-BASE-001〜008、v1.20 stem重複条項の訂正含む）
# =============================================================================


def test_base_002_agent_mismatch_exit_2(tmp_path):
    """baselineの agent 識別子が現在の agent と不一致なら exit 2。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent="agents/OTHER", stems=["c01"]))

    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result())})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 2


def test_base_003_stem_set_mismatch_exit_2(tmp_path):
    """baselineのunique config stem集合が現在の発見config集合と不一致（過不足）なら exit 2。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    # baselineはc01のみ(c02が不足) -> 過不足あり
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"]))

    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result()),
            "c02": ConfigAction(kind="success", result_json=success_result()),
        }
    )
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 2


def test_base_003_revoked_duplicate_stem_rows_are_normal(tmp_path):
    """訂正：同一config stemの複数行（task毎行）は正常。stem重複自体はエラーではない。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    # c01について2行(task 000/001)保持 -> stem重複行だが正常
    write_jsonl(
        baseline_path,
        [
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id="000", status="success", fill_score=50.0),
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id="001", status="success", fill_score=50.0),
        ],
    )
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=50.0))})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 0  # schema不正でもstem不一致でもない


@pytest.mark.parametrize(
    "bad_rows_builder",
    [
        # (i) 同一 (config, task_id) の重複行
        lambda: [
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id="000", status="success", fill_score=50.0),
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id="000", status="success", fill_score=50.0),
        ],
        # (ii) 同一configにsynthetic task_id=null行が複数
        lambda: [
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id=None, status="exec_exception", exec_exception=True),
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id=None, status="exec_exception", exec_exception=True),
        ],
        # (iii) synthetic exec_exception行と通常task行が同一configに混在
        lambda: [
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id=None, status="exec_exception", exec_exception=True),
            make_row(config="c01", agent=AGENT_NORMALIZED, task_id="000", status="success", fill_score=50.0),
        ],
    ],
    ids=["dup-config-task_id", "dup-synthetic-null", "synthetic-and-normal-mixed"],
)
def test_base_004_schema_invalid_baseline_exit_2(tmp_path, bad_rows_builder):
    """schema不正baseline：(i)(ii)(iii) のいずれかに該当すれば exit 2。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, bad_rows_builder())

    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=50.0))})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 2


def test_base_005_regression_completion_rate_drop(tmp_path):
    """退行(a)：baseline全完走・currentが未完走configを含む -> completion_rate低下で退行(exit 1)。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01", "c02"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(
        baseline_path,
        make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01", "c02"], mean_fill_per_stem={"c01": 50.0, "c02": 50.0}),
    )
    plan = FakeRunPlan(
        {
            "c01": ConfigAction(kind="success", result_json=success_result(fill_score=50.0)),
            "c02": ConfigAction(kind="nonzero"),  # 未完走化
        }
    )
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 1


def test_base_006_regression_mean_fill_drop_over_1pt(tmp_path):
    """退行(b)：baseline_mean_fill - current_mean_fill > 1.0 で退行(exit 1)。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 50.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=48.0))})  # -2.0pt
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 1


def test_base_007_boundary_exactly_1pt_is_not_regression(tmp_path):
    """境界：ちょうど1.0pt低下（==1.0）は退行としない（exit 0）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 50.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=49.0))})  # ちょうど-1.0
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 0


def test_base_008_non_regression_exit_0(tmp_path):
    """前提充足かつ非退行 -> exit 0。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 50.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=55.0))})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 0


# =============================================================================
# EXIT: exit code 0/1/2（T030B-EXIT-001〜006）
# =============================================================================


# T030B-EXIT-001（レポート成功かつbaseline未指定 or 非退行 -> exit 0）は
# test_args_007_baseline_is_optional / test_base_008_non_regression_exit_0 で検証済み（重複回避）。


def test_exit_002_regression_returns_1(tmp_path):
    """退行時は exit 1（BASE系と同一シナリオの再確認）。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 50.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=40.0))})
    code = _run_suite_with_plan(run_suite, config_dir, result_dir, plan, baseline=str(baseline_path))
    assert code == 1


def test_exit_006_priority_2_over_1_invalid_timeout_with_regressing_baseline(tmp_path):
    """優先順位 2>1：timeout不正（本来2）は、たとえbaselineが退行を示唆していても2のまま。"""
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 90.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=10.0))})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path), timeout_sec=-1.0
    )
    assert code == 2  # timeout不正が優先


def test_exit_006_priority_2_over_1_report_error_with_regressing_baseline(tmp_path):
    """優先順位 2>1：レポート出力エラー（本来2）は、baselineが退行を示唆していても2のまま。

    `result_dir` 自体ではなく `suite_results.jsonl` の書込み先だけを塞ぎ、per-config
    subprocess（fake）の正常動作とレポート出力エラーを分離する（test_rep_004と同じ理由）。
    """
    run_suite = _import_run_suite()
    config_dir = tmp_path / "cfgs"
    write_stub_configs(config_dir, ["c01"])
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "suite_results.jsonl").mkdir()  # 書込み先をディレクトリで塞ぐ
    baseline_path = tmp_path / "baseline.jsonl"
    write_jsonl(baseline_path, make_baseline_rows(agent=AGENT_NORMALIZED, stems=["c01"], mean_fill_per_stem={"c01": 90.0}))
    plan = FakeRunPlan({"c01": ConfigAction(kind="success", result_json=success_result(fill_score=10.0))})
    code = _run_suite_with_plan(
        run_suite, config_dir, result_dir, plan, baseline=str(baseline_path)
    )
    assert code == 2


# =============================================================================
# GUARD: 既存前提保護（現状GREEN、T030B-GUARD-001〜002）
# =============================================================================


def test_guard_001_c01_config_valid():
    """`configs/local_suite/c01.json` が存在し必須キーを持つ有効JSON。"""
    path = SIMULATOR_ROOT / "configs" / "local_suite" / "c01.json"
    assert path.exists()
    cfg = json.loads(path.read_text())
    task = next(iter(cfg.values()))
    for key in ("containers", "item_stream", "agent", "validator"):
        assert key in task


def test_guard_002_run_local_py_exists():
    """`scripts/run_local.py` が存在する（run_suite再利用対象の前提）。"""
    assert (SIMULATOR_ROOT / "scripts" / "run_local.py").exists()


# =============================================================================
# C05: 途中積付あり構築・二重確認（T030B-C05-001〜005、DoD必須・skip不可）
# =============================================================================


def _load_c05_container_cfg() -> dict:
    path = SIMULATOR_ROOT / "configs" / "local_suite" / "c05.json"
    cfg = json.loads(path.read_text())
    task = next(iter(cfg.values()))
    return task


def test_c05_001_static_contract():
    """静的契約：1台/look_ahead=5/optimize=false/allowed_methodsにoptimize無/packed_items1件/
    orn=identity/indexが item_stream と非重複で一意。"""
    task = _load_c05_container_cfg()
    assert len(task["containers"]["container_list"]) == 1
    assert task["item_stream"]["look_ahead"] == 5
    assert task["agent"]["optimize"] is False
    assert "optimize" not in task["agent"]["allowed_methods"]

    packed = task["containers"]["container_list"][0]["packed_items"]
    assert len(packed) == 1
    pre_item = packed[0]
    assert pre_item["orn"] == [0.0, 0.0, 0.0, 1.0]

    stream_indices = {item["index"] for item in task["item_stream"]["item_list"]}
    assert pre_item["index"] not in stream_indices


def test_c05_002_position_derived_from_geometry_not_guessed():
    """pos が実コンテナ形状・床面高さ・箱寸法から再導出した値と一致する（推測固定でない）。"""
    from src.packing_core.constants import GridParams
    from src.packing_core.container_space import build_container_space
    from tests.fixtures.container_space_golden import build_cdict_from_raw_config

    task = _load_c05_container_cfg()
    container_cfg = task["containers"]["container_list"][0]
    pre_item = container_cfg["packed_items"][0]

    raw_config = {
        "length": container_cfg["length"], "width": container_cfg["width"],
        "height": container_cfg["height"], "thickness": container_cfg["thickness"],
        "cut_x": container_cfg["cut_x"], "cut_y": container_cfg["cut_y"],
        "buffer": container_cfg["buffer"], "require_shelf": container_cfg["require_shelf"],
    }
    cdict = build_cdict_from_raw_config(raw_config, offset_x=0.0, index=0)
    space = build_container_space(cdict, index=0, cell=GridParams().cell)

    x_rel, y_rel, z_rel = pre_item["pos"]
    half_l, half_w, half_h = pre_item["length"] / 2, pre_item["width"] / 2, pre_item["height"] / 2

    # XY: interior AABB内に、実際の箱半径分の余裕を持って収まる(cutを含む実形状ベース)
    assert space.inner_min_rel[0] + half_l <= x_rel <= space.inner_max_rel[0] - half_l
    assert space.inner_min_rel[1] + half_w <= y_rel <= space.inner_max_rel[1] - half_w

    # Z: 実測floor_z(x_rel,y_rel)（cut_planesの影響を反映）以上、天井未満
    floor_z = float(space.inner_min_rel[2])
    EPS_GEOM = 1e-9
    for normal_rel, d in space.cut_planes:
        nz = normal_rel[2]
        if nz < -EPS_GEOM:
            candidate = (d - normal_rel[0] * x_rel - normal_rel[1] * y_rel) / nz
            floor_z = max(floor_z, candidate)
    assert z_rel >= floor_z + half_h - 1e-6
    assert z_rel < float(space.inner_max_rel[2])


def test_c05_005_c02_to_c05_matrix_diversity():
    """c02〜c05が承認済みマトリクス（台数1-2/lookahead1・5・20/optimize有無/途中積付）を満たす。"""
    expected = {
        "c02": {"containers": 1, "look_ahead": 20, "optimize": False},
        "c03": {"containers": 2, "look_ahead": 5, "optimize": True},
        "c04": {"containers": 2, "look_ahead": 1, "optimize": False},
        "c05": {"containers": 1, "look_ahead": 5, "optimize": False},
    }
    for stem, expect in expected.items():
        cfg = json.loads((SIMULATOR_ROOT / "configs" / "local_suite" / f"{stem}.json").read_text())
        task = next(iter(cfg.values()))
        assert len(task["containers"]["container_list"]) == expect["containers"], stem
        assert task["item_stream"]["look_ahead"] == expect["look_ahead"], stem
        assert task["agent"]["optimize"] == expect["optimize"], stem
        if not expect["optimize"]:
            assert "optimize" not in task["agent"]["allowed_methods"], stem

    look_aheads = {json.loads((SIMULATOR_ROOT / "configs" / "local_suite" / f"{s}.json").read_text())
                   ["000"]["item_stream"]["look_ahead"] for s in ("c01", "c02", "c03", "c04", "c05")}
    assert {1, 5, 20} <= look_aheads
    container_counts = {len(json.loads((SIMULATOR_ROOT / "configs" / "local_suite" / f"{s}.json").read_text())
                            ["000"]["containers"]["container_list"]) for s in ("c01", "c02", "c03", "c04", "c05")}
    assert {1, 2} <= container_counts


def test_c05_003_run_local_single_run_completes(tmp_path):
    """①run_local単走：exit0・JSON parse可・全task success・format_error/exec_exceptionなし。

    DoD必須（PyBulletが利用可能なこの環境で実際にpassさせる。skipはDoD達成とみなさない）。
    """
    pytest.importorskip("pybullet")
    result_dir = tmp_path / "c05-run-local"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_local",
            "--config-path", "configs/local_suite/c05.json",
            "--module-path", AGENT_MODULE_PATH,
            "--result-dir", str(result_dir),
        ],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    result_json_path = result_dir / "evaluation_results.json"
    assert result_json_path.exists()
    results = json.loads(result_json_path.read_text())
    assert len(results) >= 1
    for task_id, task_result in results.items():
        assert task_result["status"] == "success", (task_id, task_result)


def test_c05_004_official_build_direct_verification():
    """②公式環境構築を直接実行し、settle後の packed_items数1・index一致・inside判定成功を確認する。

    DoD必須（PyBulletが利用可能なこの環境で実際にpassさせる。skipはDoD達成とみなさない）。
    `evaluator.py`/`validator.check_inclusion` はこの config の `inclusion_margin=-0.005` に対し
    floor面ちょうど接地（zero-slack）placementを恒常的に非包含扱いする既知の縁ケースを持つため
    （T-024 heuristicエージェントの本番配置でも同型の "not included" が再現することを事前調査で
    実測確認済み。validator NG自体は既存DoDで不合格としない）、本テストでは production の
    `container_space.contains_oriented_box`（masks.py DIMS段が使う実containment判定、
    margin=0で面接触を許容する契約）を、PyBullet剛体接触解決による実測10^-5m オーダーの
    めり込み（settling tolerance）を許容する `margin=-1e-3` で用いる。
    """
    pytest.importorskip("pybullet")
    import pybullet as p
    from pybullet_utils.bullet_client import BulletClient
    from src.ground_handling.containers import MultiContainerManager
    from src.packing_core.constants import GridParams
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    task = _load_c05_container_cfg()
    containers_cfg = task["containers"]
    expected_index = containers_cfg["container_list"][0]["packed_items"][0]["index"]

    client = BulletClient(connection_mode=p.DIRECT)
    try:
        mgr = MultiContainerManager(client, containers_cfg)
        mgr.build()

        container = mgr.containers[0]
        assert len(container.packed_items) == 1
        item = container.packed_items[0]
        assert item.index == expected_index
        assert item.pos is not None and item.orn is not None

        cdict = {
            "index": container.index, "length": container.length, "width": container.width,
            "height": container.height, "thickness": container.thickness,
            "cut_x": container.cut_x, "cut_y": container.cut_y, "center": container.center,
            "n_vecs": container.n_vecs, "points": container.points,
            "shelf": container.require_shelf,
        }
        space = build_container_space(cdict, index=container.index, cell=GridParams().cell)
        offset_x = container.center[0]
        center_rel = (item.pos[0] - offset_x, item.pos[1], item.pos[2])
        osize = (item.length, item.width, item.height)
        inside = contains_oriented_box(space, center_rel=center_rel, osize=osize, margin=-1e-3)
        assert inside is True
    finally:
        client.disconnect()
