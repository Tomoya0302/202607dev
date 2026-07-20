"""T-028: `scripts/check_telemetry.py` CLIの契約テスト（詳細仕様書 v1.22 §4.13「キー検査スクリプト」）。

exit code契約（D-Q1確定、本セッションでのユーザー確定事項）：

* exit 0 ＝ 対象JSONL全行が16キーちょうど・型・`regime`集合・NaN/Infinity不在を満たす（正常）。
* exit 1 ＝ 非JSON行・16キー過不足・型違反・`regime`集合外・NaN/Infinity混入等、**出力内容の異常**。
* exit 2 ＝ 引数欠如・対象ファイル不存在・読取不能（権限等）等、**CLI・入力手段の異常**。

`regime`はT-028の生成行では常に`null`だが（§4.13 D4）、`check_telemetry.py`自体は将来
（T-036以降）`"conservative"`/`"aggressive"`を出力するAgentとの互換性のため、
`null | "conservative" | "aggressive"` の3値を正常として受理する（本節のCHK-REGIME）。

**本テストが要求する最小公開API**：`scripts/check_telemetry.py` が `main(argv: list[str] | None) -> int`
を公開する（`scripts/run_suite.py` の `main(argv)` 呼出し規約に揃える）。CLI引数不正は
argparseの`SystemExit`でもよく、`_run_check()`ヘルパがその両方を吸収して整数exit codeへ正規化する。
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

from tests.fixtures.t028.scenarios import valid_row, write_jsonl, write_valid_jsonl


def _import_check_telemetry():
    return importlib.import_module("scripts.check_telemetry")


def _run_check(argv: list[str]) -> int:
    """`main(argv)` を呼び、戻り値かSystemExitのいずれでも整数exit codeへ正規化する。"""
    check_telemetry = _import_check_telemetry()
    try:
        result = check_telemetry.main(argv)
    except SystemExit as exc:
        return int(exc.code)
    return int(result)


# =============================================================================
# CHK: 正常系
# =============================================================================


def test_chk_ok_all_valid_rows_exit_0(tmp_path):
    path = write_valid_jsonl(tmp_path / "run.jsonl", n_rows=3)
    assert _run_check([str(path)]) == 0


def test_chk_ok_empty_file_exit_0(tmp_path):
    """0行のJSONLは検査対象が無いだけであり、schema違反ではない。"""
    path = write_jsonl(tmp_path / "empty.jsonl", [])
    assert _run_check([str(path)]) == 0


# =============================================================================
# CHK: schema違反 → exit 1
# =============================================================================


def test_chk_missing_key_exit_1(tmp_path):
    row = valid_row()
    del row["layer_error"]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_extra_key_exit_1(tmp_path):
    row = valid_row()
    row["extra_unexpected_key"] = "x"
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_type_step_as_string_exit_1(tmp_path):
    row = valid_row()
    row["step"] = "0"
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_type_t_total_as_string_exit_1(tmp_path):
    row = valid_row()
    row["t_total"] = "0.1"
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


@pytest.mark.parametrize(
    "int_key", ["step", "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer"]
)
def test_chk_type_bool_as_int_exit_1(tmp_path, int_key):
    """int項目へTrue/Falseを与えたケース。boolはintの派生型のため、組込みintであることを
    厳密に検証する契約（isinstanceではなくtype(v) is int相当）。"""
    row = valid_row()
    row[int_key] = True
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_type_int_as_bool_first_step_exit_1(tmp_path):
    """first_stepへ0/1（int）を与えたケースはexit 1（組込みboolのみ許容）。"""
    row = valid_row()
    row["first_step"] = 1
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


@pytest.mark.parametrize("bad_regime", ["unknown", "Conservative", 1, 1.0])
def test_chk_regime_outside_allowed_set_exit_1(tmp_path, bad_regime):
    row = valid_row()
    row["regime"] = bad_regime
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


@pytest.mark.parametrize("good_regime", [None, "conservative", "aggressive"])
def test_chk_regime_null_conservative_aggressive_all_exit_0(tmp_path, good_regime):
    """将来互換schema検査：T-028生成行はnullのみだが、check_telemetryは3値を正常受理する
    （CHK-REGIME、ユーザー確定事項）。"""
    row = valid_row()
    row["regime"] = good_regime
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 0


def test_chk_nan_in_t_total_exit_1(tmp_path):
    """正常writerはallow_nan=FalseでNaN行を出力しないため、破損ファイルを模した手書き注入。"""
    line = valid_row()
    text = json.dumps(line).replace('"t_total": 0.15', '"t_total": NaN')
    path = write_jsonl(tmp_path / "run.jsonl", [text])
    assert _run_check([str(path)]) == 1


def test_chk_infinity_in_t_state_exit_1(tmp_path):
    line = valid_row()
    text = json.dumps(line).replace('"t_state": 0.01', '"t_state": Infinity')
    path = write_jsonl(tmp_path / "run.jsonl", [text])
    assert _run_check([str(path)]) == 1


def test_chk_reject_top3_wrong_order_exit_1(tmp_path):
    row = valid_row()
    row["reject_top3"] = [{"reason": "ceiling", "count": 2}, {"reason": "dims", "count": 4}]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_reject_top3_more_than_three_entries_exit_1(tmp_path):
    row = valid_row()
    row["reject_top3"] = [
        {"reason": "dims", "count": 4},
        {"reason": "ceiling", "count": 3},
        {"reason": "overlap", "count": 2},
        {"reason": "path", "count": 1},
    ]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_reject_top3_zero_count_entry_exit_1(tmp_path):
    row = valid_row()
    row["reject_top3"] = [{"reason": "dims", "count": 0}]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_layer_error_extra_key_exit_1(tmp_path):
    row = valid_row()
    row["layer_error"] = [{"layer": 1, "error_type": "ValueError", "message": "boom"}]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_layer_error_out_of_range_layer_exit_1(tmp_path):
    row = valid_row()
    row["layer_error"] = [{"layer": 5, "error_type": "ValueError"}]
    path = write_jsonl(tmp_path / "run.jsonl", [json.dumps(row)])
    assert _run_check([str(path)]) == 1


def test_chk_bad_line_non_json_exit_1(tmp_path):
    """ファイルは開けるが1行が有効JSONでない（外部killによる切詰め等）。
    内容異常として exit 1（D-Q1確定：入力手段の異常(2)ではなく出力内容の異常(1)）。"""
    valid = json.dumps(valid_row(step=0))
    truncated = '{"step": 1, "first_step": false, "t_total": 0.2'  # 意図的に不完全なJSON
    path = write_jsonl(tmp_path / "run.jsonl", [valid, truncated])
    assert _run_check([str(path)]) == 1


# =============================================================================
# CHK: CLI・入力エラー → exit 2
# =============================================================================


def test_chk_no_argument_exit_2():
    assert _run_check([]) == 2


def test_chk_file_does_not_exist_exit_2(tmp_path):
    assert _run_check([str(tmp_path / "does-not-exist.jsonl")]) == 2


def test_chk_unreadable_permission_denied_exit_2(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ではパーミッションビットが読取を妨げないため検証不可（評価コンテナ等）")
    path = write_valid_jsonl(tmp_path / "run.jsonl", n_rows=1)
    path.chmod(0o000)
    try:
        assert _run_check([str(path)]) == 2
    finally:
        path.chmod(0o644)  # cleanup（pytest tmp_path削除の妨げにならないよう復元）
