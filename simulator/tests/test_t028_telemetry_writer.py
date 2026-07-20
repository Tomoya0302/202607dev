"""T-028: telemetry writer・row整形の契約テスト（詳細仕様書 v1.22 §4.13「T-028 出力契約確定」）。

対象は未実装の `agents/heuristic/telemetry.py`。本ファイルは以下3グループを扱う：

* KEY-*（row整形単体）：`format_row(...)` が返す16キーちょうどのdictの型・値域・schemaを検証する。
* WR-*（writer単体）：環境変数の有無に応じた `make_writer_from_env(...)` の分岐（実writer／
  NullWriter）と、append・flush・`allow_nan=False` の直列化契約を検証する。
* WR-FAIL-*（writer障害系）：open/write/flush失敗時に例外がwriter内部で捕捉され、呼び出し元へ
  漏れないことを検証する。

**本テストが要求する最小公開API（テスト先行のため本セッションで提案する契約。
実装セッションはこの形状を満たせばよく、内部実装・追加のprivate helperは自由）**：

```python
# agents/heuristic/telemetry.py
class NullWriter:
    def write_row(self, row: dict) -> None: ...  # 常にno-op

class JsonlTelemetryWriter:
    def __init__(self, dir_path: str, run_id: str) -> None: ...
    def write_row(self, row: dict) -> None: ...
        # ファイル "<dir_path>/<run_id>.jsonl" へ json.dumps(row, allow_nan=False) を1行
        # append・書込み後flush。open/write/flush/直列化失敗はここで捕捉しログ警告のみ
        # （例外を外へ送出しない）。

def make_writer_from_env(env: dict | None = None) -> "NullWriter | JsonlTelemetryWriter":
    ...  # env（既定 os.environ）から TELEMETRY_DIR・TELEMETRY_RUN_ID を読む。
        # 両方揃った場合のみ JsonlTelemetryWriter、それ以外（片方のみ・両方欠如）は NullWriter。

def format_row(
    *, step: int, first_step: bool, timing: dict, local_telemetry: dict,
    p_ng_chosen: float | None,
) -> dict:
    ...  # timing（t_state/t_enum/t_mask/t_lpath/t_total）とlocal_telemetry
        # （n_cand0/n_after_dims/n_after_geo/n_lpath_pass/reject_reason_counts/
        # decided_layer/layer_error）とp_ng_chosenから16キーちょうどのrowを組み立てる
        # 純粋関数。regimeは常にNone。reject_top3はreject_reason_countsから
        # count降順・同数reason昇順・count>0のみ・最大3件で導出する。
```

source文字列検査・内部private名の固定は避け、上記4公開シンボルの入出力（公開動作・schema）のみ
を検証する。
"""
from __future__ import annotations

import json
import math

import pytest

from tests.fixtures.t028.scenarios import sample_local_telemetry, sample_timing

_REQUIRED_KEYS = frozenset(
    {
        "step", "first_step", "t_total", "t_state", "t_enum", "t_mask", "t_lpath",
        "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer",
        "regime", "p_ng_chosen", "reject_top3", "layer_error",
    }
)


def _telemetry_module():
    import agents.heuristic.telemetry as telemetry_module

    return telemetry_module


def _format_row(step=0, first_step=True, timing=None, local_telemetry=None, p_ng_chosen=None):
    telemetry_module = _telemetry_module()
    return telemetry_module.format_row(
        step=step,
        first_step=first_step,
        timing=timing if timing is not None else sample_timing(),
        local_telemetry=local_telemetry if local_telemetry is not None else sample_local_telemetry(),
        p_ng_chosen=p_ng_chosen,
    )


# =============================================================================
# KEY: row整形単体（16キーちょうど・各キーの型/値域）
# =============================================================================


def test_key_000_exactly_16_keys_no_more_no_less():
    row = _format_row(p_ng_chosen=0.2)
    assert set(row.keys()) == _REQUIRED_KEYS
    assert len(row) == 16


def test_key_step_is_builtin_int():
    row = _format_row(step=3, first_step=False, p_ng_chosen=0.2)
    assert row["step"] == 3
    assert type(row["step"]) is int


def test_key_first_step_matches_step_zero_contract():
    row0 = _format_row(step=0, first_step=True, p_ng_chosen=0.2)
    row1 = _format_row(step=1, first_step=False, p_ng_chosen=0.2)
    assert row0["first_step"] is True
    assert row1["first_step"] is False
    assert type(row0["first_step"]) is bool


@pytest.mark.parametrize("key", ["t_total", "t_state", "t_enum", "t_mask", "t_lpath"])
def test_key_timing_keys_are_finite_nonnegative_float(key):
    row = _format_row(p_ng_chosen=0.2)
    assert isinstance(row[key], float)
    assert math.isfinite(row[key])
    assert row[key] >= 0.0


def test_key_timing_unreached_stage_stays_zero():
    timing = {"t_state": 0.01, "t_enum": 0.0, "t_mask": 0.0, "t_lpath": 0.0, "t_total": 0.01}
    row = _format_row(timing=timing, p_ng_chosen=None)
    assert row["t_enum"] == 0.0
    assert row["t_mask"] == 0.0
    assert row["t_lpath"] == 0.0


@pytest.mark.parametrize(
    "key", ["n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer"]
)
def test_key_count_keys_are_builtin_int_ge_zero(key):
    row = _format_row(p_ng_chosen=0.2)
    assert type(row[key]) is int
    assert row[key] >= 0


def test_key_decided_layer_range_0_to_4():
    for decided_layer in (0, 1, 2, 3, 4):
        local = dict(sample_local_telemetry())
        local["decided_layer"] = decided_layer
        row = _format_row(local_telemetry=local, p_ng_chosen=None)
        assert row["decided_layer"] == decided_layer


def test_key_regime_generated_row_is_always_null():
    """T-028時点の生成行はregimeを常にnullとする（§4.13 D4）。"""
    row = _format_row(p_ng_chosen=0.2)
    assert row["regime"] is None


@pytest.mark.parametrize("bad_step", [True, False])
def test_key_strict_int_step_rejects_bool(bad_step):
    """boolはintの派生型だが、stepへboolを渡した場合は組込みintへ厳密変換される
    （実装が`type(v) is int`を満たすことをrow生成側でも要求する）。"""
    row = _format_row(step=bad_step, first_step=False, p_ng_chosen=None)
    assert type(row["step"]) is int
    assert type(row["step"]) is not bool


def test_key_strict_bool_first_step_is_builtin_bool_not_int():
    row = _format_row(step=0, first_step=True, p_ng_chosen=None)
    assert type(row["first_step"]) is bool


def test_key_png_chosen_finite_when_selected():
    row = _format_row(p_ng_chosen=0.42)
    assert row["p_ng_chosen"] == pytest.approx(0.42)
    assert math.isfinite(row["p_ng_chosen"])


@pytest.mark.parametrize("bad_value", [None, math.nan, math.inf, -math.inf])
def test_key_png_chosen_null_when_unselected_or_nonfinite(bad_value):
    row = _format_row(p_ng_chosen=bad_value)
    assert row["p_ng_chosen"] is None


def test_key_reject_top3_sorted_count_desc_reason_asc_tie():
    local = dict(sample_local_telemetry())
    local["reject_reason_counts"] = {"ceiling": 2, "dims": 2, "overlap": 5, "inclusion": 0, "path": 0}
    row = _format_row(local_telemetry=local, p_ng_chosen=None)
    assert row["reject_top3"] == [
        {"reason": "overlap", "count": 5},
        {"reason": "ceiling", "count": 2},
        {"reason": "dims", "count": 2},
    ]


def test_key_reject_top3_excludes_zero_and_caps_at_three():
    local = dict(sample_local_telemetry())
    local["reject_reason_counts"] = {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0, "path": 0}
    row = _format_row(local_telemetry=local, p_ng_chosen=None)
    assert row["reject_top3"] == []


def test_key_reject_top3_short_list_not_padded_with_null():
    local = dict(sample_local_telemetry())
    local["reject_reason_counts"] = {"dims": 1}
    row = _format_row(local_telemetry=local, p_ng_chosen=None)
    assert row["reject_top3"] == [{"reason": "dims", "count": 1}]


def test_key_layer_error_empty_when_no_exceptions():
    local = dict(sample_local_telemetry())
    local["layer_error"] = []
    row = _format_row(local_telemetry=local, p_ng_chosen=None)
    assert row["layer_error"] == []


def test_key_layer_error_preserves_occurrence_order_and_two_keys_only():
    local = dict(sample_local_telemetry())
    local["layer_error"] = [
        {"layer": 1, "error_type": "ValueError"},
        {"layer": 2, "error_type": "RuntimeError"},
    ]
    row = _format_row(local_telemetry=local, p_ng_chosen=None)
    assert row["layer_error"] == [
        {"layer": 1, "error_type": "ValueError"},
        {"layer": 2, "error_type": "RuntimeError"},
    ]
    for entry in row["layer_error"]:
        assert set(entry.keys()) == {"layer", "error_type"}
        assert type(entry["layer"]) is int


def test_key_row_is_json_serializable_with_allow_nan_false():
    row = _format_row(p_ng_chosen=0.2)
    # allow_nan=False で例外にならない（NaN/Infinityを含まない）ことの直接確認。
    json.dumps(row, allow_nan=False)


# =============================================================================
# WR: writer単体（環境変数分岐・append・flush・allow_nan）
# =============================================================================


def _make_writer(monkeypatch, tmp_path, telemetry_dir=None, run_id=None):
    telemetry_module = _telemetry_module()
    env = {}
    if telemetry_dir is not None:
        env["TELEMETRY_DIR"] = str(telemetry_dir)
    if run_id is not None:
        env["TELEMETRY_RUN_ID"] = run_id
    return telemetry_module.make_writer_from_env(env)


def test_wr_act_both_env_present_returns_real_writer_and_creates_file(tmp_path):
    telemetry_module = _telemetry_module()
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-both")
    assert not isinstance(writer, telemetry_module.NullWriter)
    writer.write_row(sample_local_telemetry() | {"note": "unused-in-real-row"})
    out = tmp_path / "telemetry" / "run-both.jsonl"
    assert out.exists()


def test_wr_act_none_both_missing_returns_nullwriter_and_writes_no_file(tmp_path, monkeypatch):
    telemetry_module = _telemetry_module()
    monkeypatch.chdir(tmp_path)
    writer = telemetry_module.make_writer_from_env({})
    assert isinstance(writer, telemetry_module.NullWriter)
    writer.write_row({"step": 0})
    # cwd相対の代替ファイルを作らない（§4.13 D1）。
    assert list(tmp_path.rglob("*.jsonl")) == []


@pytest.mark.parametrize(
    "env",
    [
        {"TELEMETRY_DIR": "only-dir-set"},
        {"TELEMETRY_RUN_ID": "only-run-id-set"},
    ],
    ids=["dir-only", "run-id-only"],
)
def test_wr_act_one_partial_env_returns_nullwriter(tmp_path, monkeypatch, env):
    """片方のみ設定はNullWriter（D-Q2確定：両方揃って初めて有効化）。"""
    telemetry_module = _telemetry_module()
    resolved_env = dict(env)
    if "TELEMETRY_DIR" in resolved_env:
        resolved_env["TELEMETRY_DIR"] = str(tmp_path / "telemetry")
    monkeypatch.chdir(tmp_path)
    writer = telemetry_module.make_writer_from_env(resolved_env)
    assert isinstance(writer, telemetry_module.NullWriter)
    writer.write_row({"step": 0})
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_wr_append_multiple_rows_accumulate_without_overwrite(tmp_path):
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-append")
    writer.write_row({"step": 0})
    writer.write_row({"step": 1})
    writer.write_row({"step": 2})
    out = tmp_path / "telemetry" / "run-append.jsonl"
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["step"] for l in lines] == [0, 1, 2]


def test_wr_append_reopen_same_run_id_continues_appending(tmp_path):
    telemetry_module = _telemetry_module()
    telemetry_dir = tmp_path / "telemetry"
    w1 = telemetry_module.make_writer_from_env(
        {"TELEMETRY_DIR": str(telemetry_dir), "TELEMETRY_RUN_ID": "run-reopen"}
    )
    w1.write_row({"step": 0})
    w2 = telemetry_module.make_writer_from_env(
        {"TELEMETRY_DIR": str(telemetry_dir), "TELEMETRY_RUN_ID": "run-reopen"}
    )
    w2.write_row({"step": 1})
    out = telemetry_dir / "run-reopen.jsonl"
    lines = out.read_text().splitlines()
    assert len(lines) == 2


def test_wr_flush_row_is_visible_immediately_after_write(tmp_path):
    """closeを明示せずとも、write_row直後にファイル内容から読み取れる（各行flush、§4.13）。"""
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-flush")
    writer.write_row({"step": 0})
    out = tmp_path / "telemetry" / "run-flush.jsonl"
    # writerをcloseしない状態で直接読む。flushされていなければ空/不完全になり得る。
    content = out.read_text()
    assert json.loads(content.strip().splitlines()[-1])["step"] == 0


@pytest.mark.parametrize(
    "bad_row",
    [
        {"t_total": math.nan},
        {"t_total": math.inf},
        {"t_total": -math.inf},
    ],
    ids=["nan", "inf", "neg-inf"],
)
def test_wr_nan_row_is_not_written_allow_nan_false(tmp_path, bad_row):
    """allow_nan=Falseにより、NaN/Infinityを含む行はJSON直列化に失敗し、その行は欠落する
    （writer内で捕捉、§4.13「1行の書込みタイミング・I/O契約」）。"""
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-nan")
    writer.write_row(bad_row)  # 例外を外へ漏らさない
    out = tmp_path / "telemetry" / "run-nan.jsonl"
    if out.exists():
        assert out.read_text().strip() == ""


# =============================================================================
# WR-FAIL: writer障害系（open/write/flush失敗はwriter内部で捕捉）
# =============================================================================


def test_wr_fail_open_does_not_raise(tmp_path, monkeypatch):
    """出力先ディレクトリ作成不可等でopenが失敗しても、write_rowは例外を送出しない。"""
    telemetry_module = _telemetry_module()
    # 既存ファイルをディレクトリの位置に置き、ディレクトリ作成/オープンを失敗させる。
    blocker = tmp_path / "telemetry"
    blocker.write_text("not a directory")
    writer = telemetry_module.make_writer_from_env(
        {"TELEMETRY_DIR": str(blocker), "TELEMETRY_RUN_ID": "run-fail-open"}
    )
    writer.write_row({"step": 0})  # 例外を送出しないことそのものが主張


def test_wr_fail_write_does_not_raise(tmp_path, monkeypatch):
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-fail-write")

    def _raising_write(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr("builtins.open", _raising_open_after_first_call(_raising_write))
    writer.write_row({"step": 0})  # 例外を送出しないことそのものが主張


def _raising_open_after_first_call(write_replacement):
    """`open` をラップし、返すファイルオブジェクトの `write` を例外化する。"""
    import builtins

    real_open = builtins.open

    def _wrapped(*args, **kwargs):
        f = real_open(*args, **kwargs)
        f.write = write_replacement
        return f

    return _wrapped


def test_wr_fail_flush_does_not_raise(tmp_path, monkeypatch):
    writer = _make_writer(None, tmp_path, telemetry_dir=tmp_path / "telemetry", run_id="run-fail-flush")

    def _raising_flush(*args, **kwargs):
        raise OSError("injected flush failure")

    import builtins

    real_open = builtins.open

    def _wrapped(*args, **kwargs):
        f = real_open(*args, **kwargs)
        f.flush = _raising_flush
        return f

    monkeypatch.setattr("builtins.open", _wrapped)
    writer.write_row({"step": 0})  # 例外を送出しないことそのものが主張
