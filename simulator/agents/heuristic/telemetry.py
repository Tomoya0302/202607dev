"""T-028: heuristic Agent telemetry JSONL 出力（詳細仕様書 v1.22 §4.13）。

標準ライブラリのみを使用する。公開API（固定契約テストが要求する最小形状）：

* `NullWriter` / `JsonlTelemetryWriter`: `write_row(row: dict) -> None` を持つwriter。
* `make_writer_from_env(env)`: `TELEMETRY_DIR`/`TELEMETRY_RUN_ID` の有無から
  実writer／NullWriterを選ぶファクトリ。
* `format_row(...)`: 段階時間・局所telemetry・`p_ng_chosen` から16キーちょうどの
  row dictを組み立てる純粋関数。

モジュールグローバルな可変状態は持たない。`print` は使わず `logging` のみを使う。
"""
from __future__ import annotations

import json
import logging
import math
import os

logger = logging.getLogger("packing")

_REJECT_REASONS = ("dims", "inclusion", "overlap", "ceiling", "path")


class NullWriter:
    """telemetry出力を無効化するno-op writer。"""

    def write_row(self, row: dict) -> None:
        """常に何もしない（cwd相対の代替ファイルも作らない）。"""
        return None


class JsonlTelemetryWriter:
    """`<dir_path>/<run_id>.jsonl` へ1行ずつappend書込みするwriter。"""

    def __init__(self, dir_path: str, run_id: str) -> None:
        """パス材料のみ保持する（openはwrite_rowごとに行う）。

        Args:
            dir_path: 出力先ディレクトリ（未存在なら書込み時に作成）。
            run_id: 出力ファイル名の拠り所（`<run_id>.jsonl`）。
        """
        self._dir_path = dir_path
        self._run_id = run_id

    def write_row(self, row: dict) -> None:
        """1行分のJSONをappend・flushする。

        open／直列化／write／flushのいずれの失敗も内部で捕捉しログ警告のみとし、
        呼び出し元（Agent policy）へ例外を漏らさない。
        """
        try:
            serialized = json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            logger.warning("telemetry: failed to serialize row: %s", exc)
            return

        try:
            os.makedirs(self._dir_path, exist_ok=True)
            path = os.path.join(self._dir_path, f"{self._run_id}.jsonl")
            with open(path, "a") as f:
                f.write(serialized + "\n")
                f.flush()
        except OSError as exc:
            logger.warning("telemetry: failed to write row: %s", exc)


def make_writer_from_env(env: dict | None = None) -> "NullWriter | JsonlTelemetryWriter":
    """環境変数から実writer／NullWriterを選ぶ。

    `TELEMETRY_DIR`と`TELEMETRY_RUN_ID`が両方揃った場合のみ`JsonlTelemetryWriter`を返す。
    片方のみ・両方欠如は`NullWriter`（片方のみの場合は警告ログを1回出す）。

    Args:
        env: 環境変数辞書（既定は`os.environ`）。

    Returns:
        `JsonlTelemetryWriter`（両方揃った場合）または`NullWriter`。
    """
    resolved_env = os.environ if env is None else env
    dir_path = resolved_env.get("TELEMETRY_DIR")
    run_id = resolved_env.get("TELEMETRY_RUN_ID")

    if dir_path and run_id:
        return JsonlTelemetryWriter(dir_path, run_id)
    if dir_path or run_id:
        logger.warning(
            "telemetry: TELEMETRY_DIR/TELEMETRY_RUN_ID partially set "
            "(dir=%r, run_id=%r); disabling telemetry output",
            dir_path, run_id,
        )
    return NullWriter()


def _format_reject_top3(reject_reason_counts: dict) -> list[dict]:
    """count>0のみ、count降順・同数reason昇順で最大3件を返す。"""
    entries = [
        {"reason": reason, "count": int(count)}
        for reason, count in (reject_reason_counts or {}).items()
        if int(count) > 0
    ]
    entries.sort(key=lambda entry: (-entry["count"], entry["reason"]))
    return entries[:3]


def _format_layer_error(layer_error: list) -> list[dict]:
    """発生順を維持し、`{"layer": int, "error_type": str}` の2キーちょうどへ整形する。"""
    return [
        {"layer": int(entry["layer"]), "error_type": str(entry["error_type"])}
        for entry in (layer_error or [])
    ]


def _format_timing_value(timing: dict, key: str) -> float:
    """finiteかつ0以上ならfloatを返し、それ以外（未到達・異常値）は`0.0`とする。"""
    value = timing.get(key, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _format_count_value(local_telemetry: dict, key: str) -> int:
    """0以上のbuilt-in intへ正規化する（負値・異常値は0）。"""
    value = local_telemetry.get(key, 0)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if value >= 0 else 0


def _format_p_ng_chosen(p_ng_chosen) -> float | None:
    """選択済み・有限数値のみfloatとして残し、それ以外（未選択・欠落・非有限）はNoneとする。"""
    if p_ng_chosen is None or isinstance(p_ng_chosen, bool):
        return None
    if not isinstance(p_ng_chosen, (int, float)):
        return None
    value = float(p_ng_chosen)
    if not math.isfinite(value):
        return None
    return value


def format_row(
    *, step: int, first_step: bool, timing: dict, local_telemetry: dict,
    p_ng_chosen: float | None,
) -> dict:
    """段階時間・局所telemetry・`p_ng_chosen`から16キーちょうどのrow dictを組み立てる。

    純粋関数（副作用なし）。`regime`は T-028 では常に`None`。

    Args:
        step: 現在のpolicy呼出し通番（0始まり）。
        first_step: `step == 0` を表す値（このAPIは`step`から正規化した値を出力する）。
        timing: `t_state`/`t_enum`/`t_mask`/`t_lpath`/`t_total` を含む段階時間辞書。
        local_telemetry: `n_cand0`/`n_after_dims`/`n_after_geo`/`n_lpath_pass`/
            `reject_reason_counts`/`decided_layer`/`layer_error` を含む局所telemetry辞書。
        p_ng_chosen: 選択済みCandidateの`provisional_p_ng`（未選択・非有限なら`None`可）。

    Returns:
        16キーちょうどのrow dict。
    """
    normalized_step = int(step)
    del first_step  # 出力のfirst_stepは常にnormalized_stepから導出する（step-first_step整合契約）。

    decided_layer = _format_count_value(local_telemetry, "decided_layer")

    return {
        "step": normalized_step,
        "first_step": normalized_step == 0,
        "t_total": _format_timing_value(timing, "t_total"),
        "t_state": _format_timing_value(timing, "t_state"),
        "t_enum": _format_timing_value(timing, "t_enum"),
        "t_mask": _format_timing_value(timing, "t_mask"),
        "t_lpath": _format_timing_value(timing, "t_lpath"),
        "n_cand0": _format_count_value(local_telemetry, "n_cand0"),
        "n_after_dims": _format_count_value(local_telemetry, "n_after_dims"),
        "n_after_geo": _format_count_value(local_telemetry, "n_after_geo"),
        "n_lpath_pass": _format_count_value(local_telemetry, "n_lpath_pass"),
        "decided_layer": decided_layer,
        "regime": None,
        "p_ng_chosen": _format_p_ng_chosen(p_ng_chosen),
        "reject_top3": _format_reject_top3(local_telemetry.get("reject_reason_counts")),
        "layer_error": _format_layer_error(local_telemetry.get("layer_error")),
    }
