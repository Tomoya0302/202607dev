"""T-028: telemetry JSONLのキー・型・schema検査CLI（詳細仕様書 v1.22 §4.13）。

exit code契約（D-Q1確定）：

* 0: 対象JSONL全行が16キーちょうど・型・`regime`集合・NaN/Infinity不在を満たす（正常）。
* 1: 非JSON行（空行・空白のみの行を含む）・16キー過不足・型違反・`regime`集合外・
  NaN/Infinity混入等、出力内容の異常。
* 2: 引数欠如・過剰、対象ファイル不存在、ディレクトリ指定、読取不能（権限等）等、
  CLI・入力手段の異常。

`main(argv: list[str] | None = None) -> int` を公開する。CLI引数不正等はここで
整数exit codeへ正規化し、`SystemExit`を`main()`の外へ漏らさない。`__main__`入口のみ
`raise SystemExit(main())` とする。
"""
from __future__ import annotations

import json
import math
import os
import sys

_REQUIRED_KEYS = frozenset(
    {
        "step", "first_step", "t_total", "t_state", "t_enum", "t_mask", "t_lpath",
        "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer",
        "regime", "p_ng_chosen", "reject_top3", "layer_error",
    }
)
_INT_KEYS = ("step", "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass", "decided_layer")
_FLOAT_KEYS = ("t_total", "t_state", "t_enum", "t_mask", "t_lpath")
_REJECT_REASONS = frozenset({"dims", "inclusion", "overlap", "ceiling", "path"})
_REGIME_VALUES = (None, "conservative", "aggressive")


class _RowInvalid(Exception):
    """1行のschema・型・値域違反（内容異常 → exit 1）を表す内部例外。"""


class _NonFiniteConstant(Exception):
    """`json.loads`がNaN/Infinity/-Infinityへ到達したことを示す内部例外。

    黙って非有限floatを受理しないよう、`parse_constant`から送出する。
    """


def _reject_constant(token: str) -> float:
    raise _NonFiniteConstant(token)


def _is_strict_int(value) -> bool:
    return type(value) is int


def _is_strict_bool(value) -> bool:
    return type(value) is bool


def _is_finite_nonneg_float(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def _validate_reject_top3(reject_top3) -> None:
    if not isinstance(reject_top3, list) or len(reject_top3) > 3:
        raise _RowInvalid("reject_top3: not a list or too many entries")
    prev = None
    for entry in reject_top3:
        if not isinstance(entry, dict) or set(entry.keys()) != {"reason", "count"}:
            raise _RowInvalid("reject_top3: entry schema mismatch")
        reason = entry["reason"]
        count = entry["count"]
        if reason not in _REJECT_REASONS:
            raise _RowInvalid("reject_top3: reason outside allowed set")
        if not _is_strict_int(count) or count <= 0:
            raise _RowInvalid("reject_top3: count must be positive builtin int")
        if prev is not None:
            prev_count, prev_reason = prev
            if count > prev_count or (count == prev_count and reason < prev_reason):
                raise _RowInvalid("reject_top3: not sorted by count desc / reason asc")
        prev = (count, reason)


def _validate_layer_error(layer_error) -> None:
    if not isinstance(layer_error, list):
        raise _RowInvalid("layer_error: not a list")
    for entry in layer_error:
        if not isinstance(entry, dict) or set(entry.keys()) != {"layer", "error_type"}:
            raise _RowInvalid("layer_error: entry schema mismatch")
        layer = entry["layer"]
        error_type = entry["error_type"]
        if not _is_strict_int(layer) or not (1 <= layer <= 4):
            raise _RowInvalid("layer_error: layer out of range 1..4")
        if not isinstance(error_type, str) or error_type == "":
            raise _RowInvalid("layer_error: error_type must be non-empty str")


def _validate_row(row) -> None:
    """1行のschema・型・値域を検査する。違反があれば`_RowInvalid`を送出する。"""
    if not isinstance(row, dict) or set(row.keys()) != _REQUIRED_KEYS:
        raise _RowInvalid("row does not have exactly the 16 required keys")

    for key in _INT_KEYS:
        if not _is_strict_int(row[key]) or row[key] < 0:
            raise _RowInvalid(f"{key}: must be builtin int >= 0")

    if not _is_strict_bool(row["first_step"]):
        raise _RowInvalid("first_step: must be builtin bool")
    if row["first_step"] != (row["step"] == 0):
        raise _RowInvalid("first_step/step mismatch")

    for key in _FLOAT_KEYS:
        if not _is_finite_nonneg_float(row[key]):
            raise _RowInvalid(f"{key}: must be finite float >= 0")

    if not (0 <= row["decided_layer"] <= 4):
        raise _RowInvalid("decided_layer: out of range 0..4")

    p_ng_chosen = row["p_ng_chosen"]
    if p_ng_chosen is not None:
        if isinstance(p_ng_chosen, bool) or not isinstance(p_ng_chosen, (int, float)):
            raise _RowInvalid("p_ng_chosen: must be None or finite number")
        if not math.isfinite(float(p_ng_chosen)):
            raise _RowInvalid("p_ng_chosen: must be finite")

    regime = row["regime"]
    if not (regime is None or regime in _REGIME_VALUES[1:]):
        raise _RowInvalid("regime: outside allowed set")

    _validate_reject_top3(row["reject_top3"])
    _validate_layer_error(row["layer_error"])


def _run_check(path: str) -> int:
    """指定パスのJSONLを検査し、exit code（0/1/2）を返す。"""
    try:
        with open(path) as f:
            lines = f.readlines()
    except (OSError, ValueError):
        return 2

    for line in lines:
        if line.strip() == "":
            # 空行・空白のみの行は有効JSONではなく、CHK-BADLINEと同じ内容異常。
            return 1
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except (json.JSONDecodeError, _NonFiniteConstant):
            return 1
        try:
            _validate_row(row)
        except _RowInvalid:
            return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLIエントリポイント本体。整数exit codeを返す（`SystemExit`は送出しない）。

    Args:
        argv: CLI引数（省略時は`sys.argv[1:]`）。ちょうど1個のpositional（対象パス）を要求する。

    Returns:
        int: 0（正常）/ 1（内容異常）/ 2（CLI・入力手段の異常）。
    """
    resolved_argv = sys.argv[1:] if argv is None else argv

    if len(resolved_argv) != 1:
        return 2

    path = resolved_argv[0]

    if not os.path.exists(path):
        return 2
    if os.path.isdir(path):
        return 2

    return _run_check(path)


if __name__ == "__main__":
    raise SystemExit(main())
