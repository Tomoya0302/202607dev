"""E2E回帰・5課題suite集計基盤（T-030B、詳細仕様書 §5.5 v1.19/v1.20）。

`run_local.py`（T-030A、§5.7）が確立した AgentFactory／Env／Runner 呼出し経路を、課題
（`configs/local_suite/c*.json`）ごとに `sys.executable` サブプロセスとして繰り返し起動する
薄いラッパー。`run_local.py` 自体は無変更とし、実行ロジック（`EvaluationApp` の step 実行・
Agent load・集計）を本スクリプトで重複実装しない（再利用機構、§5.5 v1.19）。

公開契約（§5.5 v1.19/v1.20 を正とする。詳細はdocsを参照し、本docstringには要約のみ記載）：

  - CLI: ``python -m scripts.run_suite --config-dir <dir> --module-path <path>
    --result-dir <dir> [--baseline <suite_results.jsonl>] [--timeout-sec <float>]``
  - 決定論順序: ``--config-dir`` 配下の ``c*.json`` をファイル名昇順で発見・実行
  - 3層タクソノミ: validator NG（完走）／format_error（未完走）／exec_exception（未完走）
  - 生成物: ``suite_results.jsonl``（行粒度契約）／``suite_summary.json``（固定10キー）／
    ``suite_report.md``（人間可読、stdout併記）
  - exit code: 0=成功（非退行）／1=baseline比較による退行／2=CLI・入力・schema・出力エラー
    （優先順位 2 > 1 > 0）

本スクリプトが公開する契約は ``run_suite()``／``main()``／CLI／生成物schema／exit code のみ
であり、内部のヘルパー関数名・シグネチャは公開APIではない（契約テスト設計計画で確認済み）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_RESULT_FNAME = "evaluation_results.json"  # run_local.py の既定出力名と一致させる
_TIMEOUT_DEFAULT = 600.0

_ROW_KEYS = (
    "config", "agent", "task_id", "status", "fill_score", "num_placed_items",
    "n_validator_ng", "format_error", "exec_exception", "policy_time",
    "optimization_time", "wall_time",
)
_SUMMARY_KEYS = (
    "agent", "num_discovered", "num_completed", "completion_rate",
    "n_validator_ng_total", "n_format_error_total", "n_exec_exception_total",
    "mean_fill", "suite_score", "compared_config_stems",
)
_REPORT_COLUMNS = (
    "config", "task_id", "status", "n_validator_ng", "num_placed_items",
    "fill_score", "policy_time", "optimization_time", "wall_time",
)

_EXIT_OK = 0
_EXIT_REGRESSION = 1
_EXIT_INPUT_ERROR = 2


class _SuiteError(Exception):
    """CLI・config-dir・baseline schema等の入力エラー（内部実装専用。公開契約はexit codeのみ）。"""


# --- 数値サニタイズ（SAN契約：NaN/±Infはnullへ、書き出しJSONへ一切含めない） -----------------


def _sanitize_number(x) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf):
        return None
    return xf


def _validate_timeout(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise _SuiteError(f"--timeout-sec must be a float: {value!r}") from exc
    if not math.isfinite(v) or v <= 0:
        raise _SuiteError(f"--timeout-sec must be finite and > 0: {value!r}")
    return v


def _normalize_agent(module_path: str) -> str:
    """`--module-path` を正規化する（末尾 `/`（および `\\`）を除去したパス表記）。"""
    return module_path.rstrip("/").rstrip("\\")


def _discover_configs(config_dir: str) -> list[Path]:
    """`config_dir` 配下の `c*.json` をファイル名昇順で発見する。0件・不存在は `_SuiteError`。"""
    cdir = Path(config_dir)
    if not cdir.is_dir():
        raise _SuiteError(f"--config-dir does not exist or is not a directory: {config_dir}")
    found = sorted(cdir.glob("c*.json"), key=lambda p: p.name)
    if not found:
        raise _SuiteError(f"no c*.json configs discovered in {config_dir}")
    return found


# --- task結果分類（3層タクソノミのうち success/format_error 側、exec_exceptionは呼出し側） ----


def _classify_task(task_result: dict) -> tuple[str, int | None]:
    """`(status, n_validator_ng)` を返す。`format_error` は `n_validator_ng=None`（Q3確定）。"""
    status = task_result.get("status")
    if status == "format_error":
        return "format_error", None
    place_states = task_result.get("place_states") or {}
    n_ng = sum(1 for v in place_states.values() if v is False)
    return "success", n_ng


def _rows_from_result_json(config_stem: str, agent: str, result_json: dict, wall_time: float) -> list[dict]:
    rows = []
    for task_id, task_result in result_json.items():
        status, n_ng = _classify_task(task_result)
        evaluation = task_result.get("evaluation") or {}
        time_results = task_result.get("time_results") or {}
        rows.append({
            "config": config_stem,
            "agent": agent,
            "task_id": str(task_id),
            "status": status,
            "fill_score": _sanitize_number(evaluation.get("fill_score")),
            "num_placed_items": _sanitize_number(evaluation.get("num_placed_items")),
            "n_validator_ng": n_ng if status == "success" else None,
            "format_error": status == "format_error",
            "exec_exception": False,
            "policy_time": _sanitize_number(time_results.get("policy")),
            "optimization_time": _sanitize_number(time_results.get("optimization")),
            "wall_time": float(wall_time),
        })
    return rows


def _synthetic_exec_exception_row(config_stem: str, agent: str, wall_time: float) -> dict:
    return {
        "config": config_stem,
        "agent": agent,
        "task_id": None,
        "status": "exec_exception",
        "fill_score": None,
        "num_placed_items": None,
        "n_validator_ng": None,
        "format_error": False,
        "exec_exception": True,
        "policy_time": None,
        "optimization_time": None,
        "wall_time": float(wall_time),
    }


# --- 課題1件の実行（run_local.py をsubprocess再利用、3層タクソノミの境界判定） ------------------


def _run_one_config(
    config_path: Path, module_path: str, agent: str, suite_result_dir: Path, timeout_sec: float
) -> list[dict]:
    """1 config を `run_local.py` subprocess経由で実行し、行群を返す（synthetic含む）。

    nonzero exit・結果JSON欠落・parse不可・`TimeoutExpired` のいずれでも例外を外へ投げず、
    1行の synthetic exec_exception 行を返す（呼び出し側の継続実行はこの関数の外側で担保）。
    """
    stem = config_path.stem
    cfg_result_dir = suite_result_dir / stem
    args = [
        sys.executable, "-m", "scripts.run_local",
        "--config-path", str(config_path),
        "--module-path", module_path,
        "--result-dir", str(cfg_result_dir),
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            args, shell=False, timeout=timeout_sec, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        wall_time = time.perf_counter() - t0
        return [_synthetic_exec_exception_row(stem, agent, wall_time)]
    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        return [_synthetic_exec_exception_row(stem, agent, wall_time)]

    result_json_path = cfg_result_dir / _RESULT_FNAME
    if not result_json_path.exists():
        return [_synthetic_exec_exception_row(stem, agent, wall_time)]

    try:
        with open(result_json_path) as f:
            result_json = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [_synthetic_exec_exception_row(stem, agent, wall_time)]

    if not isinstance(result_json, dict) or len(result_json) == 0:
        return [_synthetic_exec_exception_row(stem, agent, wall_time)]

    return _rows_from_result_json(stem, agent, result_json, wall_time)


# --- 完走定義・集計 ----------------------------------------------------------------------


def _is_config_completed(rows: list[dict]) -> bool:
    """完走(config単位) = task 1件以上 ∧ 全task `status=='success'`（validator NGは完走に含む）。"""
    if not rows:
        return False
    return all(r["status"] == "success" for r in rows)


def _config_score(rows: list[dict], completed: bool) -> float:
    if not completed:
        return 0.0
    finite_fills = [r["fill_score"] for r in rows if r["fill_score"] is not None]
    if not finite_fills:
        return 0.0
    return sum(finite_fills) / len(finite_fills)


def _aggregate_summary(
    agent: str, discovered_stems: list[str], rows_by_stem: dict[str, list[dict]]
) -> dict:
    num_discovered = len(discovered_stems)
    completed_flags = {stem: _is_config_completed(rows_by_stem.get(stem, [])) for stem in discovered_stems}
    num_completed = sum(1 for v in completed_flags.values() if v)
    completion_rate = (num_completed / num_discovered) if num_discovered > 0 else 0.0

    all_rows = [r for stem in discovered_stems for r in rows_by_stem.get(stem, [])]
    n_validator_ng_total = sum(r["n_validator_ng"] for r in all_rows if r["n_validator_ng"] is not None)
    n_format_error_total = sum(1 for r in all_rows if r["status"] == "format_error")
    n_exec_exception_total = sum(1 for r in all_rows if r["status"] == "exec_exception")

    scores = [
        _config_score(rows_by_stem.get(stem, []), completed_flags[stem]) for stem in discovered_stems
    ]
    mean_fill = (sum(scores) / len(scores)) if scores else 0.0

    return {
        "agent": agent,
        "num_discovered": num_discovered,
        "num_completed": num_completed,
        "completion_rate": completion_rate,
        "n_validator_ng_total": n_validator_ng_total,
        "n_format_error_total": n_format_error_total,
        "n_exec_exception_total": n_exec_exception_total,
        "mean_fill": mean_fill,
        "suite_score": mean_fill,  # SCORE-004/SUM-005: mean_fill と suite_score は同一値
        "compared_config_stems": sorted(set(discovered_stems)),
    }


# --- baseline 比較（agent一致・unique stem集合一致・schema検査・退行判定） ------------------


def _validate_baseline_schema(rows: list[dict]) -> None:
    """schema不正を検出する（v1.20訂正：stem重複行は正常。以下(i)(ii)(iii)のみエラー）。"""
    seen_task_rows: set[tuple[str, str]] = set()
    synthetic_count_by_config: dict[str, int] = {}
    has_normal_task_by_config: dict[str, bool] = {}

    for row in rows:
        config = row.get("config")
        task_id = row.get("task_id")
        if task_id is None:
            synthetic_count_by_config[config] = synthetic_count_by_config.get(config, 0) + 1
        else:
            key = (config, task_id)
            if key in seen_task_rows:  # (i) 同一 (config, task_id) の重複
                raise _SuiteError(f"baseline schema invalid: duplicate (config, task_id) {key}")
            seen_task_rows.add(key)
            has_normal_task_by_config[config] = True

    for config, count in synthetic_count_by_config.items():
        if count > 1:  # (ii) 同一configにsynthetic task_id=null行が複数
            raise _SuiteError(f"baseline schema invalid: multiple synthetic rows for config {config}")
        if has_normal_task_by_config.get(config):  # (iii) synthetic行と通常task行が同一configに混在
            raise _SuiteError(f"baseline schema invalid: synthetic and normal rows mixed for config {config}")


def _load_baseline(path: str) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise _SuiteError(f"failed to read/parse baseline: {path}") from exc
    _validate_baseline_schema(rows)
    return rows


def _aggregate_baseline(rows: list[dict]) -> dict:
    """baseline行群を現在run同様に (agent, stems, completion_rate, mean_fill) へ集計する。"""
    if not rows:
        raise _SuiteError("baseline is empty")
    agents = {row.get("agent") for row in rows}
    if len(agents) != 1:
        raise _SuiteError(f"baseline schema invalid: multiple/absent agent values: {agents}")
    agent = next(iter(agents))

    rows_by_stem: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_stem.setdefault(row["config"], []).append(row)
    stems = sorted(rows_by_stem.keys())
    summary = _aggregate_summary(agent, stems, rows_by_stem)
    return summary


def _compare_baseline(current_summary: dict, baseline_summary: dict) -> bool:
    """agent一致・unique stem集合一致を前提とし、非退行なら True を返す（前提はrun_suite側で検査済み）。

    退行 = (a) completion_rate の低下、または (b) baseline_mean_fill-current_mean_fill > 1.0。
    ちょうど1.0pt低下は退行としない。
    """
    if current_summary["completion_rate"] < baseline_summary["completion_rate"]:
        return False
    if (baseline_summary["mean_fill"] - current_summary["mean_fill"]) > 1.0:
        return False
    return True


# --- レポート生成（suite_results.jsonl / suite_summary.json / suite_report.md + stdout） ------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def _write_summary(path: Path, summary: dict) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)


def _fmt_cell(value) -> str:
    return "-" if value is None else str(value)


def _format_report(rows: list[dict]) -> str:
    header = "| " + " | ".join(_REPORT_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _REPORT_COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [_fmt_cell(row.get(col)) for col in _REPORT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_report(path: Path, rows: list[dict]) -> str:
    text = _format_report(rows)
    with open(path, "w") as f:
        f.write(text)
    return text


# --- 中核関数（公開契約：sys.exitしない、常に int を返す） ---------------------------------


def run_suite(
    config_dir: str,
    module_path: str,
    result_dir: str,
    baseline: str | None = None,
    timeout_sec: float = _TIMEOUT_DEFAULT,
) -> int:
    """課題(config)群を一括実行・集計・レポート生成し、exit codeに相当する int を返す。

    公開契約は詳細仕様書 v1.19/v1.20 §5.5 を正とする（CLI/schema/完走定義/exit code/c05）。
    予期される入力エラー・baseline schema不正・レポート出力エラーはこの関数の内部で捕捉し、
    2（規定の優先順位で1より優先）として返す。`sys.exit` は呼ばない。
    """
    try:
        timeout_sec = _validate_timeout(timeout_sec)
        configs = _discover_configs(config_dir)
    except _SuiteError:
        return _EXIT_INPUT_ERROR

    agent = _normalize_agent(module_path)
    result_dir_path = Path(result_dir)
    discovered_stems = [c.stem for c in configs]

    rows_by_stem: dict[str, list[dict]] = {}
    for cfg_path in configs:
        rows_by_stem[cfg_path.stem] = _run_one_config(
            cfg_path, module_path, agent, result_dir_path, timeout_sec
        )

    summary = _aggregate_summary(agent, discovered_stems, rows_by_stem)
    all_rows = [r for stem in discovered_stems for r in rows_by_stem[stem]]

    try:
        result_dir_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(result_dir_path / "suite_results.jsonl", all_rows)
        _write_summary(result_dir_path / "suite_summary.json", summary)
        report_text = _write_report(result_dir_path / "suite_report.md", all_rows)
    except (OSError, ValueError):
        return _EXIT_INPUT_ERROR

    print(report_text)

    if baseline is not None:
        try:
            baseline_rows = _load_baseline(baseline)
            baseline_summary = _aggregate_baseline(baseline_rows)
            if baseline_summary["agent"] != agent:
                raise _SuiteError(
                    f"baseline agent mismatch: {baseline_summary['agent']!r} != {agent!r}"
                )
            if set(baseline_summary["compared_config_stems"]) != set(summary["compared_config_stems"]):
                raise _SuiteError("baseline config stem set mismatch")
        except _SuiteError:
            return _EXIT_INPUT_ERROR

        if not _compare_baseline(summary, baseline_summary):
            return _EXIT_REGRESSION

    return _EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI引数を解析する（`run_test.py`/`run_local.py` と同名規約、新規フラグ名を作らない）。"""
    parser = argparse.ArgumentParser(
        description="T-030B E2E suite runner (thin wrapper repeating scripts.run_local per config)."
    )
    parser.add_argument("--config-dir", required=True, type=str, help="configs/local_suite 相当のディレクトリ")
    parser.add_argument("--module-path", required=True, type=str, help="agent module path")
    parser.add_argument("--result-dir", required=True, type=str, help="suite結果出力先ディレクトリ")
    parser.add_argument("--baseline", default=None, type=str, help="前回 suite_results.jsonl のパス（任意）")
    parser.add_argument("--timeout-sec", default=_TIMEOUT_DEFAULT, type=float, help="課題毎のsubprocess timeout秒")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLIエントリポイント。`run_suite()` の戻り値でプロセスを終了する。"""
    args = parse_args(argv)
    code = run_suite(
        config_dir=args.config_dir,
        module_path=args.module_path,
        result_dir=args.result_dir,
        baseline=args.baseline,
        timeout_sec=args.timeout_sec,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
