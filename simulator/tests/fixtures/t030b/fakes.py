"""T-030B契約テスト用fixture（`docs/実装詳細仕様書.md` §5.5 v1.19/v1.20、契約テスト設計計画）。

`scripts/run_suite.py` は課題ごとに `subprocess.run` で `python -m scripts.run_local` を
起動する契約（再利用機構）。本モジュールはその `subprocess.run` を差し替える spy/fake
（`FakeRunPlan`）と、`EvaluationApp` が書き出す結果JSON（`evaluation_results.json`）の
雛形、baseline `suite_results.jsonl` 生成ヘルパーを提供する。

source文字列検査ではなく、公開動作（`subprocess.run` 呼出しの引数・戻り値・例外）を
観測する方針（§5.5「テスト方針注記」）に従う。
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# --- 結果JSON雛形（EvaluationApp が書き出すスキーマ、app.py 準拠） -------------------------


def success_result(
    task_id: str = "000",
    fill_score: float = 42.0,
    num_placed_items: float = 0.5,
    policy_time: float = 1.2,
    optimization_time: float = 0.0,
    place_states: dict | None = None,
) -> dict:
    """`status=='success'` の単一task結果（validator NGなし、全 place_states=True）。"""
    if place_states is None:
        place_states = {"is_included": True, "is_valid": True, "is_placed_safe": True}
    return {
        task_id: {
            "evaluation": {"fill_score": fill_score, "num_placed_items": num_placed_items},
            "message": "ok",
            "status": "success",
            "place_states": place_states,
            "time_results": {"optimization": optimization_time, "policy": policy_time},
        }
    }


def validator_ng_result(
    task_id: str = "000",
    fill_score: float = 10.0,
    num_placed_items: float = 0.1,
    policy_time: float = 0.5,
    optimization_time: float = 0.0,
    n_false: int = 1,
) -> dict:
    """`status=='success'` かつ `place_states` に False を含む（validator NG、完走扱い）。"""
    place_states = {"is_included": True, "is_valid": True, "is_placed_safe": True}
    keys = list(place_states.keys())
    for i in range(min(n_false, len(keys))):
        place_states[keys[i]] = False
    return success_result(
        task_id=task_id,
        fill_score=fill_score,
        num_placed_items=num_placed_items,
        policy_time=policy_time,
        optimization_time=optimization_time,
        place_states=place_states,
    )


def format_error_result(task_id: str = "000", message: str = "boom") -> dict:
    """`status=='format_error'`（Agent/Runner未捕捉例外、`EvaluationApp` が捕捉・記録）。"""
    return {
        task_id: {
            "evaluation": None,
            "message": message,
            "status": "format_error",
        }
    }


def multi_task_result(*task_results: dict) -> dict:
    """複数taskの結果を1つの結果JSON dictへマージする（success/validator NG/format_error混在）。"""
    merged: dict = {}
    for r in task_results:
        merged.update(r)
    return merged


def nan_inf_result(task_id: str = "000") -> dict:
    """`fill_score=NaN`・`policy=Inf` を含む結果（SAN契約：書き出し前にnullへ変換される対象）。"""
    return {
        task_id: {
            "evaluation": {"fill_score": float("nan"), "num_placed_items": 0.3},
            "message": "ok",
            "status": "success",
            "place_states": {"is_included": True, "is_valid": True, "is_placed_safe": True},
            "time_results": {"optimization": float("-inf"), "policy": float("inf")},
        }
    }


# --- fake subprocess.run（`run_one_config`/`run_suite` の spy 対象を差し替える） -------------


@dataclass
class ConfigAction:
    """1 config に対して fake `subprocess.run` が取るべき挙動。

    kind:
        "success": `result_json` を `<result_dir>/evaluation_results.json` へ書き出し、
            `returncode` を返す（既定0）。
        "nonzero": 結果JSONを書かずに nonzero returncode を返す（exec exception）。
        "missing_json": returncode 0 だが結果JSONを書かない（exec exception）。
        "broken_json": 結果JSONとして不正な文字列を書く（parse不可・exec exception）。
        "timeout": `subprocess.TimeoutExpired` を送出する（exec exception）。
    """

    kind: str
    result_json: dict | None = None
    returncode: int = 0


@dataclass
class FakeRunPlan:
    """config stem -> `ConfigAction` の対応表を保持し、`subprocess.run` の代替として呼ばれる。

    `run_suite.subprocess.run` へ `monkeypatch.setattr` で差し替えて使う（spy）。
    呼出しの引数・順序は `self.calls` に記録される。
    """

    actions: dict[str, ConfigAction]
    calls: list[dict] = field(default_factory=list)

    def __call__(self, args, **kwargs):
        args = list(args)
        self.calls.append({"args": args, "kwargs": kwargs})
        result_dir = _extract_flag_value(args, "--result-dir")
        config_path = _extract_flag_value(args, "--config-path")
        stem = Path(config_path).stem
        action = self.actions.get(stem)
        if action is None:
            raise AssertionError(f"FakeRunPlan: no action configured for config stem {stem!r}")

        if action.kind == "timeout":
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

        if action.kind in ("success", "nonzero"):
            if result_dir is not None:
                os.makedirs(result_dir, exist_ok=True)
                with open(os.path.join(result_dir, "evaluation_results.json"), "w") as f:
                    json.dump(action.result_json or {}, f)
            returncode = 0 if action.kind == "success" else (action.returncode or 1)
            return subprocess.CompletedProcess(args=args, returncode=returncode)

        if action.kind == "missing_json":
            return subprocess.CompletedProcess(args=args, returncode=0)

        if action.kind == "broken_json":
            if result_dir is not None:
                os.makedirs(result_dir, exist_ok=True)
                with open(os.path.join(result_dir, "evaluation_results.json"), "w") as f:
                    f.write("{not valid json")
            return subprocess.CompletedProcess(args=args, returncode=0)

        raise AssertionError(f"FakeRunPlan: unknown action kind {action.kind!r}")


def _extract_flag_value(args: list[str], flag: str) -> str | None:
    """argvリストから `flag` の直後の値を取り出す（`--result-dir <値>` 形式想定）。"""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return None


# --- 合成config-dir（discovery/継続実行テスト用。内容は subprocess が fake のため無関係） -------


def write_stub_configs(config_dir: Path, stems: list[str]) -> list[Path]:
    """`config_dir` に空JSON `{}` の `<stem>.json` を作成する（発見・順序テスト専用）。"""
    paths = []
    config_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        p = config_dir / f"{stem}.json"
        with open(p, "w") as f:
            json.dump({}, f)
        paths.append(p)
    return paths


# --- baseline suite_results.jsonl 生成（BASE契約テスト用） --------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_row(
    config: str,
    agent: str,
    task_id: str | None,
    status: str,
    fill_score: float | None = None,
    num_placed_items: float | None = None,
    n_validator_ng: int | None = None,
    format_error: bool = False,
    exec_exception: bool = False,
    policy_time: float | None = None,
    optimization_time: float | None = None,
    wall_time: float = 1.0,
) -> dict:
    """`suite_results.jsonl` の行スキーマ（詳細仕様書 v1.19/v1.20 ROW契約）に沿った1行を作る。"""
    return {
        "config": config,
        "agent": agent,
        "task_id": task_id,
        "status": status,
        "fill_score": fill_score,
        "num_placed_items": num_placed_items,
        "n_validator_ng": n_validator_ng,
        "format_error": format_error,
        "exec_exception": exec_exception,
        "policy_time": policy_time,
        "optimization_time": optimization_time,
        "wall_time": wall_time,
    }


def make_baseline_rows(
    agent: str,
    stems: list[str],
    mean_fill_per_stem: dict[str, float] | None = None,
) -> list[dict]:
    """各stemにつき1つの正常successな行を持つ、正常baseline行群を作る（BASE-008用の非退行baseline）。"""
    mean_fill_per_stem = mean_fill_per_stem or {}
    rows = []
    for stem in stems:
        fill = mean_fill_per_stem.get(stem, 50.0)
        rows.append(
            make_row(
                config=stem,
                agent=agent,
                task_id="000",
                status="success",
                fill_score=fill,
                num_placed_items=1.0,
                n_validator_ng=0,
            )
        )
    return rows
