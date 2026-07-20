"""T-028 telemetry 契約テスト用の最小fixture（詳細仕様書 v1.22 §4.12/§4.13/§5.7）。

`tests/test_t024_agent_policy.py`／`tests/test_t024_emergency.py` の container/item 辞書生成
パターンをそのまま踏襲する（決定論的な最小状態を組み立てるだけで、公式 PyBullet 環境は使わない）。
`tests/fixtures/t030b/fakes.py` と同様、source文字列検査ではなく公開動作・出力schemaを
検証するテストのための入力データのみをここに置く。

未実装モジュール（`agents.heuristic.telemetry`／`scripts.check_telemetry`）への依存は、
本モジュールでは行わない（呼び出し側テストが各関数内で遅延importする）。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

INNER_MIN_SMALL = np.array([-0.02, -0.02, 0.02], dtype=np.float64)
INNER_MAX_SMALL = np.array([0.02, 0.02, 0.05], dtype=np.float64)  # どの荷物も入らない極小コンテナ
INNER_MIN_BIG = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_BIG = np.array([0.40, 0.40, 1.02], dtype=np.float64)  # 十分な広さのコンテナ


def _item_dict(index, size, mass=2.0):
    length, width, height = (float(v) for v in size)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "mass": float(mass), "is_prioritized": False, "is_soft": False,
        "belongs_to": None, "pos": None, "orn": None,
        "lateralFriction": 0.5, "rollingFriction": 0.01, "spinningFriction": 0.01,
        "restitution": 0.1, "angularDamping": 0.05,
    }


def _container_dict(index, offset_x, inner_min_rel, inner_max_rel, thickness=0.02, buffer=0.02):
    imin = np.asarray(inner_min_rel, dtype=np.float64)
    imax = np.asarray(inner_max_rel, dtype=np.float64)
    mid = (imin + imax) / 2.0
    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px + offset_x, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer
    center = (float(offset_x), 0.0, height / 2.0 + buffer)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness, "center": center,
        "n_vecs": n_vecs, "points": points, "volume": float(np.prod(imax - imin)),
        "shelf": False, "is_prioritized": False, "packed_items": [],
    }


def placeable_case(pool_index: int = 0):
    """候補が選択され `decided_layer>=1` になる init/observation ペア（AG-NORMAL-LINE用）。

    container0=極小（不適合）・container1=十分な広さ、荷物1件（container1にのみ収まる）。
    """
    c0 = _container_dict(0, 0.0, INNER_MIN_SMALL, INNER_MAX_SMALL)
    c1 = _container_dict(1, 2.0, INNER_MIN_BIG, INNER_MAX_BIG)
    pool_list = [_item_dict(index=pool_index, size=(0.1, 0.1, 0.1))]
    init = {"optimize": True, "lookahead_k": 5, "container_list": [c0, c1]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((2, 64, 64), dtype=np.float32),
        "container_list": [c0, c1], "pool_list": pool_list,
    }
    return init, observation


def no_candidate_case(pool_index: int = 0):
    """全 container が極小で候補ゼロ → emergency（`decided_layer=0`・`p_ng_chosen` null、AG-EMPTY用）。"""
    c0 = _container_dict(0, 0.0, INNER_MIN_SMALL, INNER_MAX_SMALL)
    pool_list = [_item_dict(index=pool_index, size=(0.1, 0.1, 0.1))]
    init = {"optimize": True, "lookahead_k": 5, "container_list": [c0]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((1, 64, 64), dtype=np.float32),
        "container_list": [c0], "pool_list": pool_list,
    }
    return init, observation


def make_heuristic_agent(init: dict):
    """`agents.heuristic.agent.Agent` を生成し `get_init_states` 済みで返す（関数内import）。"""
    from agents.heuristic.agent import Agent

    agent = Agent(module_path="agents/heuristic")
    agent.get_init_states(init)
    return agent


def patch_dual(monkeypatch, source_module, agent_module_name: str, attr: str, fn):
    """source_module／agent_module双方の同名属性へpatchする（import文の形式に非依存。
    `tests/test_t024_emergency.py::_patch_dual` と同一パターン）。"""
    monkeypatch.setattr(source_module, attr, fn, raising=False)
    agent_module = importlib.import_module(agent_module_name)
    monkeypatch.setattr(agent_module, attr, fn, raising=False)


def sample_local_telemetry() -> dict:
    """§4.12 局所telemetry辞書の代表値（reject集計・decided_layer・layer_errorを含む）。

    row整形関数（未実装 `agents.heuristic.telemetry` が想定する入力）のテスト用。
    """
    return {
        "n_cand0": 12,
        "n_after_dims": 8,
        "n_after_geo": 5,
        "n_lpath_pass": 3,
        "reject_reason_counts": {"dims": 4, "overlap": 2, "ceiling": 2, "inclusion": 0},
        "decided_layer": 1,
        "layer_error": [],
    }


def sample_timing() -> dict:
    """段階時間 accumulator の代表値（finite ≥0、§4.13 D2）。"""
    return {
        "t_state": 0.01,
        "t_enum": 0.02,
        "t_mask": 0.03,
        "t_lpath": 0.04,
        "t_total": 0.15,
    }


def valid_row(step: int = 0) -> dict:
    """check_telemetry.py の正常入力1行（16キーちょうど、schema準拠）。"""
    return {
        "step": step,
        "first_step": step == 0,
        "t_total": 0.15,
        "t_state": 0.01,
        "t_enum": 0.02,
        "t_mask": 0.03,
        "t_lpath": 0.04,
        "n_cand0": 12,
        "n_after_dims": 8,
        "n_after_geo": 5,
        "n_lpath_pass": 3,
        "decided_layer": 1,
        "regime": None,
        "p_ng_chosen": 0.12,
        "reject_top3": [{"reason": "dims", "count": 4}, {"reason": "ceiling", "count": 2}],
        "layer_error": [],
    }


def write_jsonl(path: Path, lines: list[str]) -> Path:
    """生の文字列行（JSON化済みまたは意図的に壊れた文字列）をJSONLとして書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def write_valid_jsonl(path: Path, n_rows: int = 3) -> Path:
    """`valid_row()` を `n_rows` 件、正しい JSON として書き出す。"""
    lines = [json.dumps(valid_row(step=i)) for i in range(n_rows)]
    return write_jsonl(path, lines)


def single_task_config_path(tmp_path: Path, n_items: int = 3) -> Path:
    """`configs/local_suite/c01.json` を1 task・少数item数に絞ったコピーを `tmp_path` へ書く
    （RL-STEP-SINGLE用。既存 `c01.json` 自体は変更しない）。

    Args:
        tmp_path: pytest の `tmp_path` fixture。
        n_items: 残す item 数（実行時間短縮のため既定3件）。

    Returns:
        書き出した config ファイルの Path。
    """
    src = Path(__file__).resolve().parents[3] / "configs" / "local_suite" / "c01.json"
    with open(src) as f:
        config = json.load(f)
    assert len(config) == 1, "c01.json は単一task前提（RL-STEP-SINGLEの成立条件）"
    (task_id,) = config.keys()
    config[task_id]["item_stream"]["item_list"] = config[task_id]["item_stream"]["item_list"][:n_items]
    out_path = tmp_path / "c01_single_task_small.json"
    with open(out_path, "w") as f:
        json.dump(config, f)
    return out_path
