"""T-016A: L字経路 公式PyBulletオラクル／凍結ゴールデン生成（詳細仕様書 §4.5・§5.6・§6）。

公式 `PlacementValidator.check_transport_path`（`src/ground_handling/validator.py`）を
**無改変のまま**参照オラクルとして実行し、`_move_item` をspyでラップしてYレグ・Xレグの
実行結果を記録した、決定論的な凍結ゴールデン1,000件を生成する。

公式コードは読み取り・import・インスタンス化・実行のみ行う。コピー・改変・整形は行わない。
本チケット（T-016A）では純NumPyプロキシ（T-016B）は実装しない。

実行（`simulator/` で）:
    python -m scripts.gen_l_path_golden --config configs/sample_config.json

seed・件数・カテゴリ分布・出力先は固定契約であり、CLI引数で変更できない
（詳細仕様書 §4.5「ゴールデン生成条件」v1.12追補）。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import subprocess
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from src.ground_handling.containers import Container
from src.ground_handling.items import Item
from src.ground_handling.utils import get_half_ext
from src.ground_handling.validator import PlacementValidator

logger = logging.getLogger("packing")

# ---------------------------------------------------------------------------
# 固定契約（§4.5 v1.12追補。CLI引数で変更しない）
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "l-path-golden-v1"
GENERATOR_VERSION = "1.0.0"
BASE_SEED = 16016

# 仕様参照HEAD（§4.5「公式コードの事実」の転記元。起動時に working tree の validator.py が
# このHEAD時点の内容とbyte-identicalであることを検証する。現在のリポジトリHEADとは別物）。
OFFICIAL_GIT_HEAD = "abf630f"
OFFICIAL_SOURCE_FILE = "simulator/src/ground_handling/validator.py"
OFFICIAL_CLASS = "PlacementValidator"
OFFICIAL_FUNCTION = "check_transport_path"

CONFIG_TASK = "000"
CANDIDATE_INDEX = 999

CATEGORIES = [
    "clear_path",
    "y_leg_blocked",
    "x_leg_blocked",
    "safety_margin_boundary",
    "random_scene",
]
CASES_PER_CATEGORY = 200
MAX_ATTEMPTS_PER_CATEGORY = CASES_PER_CATEGORY * 10  # §4.5「生成試行上限」

# safety_margin_boundary の8水準（正=衝突域から離す、負=衝突域へ近づける。§4.5）
BOUNDARY_DELTAS_MM = [-20.0, -5.0, -1.0, -0.5, 0.5, 1.0, 5.0, 20.0]
CASES_PER_BOUNDARY_LEVEL = 25

CANDIDATE_MIN_SIZE_M = 0.15
CANDIDATE_MAX_SIZE_M = 0.45
TARGET_MARGIN_M = 0.05

GOLDEN_RELPATH = "simulator/datasets/fixtures/l_path_golden_v1.jsonl"
REPORT_RELPATH = "docs/parity/l_path_golden_v1.md"

REQUIRED_CASE_FIELDS = [
    "schema_version", "case_id", "category", "random_seed", "generator_version",
    "official_git_head", "official_validator_git_blob", "official_validator_file_sha256",
    "official_validator_last_commit", "generator_repo_head",
    "official_source_file", "official_class", "official_function",
    "config_relpath", "config_file_sha256", "config_task",
    "raw_validator_config", "effective_validator_config", "validator_config_hash",
    "container_raw_config", "container_origin_world", "shelf_and_wall_and_cut_and_ceiling_info",
    "candidate_item_index", "candidate_size", "candidate_orientation",
    "candidate_target_pos", "candidate_item_kwargs",
    "placed_items", "official_inclusion_verdict", "move_calls", "move_call_count",
    "official_verdict", "official_exception", "category_metadata",
]


# ---------------------------------------------------------------------------
# 決定論的seed導出（§4.5「固定seedとカテゴリ別派生」）
# ---------------------------------------------------------------------------
def category_seed(category: str) -> int:
    """カテゴリ単位の派生seed（監査・レポート記録用）。"""
    digest = hashlib.sha256(f"lpath-v1:{BASE_SEED}:{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def case_seed(category: str, i: int) -> int:
    """ケース（試行）単位の派生seed。`hash()` は使用しない（プロセス間で不安定なため）。"""
    digest = hashlib.sha256(f"lpath-v1:{BASE_SEED}:{category}:{i:04d}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


# ---------------------------------------------------------------------------
# 公式コード同一性ガード（§4.5「公式コードの事実」・ユーザー承認済み修正 #2）
# ---------------------------------------------------------------------------
def git_output(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def check_official_identity(repo_root: Path) -> tuple[str, str]:
    """`abf630f` 時点の validator.py と working tree の内容が一致することを確認する。

    不一致の場合は生成を開始せず、付録D形式相当のエラーで停止する（詳細仕様書 §4.5・
    ユーザー承認済み修正#2）。

    Returns:
        (git blob hash（working tree・abf630f両方で一致した値）, working tree内容のSHA-256)
    """
    ref_blob = git_output(repo_root, ["rev-parse", f"{OFFICIAL_GIT_HEAD}:{OFFICIAL_SOURCE_FILE}"])
    work_blob = git_output(repo_root, ["hash-object", str(repo_root / OFFICIAL_SOURCE_FILE)])
    if ref_blob != work_blob:
        raise RuntimeError(
            "[付録D] official validator.py content mismatch:\n"
            f"  1. 停止した箇所: check_official_identity（gen_l_path_golden.py起動時ガード）\n"
            f"  2. 矛盾の内容: 仕様参照HEAD {OFFICIAL_GIT_HEAD} の validator.py blob={ref_blob} が"
            f" working tree の blob={work_blob} と一致しない\n"
            "  3. 対応: 公式コードが変更されている可能性があるため、ゴールデン生成を中止する。"
            " 人間による確認が必要。"
        )
    content = (repo_root / OFFICIAL_SOURCE_FILE).read_bytes()
    file_sha256 = hashlib.sha256(content).hexdigest()
    return work_blob, file_sha256


# ---------------------------------------------------------------------------
# 生成コンテキスト（可変グローバル状態を避けるため明示的に受け渡す。§2.2）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationContext:
    base_cdict: dict
    raw_validator_config: dict
    effective_validator_config: dict
    validator_config_hash: str
    config_relpath: str
    config_file_sha256: str
    config_task: str
    repo_root: Path
    official_git_head: str
    official_validator_git_blob: str
    official_validator_file_sha256: str
    official_validator_last_commit: str
    generator_repo_head: str


def compute_effective_validator_config(raw_validator_config: dict) -> dict:
    """公式 `PlacementValidator` インスタンスの実効属性を読み取る（ユーザー承認済み修正#3。
    `start_margin` 等をスクリプト側で再定義せず、公式コンストラクタの`.get`既定込みの実効値を
    読む）。"""
    client = BulletClient(connection_mode=p.DIRECT)
    try:
        validator = PlacementValidator(client, raw_validator_config, render_mode=None)
        return {
            "safety_margin": float(validator.safety_margin),
            "start_margin": float(validator.start_margin),
            "inclusion_margin": float(validator.inclusion_margin),
            "start_z": float(validator.start_z),
            "ceiling_margin": float(validator.ceiling_margin),
        }
    finally:
        client.disconnect()


def build_generation_context(repo_root: Path, config_path: Path) -> GenerationContext:
    official_validator_git_blob, official_validator_file_sha256 = check_official_identity(repo_root)
    official_validator_last_commit = git_output(
        repo_root, ["log", "-n", "1", "--format=%H", "--", OFFICIAL_SOURCE_FILE]
    )
    generator_repo_head = git_output(repo_root, ["rev-parse", "HEAD"])

    config_bytes = config_path.read_bytes()
    config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
    full_cfg = json.loads(config_bytes)
    task_cfg = full_cfg[CONFIG_TASK]
    raw_validator_config = task_cfg["validator"]
    base_cdict_full = task_cfg["containers"]["container_list"][0]
    base_cdict = {k: v for k, v in base_cdict_full.items() if k != "packed_items"}

    effective_validator_config = compute_effective_validator_config(raw_validator_config)
    validator_config_hash = hashlib.sha256(
        json.dumps(raw_validator_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    try:
        config_relpath = str(config_path.resolve().relative_to(repo_root))
    except ValueError:
        config_relpath = str(config_path)

    return GenerationContext(
        base_cdict=base_cdict,
        raw_validator_config=raw_validator_config,
        effective_validator_config=effective_validator_config,
        validator_config_hash=validator_config_hash,
        config_relpath=config_relpath,
        config_file_sha256=config_file_sha256,
        config_task=CONFIG_TASK,
        repo_root=repo_root,
        official_git_head=OFFICIAL_GIT_HEAD,
        official_validator_git_blob=official_validator_git_blob,
        official_validator_file_sha256=official_validator_file_sha256,
        official_validator_last_commit=official_validator_last_commit,
        generator_repo_head=generator_repo_head,
    )


# ---------------------------------------------------------------------------
# 公式オブジェクトの構築・実行（読み取り・import・インスタンス化・実行のみ。§1.3-6）
# ---------------------------------------------------------------------------
def _bind_move_item_spy(validator: PlacementValidator) -> list[dict]:
    """`_move_item` をspyでラップする。元関数は必ず呼び出し、引数・戻り値は変更しない
    （詳細仕様書 §4.5「spyの契約」）。"""
    calls: list[dict] = []
    orig = PlacementValidator._move_item

    def spy(self, *, container, item, item_id, start_pos, target_pos, target_orn, steps):
        ok, cur = orig(
            self, container=container, item=item, item_id=item_id,
            start_pos=start_pos, target_pos=target_pos, target_orn=target_orn, steps=steps,
        )
        calls.append({
            "leg": "y" if not calls else "x",
            "start_pos": [float(v) for v in start_pos],
            "target_pos": [float(v) for v in target_pos],
            "passed": bool(ok),
        })
        return ok, cur

    validator._move_item = types.MethodType(spy, validator)
    return calls


def execute_case(
    cdict: dict,
    offset_x: float,
    placed_specs: list[dict],
    candidate_size: tuple[float, float, float],
    candidate_orn: int,
    target_world: tuple[float, float, float],
    raw_validator_config: dict,
    with_spy: bool,
) -> dict:
    """公式シーンを構築し `check_inclusion` / `check_transport_path` を実行する。

    候補body自体は公式 `check_transport_path` が内部で `item.spawn` する（我々は候補を
    事前spawnしない。公式Itemインスタンスを渡すのみ。ユーザー承認済み修正#6）。

    Returns:
        {"verdict", "exception", "move_calls", "inclusion_verdict"}
    """
    client = BulletClient(connection_mode=p.DIRECT)
    result: dict = {
        "verdict": None, "exception": None, "move_calls": [], "inclusion_verdict": None,
    }
    try:
        container = Container(offset_x=offset_x, packed_items=[], **cdict)
        with contextlib.redirect_stdout(io.StringIO()):
            container.create(client)
        for spec in placed_specs:
            item = Item(
                index=spec["index"], length=spec["size"][0], width=spec["size"][1],
                height=spec["size"][2], mass=1.0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                item.spawn(client, initial_pos=spec["pos_world"], initial_orn=spec["orn_quat"])
            container.packed_items.append(item)

        validator = PlacementValidator(client, raw_validator_config, render_mode=None)
        candidate = Item(
            index=CANDIDATE_INDEX, length=candidate_size[0], width=candidate_size[1],
            height=candidate_size[2], mass=1.0,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result["inclusion_verdict"] = bool(
                validator.check_inclusion(container, candidate, target_world, candidate_orn)
            )
        calls = _bind_move_item_spy(validator) if with_spy else None
        with contextlib.redirect_stdout(io.StringIO()):
            result["verdict"] = bool(
                validator.check_transport_path(container, candidate, target_world, candidate_orn)
            )
        if calls is not None:
            result["move_calls"] = calls
    except Exception as exc:  # noqa: BLE001 - 意図的に広く捕捉。呼び出し側が再試行する
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        client.disconnect()
    return result


# ---------------------------------------------------------------------------
# シーン合成ヘルパ（すべて「合成→公式実行で検証→不一致なら再試行」の方針。私的な
# lane_x/x_min/x_max 式の再実装は行わない）
# ---------------------------------------------------------------------------
def sample_candidate_size(rng: np.random.Generator) -> tuple[float, float, float]:
    return tuple(float(v) for v in rng.uniform(CANDIDATE_MIN_SIZE_M, CANDIDATE_MAX_SIZE_M, size=3))


def sample_interior_target(
    rng: np.random.Generator, cdict: dict, half: list[float],
    margin: float = TARGET_MARGIN_M, x_bias: str = "any",
) -> tuple[float, float, float] | None:
    """コンテナ内壁マージン内でtargetを乱択する。lane_x等の私的計算式は使わず、
    採否は必ず公式 `check_inclusion`/`check_transport_path` の実行結果で判定する。"""
    length, width, height = cdict["length"], cdict["width"], cdict["height"]
    thickness = cdict["thickness"]
    x_lo = -length / 2 + thickness + half[0] + margin
    x_hi = length / 2 - thickness - half[0] - margin
    y_lo = -width / 2 + thickness + half[1] + margin
    y_hi = width / 2 - thickness - half[1] - margin
    z_lo = thickness + half[2] + margin
    z_hi = height - thickness - half[2] - margin
    if x_lo >= x_hi or y_lo >= y_hi or z_lo >= z_hi:
        return None
    span = x_hi - x_lo
    if x_bias == "left":
        xr = (x_lo, x_lo + 0.4 * span)
    elif x_bias == "right":
        xr = (x_hi - 0.4 * span, x_hi)
    else:
        xr = (x_lo, x_hi)
    x = float(rng.uniform(*xr))
    y = float(rng.uniform(y_lo, y_hi))
    z = float(rng.uniform(z_lo, z_hi))
    return (x, y, z)


def sample_container_variant(rng: np.random.Generator, base: dict) -> dict:
    """random_scene用のコンテナ寸法バリエーション。無効ジオメトリを避けるため保守的な
    範囲に収める（`write_open_cut_corner_cup_obj` の制約: cut_x<length, cut_y<height,
    wall/bottom>0, depth>bottom）。実際の妥当性は公式 `Container.create` 実行で最終確認する
    （例外時は呼び出し側が該当試行を破棄・再試行する）。"""
    length = float(base["length"] * rng.uniform(0.85, 1.15))
    width = float(base["width"] * rng.uniform(0.85, 1.15))
    height = float(base["height"] * rng.uniform(0.85, 1.15))
    thickness = float(base["thickness"])
    buffer = float(base["buffer"])
    cut_x = float(min(base["cut_x"] * (length / base["length"]), length * 0.3))
    cut_y = float(min(base["cut_y"] * (height / base["height"]), height * 0.3))
    cut_x = max(cut_x, 0.05)
    cut_y = max(cut_y, 0.05)
    require_shelf = bool(rng.random() < 0.3)
    return {
        "index": 0, "length": length, "width": width, "height": height,
        "thickness": thickness, "buffer": buffer, "cut_x": cut_x, "cut_y": cut_y,
        "require_shelf": require_shelf, "is_prioritized": False,
    }


def build_shelf_wall_info(cdict: dict) -> dict:
    """棚・小棚・cut・壁・天井の再構築用シーンメタデータ（ゴールデン契約の必須フィールド。
    ユーザー承認済み修正#1）。

    `resting_surfaces_z_world`/`ceiling_surfaces_z_world` は validator.py:103-111
    （`docs/interface_notes.md` §B に読解記録済み）の定数配列を、本スクリプトが構築した
    cdict の値で評価した結果である。Container.local_to_global はXのみをオフセットし
    Z軸は不変（containers.py:238-240）のため、これらの値はそのままworld z として扱える。
    公式の判定ロジック自体を再実装するものではなく、シーン入力のスナップショットである。
    """
    thickness = cdict["thickness"]
    height = cdict["height"]
    buffer = cdict["buffer"]
    require_shelf = cdict["require_shelf"]
    return {
        "require_shelf": require_shelf,
        "shelf_body_present": require_shelf,       # containers.py::create（require_shelf時のみ）
        "small_shelf_body_present": True,           # containers.py::create（常時生成、interface_notes §K.2）
        "cut_x": cdict["cut_x"],
        "cut_y": cdict["cut_y"],
        "thickness": thickness,
        "buffer": buffer,
        "wall_extents_rel": {
            "outer_half_extents": [cdict["length"] / 2.0, cdict["width"] / 2.0, cdict["height"] / 2.0],
            "thickness": thickness,
        },
        "resting_surfaces_z_world": [thickness, height / 2.0 + thickness + buffer],
        "ceiling_surfaces_z_world": [height / 2.0 + buffer, height + buffer - thickness],
    }


def build_case_record(
    *, category: str, attempt_seed: int, ctx: GenerationContext, cdict: dict, offset_x: float,
    candidate_size: tuple[float, float, float], candidate_orn: int,
    target_world: tuple[float, float, float], placed_specs: list[dict],
    exec_result: dict, category_metadata: dict,
) -> dict:
    move_calls = exec_result["move_calls"]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": None,  # 採択順で後から確定
        "category": category,
        "random_seed": attempt_seed,
        "generator_version": GENERATOR_VERSION,
        "official_git_head": ctx.official_git_head,
        "official_validator_git_blob": ctx.official_validator_git_blob,
        "official_validator_file_sha256": ctx.official_validator_file_sha256,
        "official_validator_last_commit": ctx.official_validator_last_commit,
        "generator_repo_head": ctx.generator_repo_head,
        "official_source_file": OFFICIAL_SOURCE_FILE,
        "official_class": OFFICIAL_CLASS,
        "official_function": OFFICIAL_FUNCTION,
        "config_relpath": ctx.config_relpath,
        "config_file_sha256": ctx.config_file_sha256,
        "config_task": ctx.config_task,
        "raw_validator_config": ctx.raw_validator_config,
        "effective_validator_config": ctx.effective_validator_config,
        "validator_config_hash": ctx.validator_config_hash,
        "container_raw_config": {
            "index": cdict["index"], "length": cdict["length"], "width": cdict["width"],
            "height": cdict["height"], "thickness": cdict["thickness"], "cut_x": cdict["cut_x"],
            "cut_y": cdict["cut_y"], "buffer": cdict["buffer"], "require_shelf": cdict["require_shelf"],
        },
        "container_origin_world": offset_x,
        "shelf_and_wall_and_cut_and_ceiling_info": build_shelf_wall_info(cdict),
        "candidate_item_index": CANDIDATE_INDEX,
        "candidate_size": list(candidate_size),
        "candidate_orientation": candidate_orn,
        "candidate_target_pos": list(target_world),
        "candidate_item_kwargs": {"mass": 1.0, "is_prioritized": False, "is_soft": False},
        "placed_items": [
            {
                "index": s["index"], "size": list(s["size"]),
                "pos_world": list(s["pos_world"]), "orn_quat": list(s["orn_quat"]),
                "item_kwargs": {"mass": 1.0, "is_prioritized": False, "is_soft": False},
            }
            for s in placed_specs
        ],
        "official_inclusion_verdict": exec_result["inclusion_verdict"],
        "move_calls": move_calls,
        "move_call_count": len(move_calls),
        "official_verdict": exec_result["verdict"],
        "official_exception": {"occurred": False, "type": None, "message": None},
        "category_metadata": category_metadata,
    }


# ---------------------------------------------------------------------------
# カテゴリ別ケース生成（§4.5「カテゴリ契約」。各関数は1試行=1回呼び出しで、内部で
# 有界な再試行を行い、成功時はケースdict、失敗時はNoneを返す）
# ---------------------------------------------------------------------------
def generate_clear_path_case(ctx: GenerationContext, rng: np.random.Generator, attempt_seed: int, max_tries: int = 15) -> dict | None:
    cdict = ctx.base_cdict
    for _ in range(max_tries):
        offset_x = float(rng.uniform(0.0, 5.0))  # 純X並進のみ（containers.py::local_to_global）
        size = sample_candidate_size(rng)
        orn = int(rng.integers(0, 6))
        half = get_half_ext(list(size), orn)
        target_rel = sample_interior_target(rng, cdict, half)
        if target_rel is None:
            continue
        target_world = (target_rel[0] + offset_x, target_rel[1], target_rel[2])
        res = execute_case(cdict, offset_x, [], size, orn, target_world, ctx.raw_validator_config, with_spy=True)
        if res["exception"] or not res["inclusion_verdict"]:
            continue
        calls = res["move_calls"]
        if len(calls) == 2 and calls[0]["passed"] and calls[1]["passed"]:
            return build_case_record(
                category="clear_path", attempt_seed=attempt_seed, ctx=ctx, cdict=cdict, offset_x=offset_x,
                candidate_size=size, candidate_orn=orn, target_world=target_world, placed_specs=[],
                exec_result=res, category_metadata={},
            )
    return None


def generate_y_leg_blocked_case(ctx: GenerationContext, rng: np.random.Generator, attempt_seed: int, max_tries: int = 15) -> dict | None:
    cdict = ctx.base_cdict
    for _ in range(max_tries):
        offset_x = float(rng.uniform(0.0, 5.0))
        size = sample_candidate_size(rng)
        orn = int(rng.integers(0, 6))
        half = get_half_ext(list(size), orn)
        target_rel = sample_interior_target(rng, cdict, half)
        if target_rel is None:
            continue
        target_world = (target_rel[0] + offset_x, target_rel[1], target_rel[2])
        clear = execute_case(cdict, offset_x, [], size, orn, target_world, ctx.raw_validator_config, with_spy=True)
        if clear["exception"] or not clear["inclusion_verdict"] or len(clear["move_calls"]) != 2:
            continue
        if not (clear["move_calls"][0]["passed"] and clear["move_calls"][1]["passed"]):
            continue
        yleg = clear["move_calls"][0]
        t = float(rng.uniform(0.3, 0.7))
        y_mid = yleg["start_pos"][1] + t * (yleg["target_pos"][1] - yleg["start_pos"][1])
        lane_x, leg_z = yleg["target_pos"][0], yleg["target_pos"][2]
        blk_half = (
            half[0] + float(rng.uniform(0.02, 0.06)),
            float(rng.uniform(0.05, 0.15)),
            half[2] + float(rng.uniform(0.02, 0.06)),
        )
        blk_spec = {
            "index": 1, "size": tuple(2 * v for v in blk_half),
            "pos_world": (lane_x, y_mid, leg_z), "orn_quat": (0.0, 0.0, 0.0, 1.0),
        }
        final = execute_case(cdict, offset_x, [blk_spec], size, orn, target_world, ctx.raw_validator_config, with_spy=True)
        if final["exception"] or not final["inclusion_verdict"]:
            continue
        calls = final["move_calls"]
        if len(calls) == 1 and calls[0]["passed"] is False:
            return build_case_record(
                category="y_leg_blocked", attempt_seed=attempt_seed, ctx=ctx, cdict=cdict, offset_x=offset_x,
                candidate_size=size, candidate_orn=orn, target_world=target_world, placed_specs=[blk_spec],
                exec_result=final, category_metadata={},
            )
    return None


def generate_x_leg_blocked_case(
    ctx: GenerationContext, rng: np.random.Generator, attempt_seed: int,
    target_max_tries: int = 15, blocker_max_tries: int = 20,
) -> dict | None:
    cdict = ctx.base_cdict
    for _ in range(target_max_tries):
        offset_x = float(rng.uniform(0.0, 5.0))
        size = sample_candidate_size(rng)
        orn = int(rng.integers(0, 6))
        half = get_half_ext(list(size), orn)
        target_rel = sample_interior_target(rng, cdict, half, x_bias="left")
        if target_rel is None:
            continue
        target_world = (target_rel[0] + offset_x, target_rel[1], target_rel[2])
        clear = execute_case(cdict, offset_x, [], size, orn, target_world, ctx.raw_validator_config, with_spy=True)
        if clear["exception"] or not clear["inclusion_verdict"] or len(clear["move_calls"]) != 2:
            continue
        if not (clear["move_calls"][0]["passed"] and clear["move_calls"][1]["passed"]):
            continue
        xleg = clear["move_calls"][1]
        x0, x1 = xleg["start_pos"][0], xleg["target_pos"][0]
        if abs(x1 - x0) < 0.03:
            continue  # target が lane 内でクランプされず、実Xレグにならなかった
        y, z = xleg["target_pos"][1], xleg["target_pos"][2]
        for _ in range(blocker_max_tries):
            f = float(rng.uniform(0.55, 0.92))  # 終端寄り。Yレグ回廊の角食い込みを避ける
            bx = x0 + f * (x1 - x0)
            blk_half = (
                float(rng.uniform(0.03, 0.08)),
                half[1] + float(rng.uniform(0.02, 0.06)),
                half[2] + float(rng.uniform(0.02, 0.06)),
            )
            blk_spec = {
                "index": 1, "size": tuple(2 * v for v in blk_half),
                "pos_world": (bx, y, z), "orn_quat": (0.0, 0.0, 0.0, 1.0),
            }
            final = execute_case(cdict, offset_x, [blk_spec], size, orn, target_world, ctx.raw_validator_config, with_spy=True)
            if final["exception"] or not final["inclusion_verdict"]:
                continue
            calls = final["move_calls"]
            if len(calls) == 2 and calls[0]["passed"] is True and calls[1]["passed"] is False:
                return build_case_record(
                    category="x_leg_blocked", attempt_seed=attempt_seed, ctx=ctx, cdict=cdict, offset_x=offset_x,
                    candidate_size=size, candidate_orn=orn, target_world=target_world, placed_specs=[blk_spec],
                    exec_result=final, category_metadata={},
                )
    return None


def generate_boundary_case(
    ctx: GenerationContext, rng: np.random.Generator, attempt_seed: int, delta_mm: float, max_tries: int = 15,
) -> dict | None:
    cdict = ctx.base_cdict
    safety_margin = ctx.effective_validator_config["safety_margin"]
    gap = safety_margin + delta_mm / 1000.0
    for _ in range(max_tries):
        size = sample_candidate_size(rng)
        orn = int(rng.integers(0, 6))
        half = get_half_ext(list(size), orn)
        target = sample_interior_target(rng, cdict, half, margin=0.08)
        if target is None:
            continue
        clear = execute_case(cdict, 0.0, [], size, orn, target, ctx.raw_validator_config, with_spy=True)
        if clear["exception"] or not clear["inclusion_verdict"] or len(clear["move_calls"]) != 2:
            continue
        yleg = clear["move_calls"][0]
        t = float(rng.uniform(0.3, 0.7))
        y_mid = yleg["start_pos"][1] + t * (yleg["target_pos"][1] - yleg["start_pos"][1])
        lane_x, leg_z = yleg["target_pos"][0], yleg["target_pos"][2]
        side = 1.0 if rng.random() < 0.5 else -1.0
        blk_half_x = float(rng.uniform(0.05, 0.12))
        bx = lane_x + side * (half[0] + blk_half_x + gap)
        blk_spec = {
            "index": 1,
            "size": (2 * blk_half_x, 2 * (half[1] + 0.05), 2 * (half[2] + 0.05)),
            "pos_world": (bx, y_mid, leg_z), "orn_quat": (0.0, 0.0, 0.0, 1.0),
        }
        final = execute_case(cdict, 0.0, [blk_spec], size, orn, target, ctx.raw_validator_config, with_spy=True)
        if final["exception"] or not final["inclusion_verdict"]:
            continue
        return build_case_record(
            category="safety_margin_boundary", attempt_seed=attempt_seed, ctx=ctx, cdict=cdict, offset_x=0.0,
            candidate_size=size, candidate_orn=orn, target_world=target, placed_specs=[blk_spec],
            exec_result=final, category_metadata={"delta_mm": delta_mm, "gap_m": gap, "side": side},
        )
    return None


def generate_random_scene_case(ctx: GenerationContext, rng: np.random.Generator, attempt_seed: int, max_tries: int = 10) -> dict | None:
    for _ in range(max_tries):
        cdict = sample_container_variant(rng, ctx.base_cdict)
        size = sample_candidate_size(rng)
        orn = int(rng.integers(0, 6))
        half = get_half_ext(list(size), orn)
        target = sample_interior_target(rng, cdict, half, margin=0.06)
        if target is None:
            continue
        n_placed = int(rng.integers(0, 4))
        near_path_bias = bool(rng.random() < 0.4)
        placed_specs = []
        for k in range(n_placed):
            psize = sample_candidate_size(rng)
            phalf = tuple(v / 2.0 for v in psize)
            ptarget = sample_interior_target(rng, cdict, phalf, margin=0.03)
            if ptarget is None:
                continue
            if near_path_bias and k == 0:
                jitter = rng.normal(0.0, 0.05, size=3)
                ptarget = tuple(float(target[j] + jitter[j]) for j in range(3))
            placed_specs.append({
                "index": k + 1, "size": psize, "pos_world": ptarget, "orn_quat": (0.0, 0.0, 0.0, 1.0),
            })
        res = execute_case(cdict, 0.0, placed_specs, size, orn, target, ctx.raw_validator_config, with_spy=True)
        if res["exception"] or not res["inclusion_verdict"]:
            continue
        return build_case_record(
            category="random_scene", attempt_seed=attempt_seed, ctx=ctx, cdict=cdict, offset_x=0.0,
            candidate_size=size, candidate_orn=orn, target_world=target, placed_specs=placed_specs,
            exec_result=res, category_metadata={"n_placed": len(placed_specs), "near_path_bias": near_path_bias},
        )
    return None


GENERATORS = {
    "clear_path": generate_clear_path_case,
    "y_leg_blocked": generate_y_leg_blocked_case,
    "x_leg_blocked": generate_x_leg_blocked_case,
    "random_scene": generate_random_scene_case,
}


def generate_category(ctx: GenerationContext, category: str) -> tuple[list[dict], int]:
    if category == "safety_margin_boundary":
        return generate_safety_margin_boundary_category(ctx)
    gen_fn = GENERATORS[category]
    accepted: list[dict] = []
    attempts_used = 0
    for attempt_i in range(MAX_ATTEMPTS_PER_CATEGORY):
        attempts_used = attempt_i + 1
        seed = case_seed(category, attempt_i)
        rng = np.random.default_rng(seed)
        case = gen_fn(ctx, rng, seed)
        if case is not None:
            accepted.append(case)
            if len(accepted) >= CASES_PER_CATEGORY:
                break
    if len(accepted) < CASES_PER_CATEGORY:
        raise RuntimeError(
            f"{category}: only {len(accepted)}/{CASES_PER_CATEGORY} cases generated "
            f"within {MAX_ATTEMPTS_PER_CATEGORY} attempts"
        )
    return accepted, attempts_used


def generate_safety_margin_boundary_category(ctx: GenerationContext) -> tuple[list[dict], int]:
    per_level_budget = MAX_ATTEMPTS_PER_CATEGORY // len(BOUNDARY_DELTAS_MM)
    accepted: list[dict] = []
    attempts_used = 0
    for level_idx, delta_mm in enumerate(BOUNDARY_DELTAS_MM):
        lo = level_idx * per_level_budget
        hi = lo + per_level_budget
        collected: list[dict] = []
        for attempt_i in range(lo, hi):
            attempts_used = attempt_i + 1
            seed = case_seed("safety_margin_boundary", attempt_i)
            rng = np.random.default_rng(seed)
            case = generate_boundary_case(ctx, rng, seed, delta_mm)
            if case is not None:
                collected.append(case)
                if len(collected) >= CASES_PER_BOUNDARY_LEVEL:
                    break
        if len(collected) < CASES_PER_BOUNDARY_LEVEL:
            raise RuntimeError(
                f"safety_margin_boundary level delta={delta_mm}mm: only "
                f"{len(collected)}/{CASES_PER_BOUNDARY_LEVEL} within budget"
            )
        accepted.extend(collected)
    return accepted, attempts_used


def assign_case_ids(cases: list[dict], category: str) -> None:
    for idx, case in enumerate(cases):
        case["case_id"] = f"lpath-v1-{category}-{idx:04d}"


# ---------------------------------------------------------------------------
# spy有無の最終bool一致検証（§4.5「spyの契約」必須条件）
# ---------------------------------------------------------------------------
def check_spy_mismatch(ctx: GenerationContext, case: dict) -> bool:
    """同一ケースをspy無しで再構築・再実行し、公式最終boolがspy有りと一致するか検証する。"""
    cdict = {
        "index": case["container_raw_config"]["index"],
        "length": case["container_raw_config"]["length"],
        "width": case["container_raw_config"]["width"],
        "height": case["container_raw_config"]["height"],
        "thickness": case["container_raw_config"]["thickness"],
        "cut_x": case["container_raw_config"]["cut_x"],
        "cut_y": case["container_raw_config"]["cut_y"],
        "buffer": case["container_raw_config"]["buffer"],
        "require_shelf": case["container_raw_config"]["require_shelf"],
        "is_prioritized": False,
    }
    placed_specs = [
        {
            "index": pi["index"], "size": tuple(pi["size"]),
            "pos_world": tuple(pi["pos_world"]), "orn_quat": tuple(pi["orn_quat"]),
        }
        for pi in case["placed_items"]
    ]
    res = execute_case(
        cdict, case["container_origin_world"], placed_specs,
        tuple(case["candidate_size"]), case["candidate_orientation"],
        tuple(case["candidate_target_pos"]), ctx.raw_validator_config, with_spy=False,
    )
    if res["exception"] is not None:
        raise RuntimeError(f"no-spy replay raised for case {case['case_id']}: {res['exception']}")
    return bool(res["verdict"]) != bool(case["official_verdict"])


# ---------------------------------------------------------------------------
# schema検査・全体分布検証・多様性診断
# ---------------------------------------------------------------------------
def validate_case(case: dict) -> None:
    missing = [f for f in REQUIRED_CASE_FIELDS if f not in case]
    if missing:
        raise RuntimeError(f"case {case.get('case_id')} missing required fields: {missing}")
    if case["move_call_count"] != len(case["move_calls"]):
        raise RuntimeError(f"case {case['case_id']}: move_call_count mismatch")
    if case["move_call_count"] not in (1, 2):
        raise RuntimeError(f"case {case['case_id']}: invalid move_call_count {case['move_call_count']}")
    if case["move_calls"] and case["move_calls"][0]["leg"] != "y":
        raise RuntimeError(f"case {case['case_id']}: first move_call leg must be 'y'")
    if len(case["move_calls"]) == 2 and case["move_calls"][1]["leg"] != "x":
        raise RuntimeError(f"case {case['case_id']}: second move_call leg must be 'x'")


def verify_overall_distribution(cases: list[dict]) -> dict:
    n_pass = sum(1 for c in cases if c["official_verdict"])
    n_fail = sum(1 for c in cases if not c["official_verdict"])
    n_exc = sum(1 for c in cases if c["official_exception"]["occurred"])
    if n_exc != 0:
        raise RuntimeError(f"official_exception count {n_exc} != 0")
    if n_pass < 200:
        raise RuntimeError(f"official_pass {n_pass} < 200")
    if n_fail < 400:
        raise RuntimeError(f"official_fail {n_fail} < 400")

    for c in cases:
        if not c["official_inclusion_verdict"]:
            raise RuntimeError(f"case {c['case_id']}: official_inclusion_verdict is not True")
        mcc = c["move_call_count"]
        y_passed = c["move_calls"][0]["passed"]
        if mcc == 1 and y_passed is not False:
            raise RuntimeError(f"case {c['case_id']}: call_count=1 but Y leg not False")
        if mcc == 2 and y_passed is not True:
            raise RuntimeError(f"case {c['case_id']}: call_count=2 but Y leg not True")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)

    cp = by_cat["clear_path"]
    if len(cp) != CASES_PER_CATEGORY or not all(
        c["official_verdict"] and c["move_call_count"] == 2
        and c["move_calls"][0]["passed"] and c["move_calls"][1]["passed"] for c in cp
    ):
        raise RuntimeError("clear_path category contract violated")

    yb = by_cat["y_leg_blocked"]
    if len(yb) != CASES_PER_CATEGORY or not all(
        (not c["official_verdict"]) and c["move_call_count"] == 1
        and c["move_calls"][0]["passed"] is False for c in yb
    ):
        raise RuntimeError("y_leg_blocked category contract violated")

    xb = by_cat["x_leg_blocked"]
    if len(xb) != CASES_PER_CATEGORY or not all(
        (not c["official_verdict"]) and c["move_call_count"] == 2
        and c["move_calls"][0]["passed"] is True and c["move_calls"][1]["passed"] is False for c in xb
    ):
        raise RuntimeError("x_leg_blocked category contract violated")

    rs = by_cat["random_scene"]
    if len(rs) != CASES_PER_CATEGORY:
        raise RuntimeError("random_scene category count != 200")
    if not any(c["official_verdict"] for c in rs) or not any(not c["official_verdict"] for c in rs):
        raise RuntimeError("random_scene must contain both pass and fail")

    smb = by_cat["safety_margin_boundary"]
    if len(smb) != CASES_PER_CATEGORY:
        raise RuntimeError("safety_margin_boundary category count != 200")

    return {
        "official_pass": n_pass, "official_fail": n_fail, "official_exception": n_exc,
    }


def _scene_fingerprint(case: dict) -> str:
    payload = {
        "container_raw_config": case["container_raw_config"],
        "container_origin_world": case["container_origin_world"],
        "candidate_size": case["candidate_size"],
        "candidate_orientation": case["candidate_orientation"],
        "candidate_target_pos": case["candidate_target_pos"],
        "placed_items": case["placed_items"],
    }
    return json.dumps(payload, sort_keys=True)


def compute_diversity(cases: list[dict]) -> dict:
    fingerprints = [_scene_fingerprint(c) for c in cases]
    unique_fp = set(fingerprints)
    cand_geo = {(tuple(c["candidate_size"]), c["candidate_orientation"]) for c in cases}
    blk_geo = {tuple(pi["size"]) for c in cases for pi in c["placed_items"]}
    orn_dist = Counter(c["candidate_orientation"] for c in cases)

    sizes = [c["candidate_size"] for c in cases]
    size_min = [min(s[i] for s in sizes) for i in range(3)] if sizes else None
    size_max = [max(s[i] for s in sizes) for i in range(3)] if sizes else None

    y_lengths, x_lengths = [], []
    for c in cases:
        calls = c["move_calls"]
        if len(calls) >= 1:
            y_lengths.append(abs(calls[0]["target_pos"][1] - calls[0]["start_pos"][1]))
        if len(calls) >= 2:
            x_lengths.append(abs(calls[1]["target_pos"][0] - calls[1]["start_pos"][0]))

    return {
        "unique_scene_input_count": len(unique_fp),
        "duplicate_scene_input_count": len(fingerprints) - len(unique_fp),
        "unique_candidate_geometry_count": len(cand_geo),
        "unique_blocker_geometry_count": len(blk_geo),
        "orientation_distribution": dict(sorted(orn_dist.items())),
        "candidate_size_min": size_min,
        "candidate_size_max": size_max,
        "y_leg_length_min": min(y_lengths) if y_lengths else None,
        "y_leg_length_max": max(y_lengths) if y_lengths else None,
        "x_leg_length_min": min(x_lengths) if x_lengths else None,
        "x_leg_length_max": max(x_lengths) if x_lengths else None,
    }


def verify_diversity(diversity: dict) -> None:
    for category, diag in diversity.items():
        if diag["duplicate_scene_input_count"] != 0:
            raise RuntimeError(f"{category}: {diag['duplicate_scene_input_count']} duplicate scene inputs found")
        if diag["unique_scene_input_count"] < CASES_PER_CATEGORY:
            raise RuntimeError(
                f"{category}: only {diag['unique_scene_input_count']} unique scene inputs (<{CASES_PER_CATEGORY})"
            )


# ---------------------------------------------------------------------------
# 出力・レポート
# ---------------------------------------------------------------------------
def serialize_jsonl(cases: list[dict]) -> tuple[str, str]:
    lines = [json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for c in cases]
    content = "\n".join(lines) + "\n"
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, sha256


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def render_report(
    ctx: GenerationContext, category_reports: dict, diversity: dict, mismatch_count: int,
    jsonl_sha256: str, total_cases: int, repro_result: bool | None, elapsed_s: float,
) -> str:
    lines: list[str] = []
    lines.append("# L字経路ゴールデン生成レポート（T-016A, l_path_golden_v1）")
    lines.append("")
    lines.append("実装詳細仕様書.md §4.5「T-016Aの最終DoD」に対応するレポート。生成日時・実行時間・")
    lines.append("body id・一時ファイルパスは正規化対象外のため本文書には含めない。")
    lines.append("")
    lines.append("## 公式コード同一性")
    lines.append("")
    lines.append(f"- `official_git_head`（仕様参照HEAD）: `{ctx.official_git_head}`")
    lines.append(f"- `official_validator_git_blob`: `{ctx.official_validator_git_blob}`")
    lines.append(f"- `official_validator_file_sha256`: `{ctx.official_validator_file_sha256}`")
    lines.append(f"- `official_validator_last_commit`: `{ctx.official_validator_last_commit}`")
    lines.append(f"- `generator_repo_head`（生成時リポジトリHEAD）: `{ctx.generator_repo_head}`")
    lines.append(f"- 判定: `{ctx.official_git_head}` の validator.py と working tree の内容は byte-identical（起動時ガードで確認済み）")
    lines.append("")
    lines.append("## config")
    lines.append("")
    lines.append(f"- `config_relpath`: `{ctx.config_relpath}`")
    lines.append(f"- `config_file_sha256`: `{ctx.config_file_sha256}`")
    lines.append(f"- `config_task`: `{ctx.config_task}`")
    lines.append(f"- `raw_validator_config`: `{json.dumps(ctx.raw_validator_config, sort_keys=True)}`")
    lines.append(f"- `effective_validator_config`（公式インスタンス属性由来）: `{json.dumps(ctx.effective_validator_config, sort_keys=True)}`")
    lines.append(f"- `validator_config_hash`: `{ctx.validator_config_hash}`")
    lines.append("")
    lines.append("## カテゴリ別件数・試行数・category_seed")
    lines.append("")
    lines.append("| category | 件数 | 試行数 | category_seed |")
    lines.append("| --- | --- | --- | --- |")
    for cat in CATEGORIES:
        rep = category_reports[cat]
        lines.append(f"| {cat} | {rep['count']} | {rep['attempts_used']} | {category_seed(cat)} |")
    lines.append("")
    lines.append("## safety_margin_boundary 水準別件数")
    lines.append("")
    lines.append("| delta_mm | gap基準からの符号 | 件数 |")
    lines.append("| --- | --- | --- |")
    boundary_levels = category_reports["safety_margin_boundary"]["boundary_levels"]
    for delta_mm, n in boundary_levels:
        lines.append(f"| {delta_mm:+.1f} | {'離す' if delta_mm > 0 else '近づける'} | {n} |")
    lines.append("")
    lines.append("## 全体分布・整合性")
    lines.append("")
    dist = category_reports["_overall_distribution"]
    lines.append(f"- `official_pass`: {dist['official_pass']}（>=200）")
    lines.append(f"- `official_fail`: {dist['official_fail']}（>=400）")
    lines.append(f"- `official_exception`: {dist['official_exception']}（==0）")
    lines.append(f"- `spy_boolean_mismatch`: {mismatch_count}（==0）")
    lines.append(f"- 総ケース数: {total_cases}（==1000）")
    lines.append("")
    lines.append("## 多様性診断")
    lines.append("")
    lines.append("| category | unique_scene_input | duplicate | unique_candidate_geo | unique_blocker_geo | y_leg長 min/max | x_leg長 min/max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for cat in CATEGORIES:
        d = diversity[cat]
        ylen = f"{d['y_leg_length_min']:.3f}/{d['y_leg_length_max']:.3f}" if d["y_leg_length_min"] is not None else "n/a"
        xlen = f"{d['x_leg_length_min']:.3f}/{d['x_leg_length_max']:.3f}" if d["x_leg_length_min"] is not None else "n/a"
        lines.append(
            f"| {cat} | {d['unique_scene_input_count']} | {d['duplicate_scene_input_count']} | "
            f"{d['unique_candidate_geometry_count']} | {d['unique_blocker_geometry_count']} | {ylen} | {xlen} |"
        )
    lines.append("")
    lines.append("orientation分布（カテゴリ別）:")
    lines.append("")
    for cat in CATEGORIES:
        lines.append(f"- {cat}: `{diversity[cat]['orientation_distribution']}`")
    lines.append("")
    lines.append("## 再現性")
    lines.append("")
    if repro_result is None:
        lines.append("- `--verify-reproducibility` は指定されなかったため未実施")
    else:
        lines.append(f"- `--verify-reproducibility` により2回生成し、正規化JSONLの一致を確認: {'一致' if repro_result else '不一致'}")
    lines.append("")
    lines.append("## 正規化JSONLハッシュ")
    lines.append("")
    lines.append(f"- SHA-256: `{jsonl_sha256}`")
    lines.append(f"- 件数: {total_cases}")
    lines.append("")
    lines.append("## 生成時間")
    lines.append("")
    lines.append(f"- 約 {elapsed_s:.1f} 秒（`time.monotonic()` 計測）")
    lines.append("")
    lines.append("## 出力")
    lines.append("")
    lines.append(f"- `{GOLDEN_RELPATH}`")
    lines.append(f"- `{REPORT_RELPATH}`（本ファイル）")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sample_config.json", help="公式configへのパス（simulator/基準）")
    parser.add_argument(
        "--verify-reproducibility", action="store_true",
        help="同一プロセス内でもう一度全カテゴリ生成し、正規化JSONLの一致を検証する（出力ファイルは変更しない）",
    )
    return parser.parse_args()


def run_all_categories(ctx: GenerationContext) -> tuple[list[dict], dict]:
    all_cases: list[dict] = []
    category_reports: dict = {}
    for category in CATEGORIES:
        logger.info("generating category=%s", category)
        if category == "safety_margin_boundary":
            cases, attempts_used = generate_safety_margin_boundary_category(ctx)
            boundary_levels = [
                (delta_mm, sum(1 for c in cases if c["category_metadata"]["delta_mm"] == delta_mm))
                for delta_mm in BOUNDARY_DELTAS_MM
            ]
            category_reports[category] = {
                "count": len(cases), "attempts_used": attempts_used, "boundary_levels": boundary_levels,
            }
        else:
            cases, attempts_used = generate_category(ctx, category)
            category_reports[category] = {"count": len(cases), "attempts_used": attempts_used}
        assign_case_ids(cases, category)
        all_cases.extend(cases)
        logger.info("category=%s accepted=%d attempts_used=%d", category, len(cases), attempts_used)
    return all_cases, category_reports


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    script_path = Path(__file__).resolve()
    simulator_root = script_path.parents[1]
    repo_root = simulator_root.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    t_start = time.monotonic()
    try:
        ctx = build_generation_context(repo_root, config_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "official validator.py identity confirmed: blob=%s sha256=%s...",
        ctx.official_validator_git_blob, ctx.official_validator_file_sha256[:16],
    )

    try:
        all_cases, category_reports = run_all_categories(ctx)

        mismatch_count = 0
        for case in all_cases:
            if check_spy_mismatch(ctx, case):
                mismatch_count += 1
                logger.error("spy_boolean_mismatch for case %s", case["case_id"])
        if mismatch_count != 0:
            raise RuntimeError(f"spy_boolean_mismatch={mismatch_count} != 0")

        for case in all_cases:
            validate_case(case)

        dist = verify_overall_distribution(all_cases)
        category_reports["_overall_distribution"] = dist

        diversity = {cat: compute_diversity([c for c in all_cases if c["category"] == cat]) for cat in CATEGORIES}
        verify_diversity(diversity)

        jsonl_content, jsonl_sha256 = serialize_jsonl(all_cases)

        repro_result = None
        if args.verify_reproducibility:
            logger.info("verifying reproducibility: running a second full generation pass")
            all_cases_2, _ = run_all_categories(ctx)
            jsonl_content_2, jsonl_sha256_2 = serialize_jsonl(all_cases_2)
            repro_result = jsonl_sha256 == jsonl_sha256_2
            if not repro_result:
                raise RuntimeError(
                    "reproducibility check failed: two generation passes produced different normalized JSONL "
                    f"({jsonl_sha256} != {jsonl_sha256_2})"
                )
            logger.info("reproducibility verified: normalized JSONL matches across 2 runs")

    except RuntimeError as exc:
        logger.error("generation failed: %s", exc)
        return 1

    elapsed_s = time.monotonic() - t_start
    report_md = render_report(
        ctx, category_reports, diversity, mismatch_count, jsonl_sha256, len(all_cases), repro_result, elapsed_s,
    )

    golden_path = repo_root / GOLDEN_RELPATH
    report_path = repo_root / REPORT_RELPATH
    atomic_write_text(golden_path, jsonl_content)
    atomic_write_text(report_path, report_md)

    logger.info("done: %d cases written to %s (sha256=%s)", len(all_cases), golden_path, jsonl_sha256[:16])
    logger.info("report written to %s", report_path)
    logger.info("total elapsed: %.1fs", elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
