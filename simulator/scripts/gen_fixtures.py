"""T-018: 幾何判定フィクスチャ生成（詳細仕様書 §5.1、§4.5 masks 契約に整合）。

公式 `GroundHandlingEnv` 上でエピソードを進行させながら、各stepの状態（state snapshot）と
K=20件のランダムEMSベース候補に対する公式幾何判定（`PlacementValidator.check_inclusion`／
`check_transport_path`、isolated `BulletClient` 上で実行）を記録する。T-019 のパリティ検証
（自作 `evaluate_stage` と公式判定の FN=0 確認）が読み込む再生可能fixtureを生成する。

`official_ok` は幾何のみの短絡積（`check_inclusion` かつ `check_transport_path`）であり、
`place_item` の物理沈降は含めない（沈降を含むラベルは T-032 の責務）。エピソード進行には
公式 validator を合法手オラクルとして用いる（T-023 ヒューリスティックAgent未実装のため）。
K件のパリティ候補用RNGと進行用合法手探索RNGは分離し、進行用候補はgeo_verdicts.jsonlへ
混入させない。

公式コード（`src/ground_handling/**`）は読み取り・import・インスタンス化・実行のみ行う。
コピー・改変・整形は行わない。

実行（`simulator/` で）:
    python -m scripts.gen_fixtures --episodes 20 --k 20 --seed 0 --out datasets/fixtures/
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from src.ground_handling.containers import Container
from src.ground_handling.env import GroundHandlingEnv
from src.ground_handling.items import Item
from src.ground_handling.validator import PlacementValidator
from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM
from src.packing_core.masks import prefilter_dims
from src.packing_core.state import PackingState, build_state, make_action, rel_to_world
from src.packing_core.types import Candidate

logger = logging.getLogger("packing")

SCHEMA_VERSION = "geo-fixtures-v1"
DEFAULT_CONFIG_RELPATH = "configs/sample_config.json"
DEFAULT_OUT_RELPATH = "datasets/fixtures/"
MAX_PROGRESS_SEARCH = 200  # 進行用合法手探索の1step当たり上限候補数（計画§8）
# isolated判定専用の一時Item index。live env / stream の実item indexとは独立の名前空間で
# あり、衝突しても実害はない（fixtureへは保存しない一時オブジェクト）。
CANDIDATE_ITEM_INDEX = 999999


# ---------------------------------------------------------------------------
# 決定論的seed導出（計画§6。hash()／np.random.seed は使用しない）
# ---------------------------------------------------------------------------
def derive_seed(*parts: object) -> int:
    """SHA-256 に基づく決定論的64bit整数seedを派生する。

    Args:
        *parts: seed導出キーを構成する要素（文字列化して連結する）。

    Returns:
        SHA-256ダイジェスト先頭8byteから得た非負整数。
    """
    key = "t018:" + ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def derive_env_seed(base_seed: int, run_idx: int) -> int:
    """`env.reset(seed=...)` 用の32bit範囲seedを派生する（内部で `np.random.seed` に渡るため）。

    Args:
        base_seed: `--seed` で指定された起点seed。
        run_idx: エピソード番号（0始まり）。

    Returns:
        `[0, 2**32)` に収まる整数。
    """
    return derive_seed("env", base_seed, run_idx) % (2**32)


def compute_dataset_id(config_sha256: str, base_seed: int, episodes: int, k: int) -> str:
    """世代ディレクトリ名に使う決定論的dataset_idを算出する（計画§10）。

    Args:
        config_sha256: 使用したconfigファイル全体のSHA-256。
        base_seed: `--seed`。
        episodes: `--episodes`。
        k: `--k`。

    Returns:
        16文字の16進文字列。
    """
    key = f"{SCHEMA_VERSION}:{config_sha256}:{base_seed}:{episodes}:{k}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# JSON安全化（numpy配列・スカラーをPython標準型へ再帰変換）
# ---------------------------------------------------------------------------
def json_safe(obj: object) -> object:
    """numpy配列／スカラー／tuple／dictを再帰的にJSON安全なPython標準型へ変換する。

    Args:
        obj: 変換対象。observation/init の任意の入れ子構造を想定する。

    Returns:
        `json.dumps` でそのままシリアライズ可能な同値の構造。
    """
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def sanitize_init(init: dict) -> dict:
    """`get_init_states()` の戻り値をstate snapshot保存用に整形する（depth_map等は元々含まない）。"""
    return {
        "optimize": bool(init["optimize"]),
        "lookahead_k": int(init["lookahead_k"]),
        "container_list": json_safe(init["container_list"]),
    }


def sanitize_observation(observation: dict) -> dict:
    """observationから`depth_map`/`shm_*`を除いた再生可能な部分のみを抽出する（計画§5.1）。"""
    return {
        "optimize": bool(observation["optimize"]),
        "lookahead_k": int(observation["lookahead_k"]),
        "container_list": json_safe(observation["container_list"]),
        "pool_list": json_safe(observation["pool_list"]),
    }


# ---------------------------------------------------------------------------
# 候補レコード・集計コンテナ
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateRecord:
    """EMSベース候補1件（計画§5.2 cand schema に対応）。"""

    candidate_index: int
    item_idx: int
    container_idx: int
    ems_id: int
    ems_min_rel: tuple[float, float, float]
    ems_max_rel: tuple[float, float, float]
    orientation: int
    size: tuple[float, float, float]
    osize: tuple[float, float, float]
    pos_rel: tuple[float, float, float]


@dataclass
class Stats:
    """全エピソード集計用の可変集計コンテナ（計画§8 ログ要件）。"""

    total_states: int = 0
    total_candidates: int = 0
    official_ok_counts: Counter = field(default_factory=Counter)
    ng_type_counts: Counter = field(default_factory=Counter)
    official_exceptions: int = 0
    successful_env_steps: int = 0
    states_with_packed_items: int = 0
    episode_end_reasons: Counter = field(default_factory=Counter)
    episode_depths: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EMSベース候補生成（計画§8）
# ---------------------------------------------------------------------------
def sample_candidate(
    rng: np.random.Generator,
    state: PackingState,
    eligible: list[int],
    candidate_index: int,
    xy_mode: str,
    wall_clearance: float,
) -> CandidateRecord:
    """EMSベースの配置候補を1件サンプリングする。

    Z はEMS下端に接地（`ems.min_rel[2] + osize[2]/2 + wall_clearance`）。X/Yは
    `xy_mode="random"` なら有効中心範囲内の一様乱数、`xy_mode="corner"` ならEMS最小コーナー
    （DBLF的接地位置）。`wall_clearance` は公式 `inclusion_margin`（例: -0.005＝内側5mm必須）を
    厳密に「接地＝壁に接触」させると必ず inclusion 不合格になってしまうため、EMS境界へ
    厳密に接触させず候補が一応の「接地」位置として意味を持つよう、呼び出し側が現stepの
    `effective_validator_config["inclusion_margin"]` から導出した最小限のクリアランスを渡す
    （数値リテラルの独自ハードコードではなく、実行時に読み取った公式設定由来）。寸法がEMSに
    収まらない軸はEMS中心へ決定論的にフォールバックし、候補を破棄しない（DIMS不合格候補として
    そのまま保存できるようにする）。

    Args:
        rng: この呼び出し専用に分離されたRNG系列（K件用または進行探索用）。
        state: 現stepの `PackingState`（`build_state` 済み）。
        eligible: EMSが1件以上あるコンテナ index の一覧。
        candidate_index: 候補通し番号（進行探索専用候補は負値、geo_verdictsには書かない）。
        xy_mode: `"random"`（K件用）または `"corner"`（進行探索用、大落下回避）。
        wall_clearance: EMS境界からの最小内側オフセット [m]。

    Returns:
        サンプリングされた `CandidateRecord`。
    """
    container_idx = eligible[int(rng.integers(0, len(eligible)))]
    ems_list = state.ems[container_idx]
    ems_id = int(rng.integers(0, len(ems_list)))
    ems = ems_list[ems_id]
    item_idx = int(rng.integers(0, len(state.pool)))
    item = state.pool[item_idx]
    orientation = int(rng.integers(0, 6))
    osize = geometry.oriented_size(item.size, orientation)

    pos = np.empty(3, dtype=np.float64)
    pos[2] = ems.min_rel[2] + osize[2] / 2.0 + wall_clearance
    for axis in (0, 1):
        lo = ems.min_rel[axis] + osize[axis] / 2.0 + wall_clearance
        hi = ems.max_rel[axis] - osize[axis] / 2.0 - wall_clearance
        if lo <= hi:
            if xy_mode == "corner":
                pos[axis] = lo
            else:
                pos[axis] = float(rng.uniform(lo, hi))
        else:
            # osize（+クリアランス）がEMSに収まらない: 除外せずEMS中心へ決定論配置
            # （DIMS不合格候補として保存できるようにする）。
            pos[axis] = (ems.min_rel[axis] + ems.max_rel[axis]) / 2.0

    # make_action（§3.4）は place_pos を float32 化して live env へ渡す契約であるため、
    # isolated 判定（本関数の呼び出し元）が見る座標も同じ丸め後の値に揃える。丸めないと
    # wall_clearance が float32 の丸め誤差（~1e-7）より小さい場合に、isolated では
    # 合格・live envでは不合格という境界不一致が生じ得るため。
    pos = pos.astype(np.float32).astype(np.float64)

    return CandidateRecord(
        candidate_index=candidate_index,
        item_idx=item_idx,
        container_idx=container_idx,
        ems_id=ems_id,
        ems_min_rel=tuple(float(v) for v in ems.min_rel),
        ems_max_rel=tuple(float(v) for v in ems.max_rel),
        orientation=orientation,
        size=tuple(float(v) for v in item.size),
        osize=tuple(float(v) for v in osize),
        pos_rel=tuple(float(v) for v in pos),
    )


def candidate_to_masks_candidate(rec: CandidateRecord) -> Candidate:
    """`CandidateRecord` から `masks.prefilter_dims` 等が要求する `Candidate` を構築する。"""
    return Candidate(
        item_idx=rec.item_idx,
        container_idx=rec.container_idx,
        ems_id=rec.ems_id,
        orientation=rec.orientation,
        pos_rel=np.asarray(rec.pos_rel, dtype=np.float64),
        osize=np.asarray(rec.osize, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# isolated PyBulletシーン構築（live envとは別client。計画§7）
# ---------------------------------------------------------------------------
def build_isolated_scene(
    client: BulletClient, task_config: dict, init: dict, observation: dict,
) -> tuple[list[Container], PlacementValidator]:
    """live env とは別の isolated client 上に静的形状＋既配置荷物を再構成する。

    静的形状は `task_config`（config固定値）から、コンテナ原点Xは
    `init["container_list"][ci]["center"][0]` から取得する（`spacing` の再計算は行わない）。
    既配置荷物は `observation` の `belongs_to` に基づき対応する `container.packed_items` へ
    必ず追加する（別コンテナへの誤登録を避ける）。

    Args:
        client: isolated `BulletClient`（`p.DIRECT`）。
        task_config: `full_config[task_id]`（`configs/sample_config.json` の1タスク分）。
        init: このstep用に固定された `env.get_init_states()` の戻り値。
        observation: このstepの現在observation（`env.reset`/`env.step` の戻り値）。

    Returns:
        `(containers, validator)`。`containers` は `init["container_list"]` と同じ列挙順。
    """
    cfg_containers = task_config["containers"]["container_list"]
    containers: list[Container] = []
    for ci, cdict_cfg in enumerate(cfg_containers):
        cdict = {k: v for k, v in cdict_cfg.items() if k != "packed_items"}
        offset_x = float(init["container_list"][ci]["center"][0])
        container = Container(offset_x=offset_x, packed_items=[], **cdict)
        with contextlib.redirect_stdout(io.StringIO()):
            container.create(client)
        containers.append(container)

    for obs_cdict in observation["container_list"]:
        for raw_item in obs_cdict["packed_items"]:
            belongs_to = int(raw_item["belongs_to"])
            item = Item(
                index=int(raw_item["index"]),
                length=float(raw_item["length"]),
                width=float(raw_item["width"]),
                height=float(raw_item["height"]),
                mass=float(raw_item["mass"]),
                is_prioritized=bool(raw_item["is_prioritized"]),
                is_soft=bool(raw_item["is_soft"]),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                item.spawn(
                    client, initial_pos=tuple(raw_item["pos"]), initial_orn=tuple(raw_item["orn"])
                )
            containers[belongs_to].packed_items.append(item)

    validator = PlacementValidator(client, task_config["validator"], render_mode=None)
    return containers, validator


def capture_fingerprint(client: BulletClient) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
    """isolated シーンの body 構成・base pose の fingerprint を採取する（状態不変性確認用）。"""
    fingerprint: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for i in range(client.getNumBodies()):
        body_id = client.getBodyUniqueId(i)
        pos, orn = client.getBasePositionAndOrientation(body_id)
        fingerprint[body_id] = (tuple(float(v) for v in pos), tuple(float(v) for v in orn))
    return fingerprint


def fingerprints_match(
    fp_a: dict[int, tuple[tuple[float, ...], tuple[float, ...]]],
    fp_b: dict[int, tuple[tuple[float, ...], tuple[float, ...]]],
    tol: float = 1e-6,
) -> bool:
    """2つの fingerprint が body 集合・base pose ともに一致するかを判定する。"""
    if set(fp_a) != set(fp_b):
        return False
    for body_id, (pos_a, orn_a) in fp_a.items():
        pos_b, orn_b = fp_b[body_id]
        if not np.allclose(pos_a, pos_b, atol=tol) or not np.allclose(orn_a, orn_b, atol=tol):
            return False
    return True


def judge_candidate(
    client: BulletClient,
    containers: list[Container],
    validator: PlacementValidator,
    rec: CandidateRecord,
    baseline_id: int,
    baseline_fp: dict[int, tuple[tuple[float, ...], tuple[float, ...]]],
) -> tuple[bool, str, dict | None]:
    """isolated client 上で1候補を公式幾何判定する（`check_inclusion` → `check_transport_path`）。

    判定後は候補bodyを除去し baseline へ復元したうえで fingerprint 一致を確認する。座標・
    状態復元・validator設定の一致契約違反（fingerprint不一致）は例外として送出する。

    Args:
        client: isolated `BulletClient`（live envとは別）。
        containers: `build_isolated_scene` が返した `Container` 一覧。
        validator: isolated シーンに束縛された `PlacementValidator`。
        rec: 判定対象候補。
        baseline_id: 候補判定前に採取した `saveState` id。
        baseline_fp: `baseline_id` 直後の body fingerprint。

    Returns:
        `(official_ok, ng_type, exception_info)`。`ng_type` は
        `""`/`"inclusion"`/`"path"`/`"error"`（`"error"` は最終fixtureへは残さない）。
        `exception_info` は例外発生時のみ `{"type","message"}`、それ以外は `None`。

    Raises:
        RuntimeError: 判定後に PyBullet 状態が baseline と一致しない場合。
    """
    container = containers[rec.container_idx]
    target_world = tuple(
        float(v) for v in rel_to_world(np.asarray(rec.pos_rel, dtype=np.float64), container)
    )
    item = Item(
        index=CANDIDATE_ITEM_INDEX,
        length=rec.size[0],
        width=rec.size[1],
        height=rec.size[2],
        mass=1.0,
        is_prioritized=False,
        is_soft=False,
    )

    official_ok = False
    ng_type = "inclusion"
    exc_info: dict | None = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            included = bool(validator.check_inclusion(container, item, target_world, rec.orientation))
        if not included:
            ng_type = "inclusion"
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                path_ok = bool(
                    validator.check_transport_path(container, item, target_world, rec.orientation)
                )
            if path_ok:
                official_ok, ng_type = True, ""
            else:
                ng_type = "path"
    except Exception as exc:  # noqa: BLE001 - 件数・型を集計し呼び出し側で非ゼロ終了させるため広く捕捉
        exc_info = {"type": type(exc).__name__, "message": str(exc)}
        official_ok, ng_type = False, "error"
    finally:
        if item.pybullet_id is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                item.remove(client=client)
        client.restoreState(stateId=baseline_id)
        fp_after = capture_fingerprint(client)
        if not fingerprints_match(baseline_fp, fp_after):
            raise RuntimeError(
                "PyBullet isolated state fingerprint mismatch after candidate judge "
                f"(container_idx={rec.container_idx}, candidate_index={rec.candidate_index})"
            )

    return official_ok, ng_type, exc_info


# ---------------------------------------------------------------------------
# state snapshot / geo_verdicts 書き出し（staging先へ）
# ---------------------------------------------------------------------------
def write_state_snapshot(
    *,
    staging_states_dir: Path,
    dataset_id: str,
    run_idx: int,
    step_idx: int,
    task_id: str,
    base_seed: int,
    env_seed: int,
    config_relpath: str,
    config_sha256: str,
    effective_validator_cfg: dict,
    init: dict,
    obs: dict,
) -> str:
    """1step分のstate snapshotをstagingへ書き出し、`state_ref`（相対パス）を返す。

    Args:
        staging_states_dir: `<staging>/states/t018_<dataset_id>/`。
        dataset_id: 決定論的世代ID。
        run_idx: エピソード番号。
        step_idx: エピソード内step番号（0始まり）。
        task_id: `configs/sample_config.json` のタスクキー。
        base_seed: `--seed`。
        env_seed: このエピソードの `env.reset(seed=...)` に使ったseed。
        config_relpath: リポジトリルート基準（不可なら絶対）のconfigパス。
        config_sha256: configファイル全体のSHA-256。
        effective_validator_cfg: isolated `PlacementValidator` インスタンス属性由来の実効設定。
        init: `env.get_init_states()` の戻り値。
        obs: `env.reset`/`env.step` が返した現在observation。

    Returns:
        `geo_verdicts.jsonl` のディレクトリを基準とした相対パス（例:
        `states/t018_<dataset_id>/00_0000.json`）。
    """
    payload = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "run": run_idx,
            "step": step_idx,
            "task_id": task_id,
            "base_seed": base_seed,
            "episode_env_seed": env_seed,
            "source_config": config_relpath,
            "source_config_sha256": config_sha256,
            "effective_validator_config": effective_validator_cfg,
        },
        "init": sanitize_init(init),
        "observation": sanitize_observation(obs),
    }
    fname = f"{run_idx:02d}_{step_idx:04d}.json"
    (staging_states_dir / fname).write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    gen_name = staging_states_dir.name
    return f"states/{gen_name}/{fname}"


def write_verdict(
    verdicts_f: IO[str], state_ref: str, rec: CandidateRecord, official_ok: bool, ng_type: str
) -> None:
    """1候補の判定結果をgeo_verdicts staging JSONLへ1行追記する。"""
    payload = {
        "state_ref": state_ref,
        "cand": {
            "candidate_index": rec.candidate_index,
            "item_idx": rec.item_idx,
            "container_idx": rec.container_idx,
            "ems_id": rec.ems_id,
            "ems_min_rel": list(rec.ems_min_rel),
            "ems_max_rel": list(rec.ems_max_rel),
            "orientation": rec.orientation,
            "size": list(rec.size),
            "osize": list(rec.osize),
            "pos_rel": list(rec.pos_rel),
        },
        "official_ok": bool(official_ok),
        "ng_type": ng_type,
    }
    verdicts_f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# エピソード進行（計画§4/§8）
# ---------------------------------------------------------------------------
def run_episode(
    *,
    run_idx: int,
    task_id: str,
    task_config: dict,
    base_seed: int,
    k: int,
    config_relpath: str,
    config_sha256: str,
    dataset_id: str,
    staging_states_dir: Path,
    verdicts_f: IO[str],
    stats: Stats,
) -> None:
    """1エピソードを進行させ、各stepのstate/K件candidateをstagingへ書き出す。

    観測は `env.reset()`/`env.step()` の戻り値のみを使用し、`env._get_obs()` を重複呼出し
    しない。`env` は呼び出し側の責務として必ず `finally` で close する。候補調査は live env
    とは別の isolated `BulletClient` 上で行い、live env の PyBullet 状態を変化させない。

    Args:
        run_idx: エピソード番号（0始まり）。
        task_id: `configs/sample_config.json` のタスクキー。
        task_config: `full_config[task_id]`。
        base_seed: `--seed`。
        k: 各stepで記録するランダム候補数。
        config_relpath: state snapshot へ記録するconfigの相対パス。
        config_sha256: configファイル全体のSHA-256。
        dataset_id: 決定論的世代ID。
        staging_states_dir: state snapshot の出力先（staging）。
        verdicts_f: geo_verdicts staging JSONLの書き込みハンドル（呼び出し元が開閉管理）。
        stats: 全エピソード共通の集計コンテナ（このエピソード分を加算する）。

    Raises:
        RuntimeError: isolated/live幾何の不一致、公式判定例外、PyBullet状態不変性違反、
            その他契約違反を検出した場合。
    """
    env_seed = derive_env_seed(base_seed, run_idx)
    env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
    end_reason = "unknown"
    states_recorded = 0
    try:
        env.reset_settings()
        init = env.get_init_states()
        env.reset_item_stream()
        obs, _reset_info = env.reset(seed=env_seed)

        max_steps = env.num_total_items + 2
        step_idx = 0

        while True:
            if step_idx >= max_steps:
                end_reason = "max_steps"
                break

            state = build_state(obs, init)
            if len(state.pool) == 0:
                raise RuntimeError(
                    f"unexpected empty pool while episode not terminated "
                    f"(run={run_idx}, step={step_idx})"
                )
            eligible = [ci for ci in range(len(state.containers)) if len(state.ems[ci]) > 0]
            if not eligible:
                end_reason = "no_ems"
                break

            progress_action: CandidateRecord | None = None
            state_ref = ""
            client = BulletClient(connection_mode=p.DIRECT)
            try:
                containers, validator = build_isolated_scene(client, task_config, init, obs)
                baseline_id = client.saveState()
                baseline_fp = capture_fingerprint(client)
                effective_cfg = {
                    "safety_margin": float(validator.safety_margin),
                    "start_margin": float(validator.start_margin),
                    "inclusion_margin": float(validator.inclusion_margin),
                    "start_z": float(validator.start_z),
                    "ceiling_margin": float(validator.ceiling_margin),
                }

                state_ref = write_state_snapshot(
                    staging_states_dir=staging_states_dir,
                    dataset_id=dataset_id,
                    run_idx=run_idx,
                    step_idx=step_idx,
                    task_id=task_id,
                    base_seed=base_seed,
                    env_seed=env_seed,
                    config_relpath=config_relpath,
                    config_sha256=config_sha256,
                    effective_validator_cfg=effective_cfg,
                    init=init,
                    obs=obs,
                )
                states_recorded += 1
                stats.total_states += 1
                if any(len(c["packed_items"]) > 0 for c in obs["container_list"]):
                    stats.states_with_packed_items += 1

                # EMS境界へ厳密に接触させると常に inclusion 不合格になるため、公式
                # inclusion_margin（このstepの実効値）から導出した最小クリアランスを与える
                # （sample_candidate のdocstring参照）。inclusion_margin が正（緩和）の場合は
                # クリアランス不要（0）とし、負（厳格）の場合のみ内側へその絶対量だけ寄せる。
                wall_clearance = max(0.0, -float(effective_cfg["inclusion_margin"])) + EPS_GEOM

                # --- K件（パリティ用、kcand_rng専用）---
                kcand_rng = np.random.default_rng(derive_seed("kcand", base_seed, run_idx, step_idx))
                k_records = [
                    sample_candidate(
                        kcand_rng, state, eligible, candidate_index=idx, xy_mode="random",
                        wall_clearance=wall_clearance,
                    )
                    for idx in range(k)
                ]

                for rec in k_records:
                    ok, ng_type, exc = judge_candidate(
                        client, containers, validator, rec, baseline_id, baseline_fp
                    )
                    if exc is not None:
                        stats.official_exceptions += 1
                        logger.error(
                            "official validator exception (K-candidate): state_ref=%s "
                            "candidate_index=%d type=%s message=%s",
                            state_ref, rec.candidate_index, exc["type"], exc["message"],
                        )
                        raise RuntimeError(
                            "official validator exception during K-candidate judging "
                            f"(state_ref={state_ref}, candidate_index={rec.candidate_index})"
                        )
                    write_verdict(verdicts_f, state_ref, rec, ok, ng_type)
                    stats.total_candidates += 1
                    stats.official_ok_counts[ok] += 1
                    stats.ng_type_counts[ng_type] += 1

                    if progress_action is None and ok:
                        ems = state.ems[rec.container_idx][rec.ems_id]
                        if prefilter_dims(candidate_to_masks_candidate(rec), ems):
                            progress_action = rec

                # --- 進行用合法手探索（K件に無かった場合のみ、progress_rng専用）---
                if progress_action is None:
                    progress_rng = np.random.default_rng(
                        derive_seed("progress", base_seed, run_idx, step_idx)
                    )
                    for search_i in range(MAX_PROGRESS_SEARCH):
                        rec = sample_candidate(
                            progress_rng, state, eligible,
                            candidate_index=-(search_i + 1), xy_mode="corner",
                            wall_clearance=wall_clearance,
                        )
                        ems = state.ems[rec.container_idx][rec.ems_id]
                        if not prefilter_dims(candidate_to_masks_candidate(rec), ems):
                            continue
                        ok, ng_type, exc = judge_candidate(
                            client, containers, validator, rec, baseline_id, baseline_fp
                        )
                        if exc is not None:
                            stats.official_exceptions += 1
                            logger.error(
                                "official validator exception (progress search): state_ref=%s "
                                "search_i=%d type=%s message=%s",
                                state_ref, search_i, exc["type"], exc["message"],
                            )
                            raise RuntimeError(
                                "official validator exception during progress search "
                                f"(state_ref={state_ref}, search_i={search_i})"
                            )
                        if ok:
                            progress_action = rec
                            break
            finally:
                if "baseline_id" in locals():
                    client.removeState(baseline_id)
                client.disconnect()

            if progress_action is None:
                end_reason = "no_legal_move"
                break

            action = make_action(
                progress_action.item_idx,
                progress_action.container_idx,
                np.asarray(progress_action.pos_rel, dtype=np.float64),
                progress_action.orientation,
            )
            next_obs, _reward, terminated, truncated, info = env.step(action)
            status = info.get("status", {})
            if "is_included" not in status:
                raise RuntimeError(
                    f"env.step rejected action as structurally invalid: status={status} action={action}"
                )
            is_included = status["is_included"]
            is_valid = status["is_valid"]
            is_placed_safe = status["is_placed_safe"]

            if is_included is False or is_valid is False:
                raise RuntimeError(
                    "isolated/live geometry mismatch: isolated official_ok=True for progress "
                    f"candidate but live env is_included={is_included} is_valid={is_valid} "
                    f"(run={run_idx}, step={step_idx}, state_ref={state_ref})"
                )

            if is_placed_safe:
                stats.successful_env_steps += 1
                if terminated or truncated:
                    if not env.stream_manager.is_empty():
                        raise RuntimeError(
                            "unexpected termination after successful placement without pool "
                            f"exhaustion (run={run_idx}, step={step_idx})"
                        )
                    end_reason = "pool_empty"
                    break
                obs = next_obs
                step_idx += 1
                continue

            end_reason = "settle_failed"
            break
    finally:
        env.close()

    stats.episode_end_reasons[end_reason] += 1
    stats.episode_depths.append(states_recorded)
    logger.info(
        "episode %d done: task_id=%s states=%d end_reason=%s",
        run_idx, task_id, states_recorded, end_reason,
    )


# ---------------------------------------------------------------------------
# stagingセルフチェック（計画§11）
# ---------------------------------------------------------------------------
def self_check(
    staging_dir: Path, staging_states_dir: Path, verdicts_path: Path, expected_k: int, stats: Stats,
) -> None:
    """staging出力を再読込し、publish前の整合性を検証する。

    Args:
        staging_dir: staging ルート（`state_ref` の解決基準）。
        staging_states_dir: `<staging>/states/t018_<dataset_id>/`。
        verdicts_path: staging の `geo_verdicts.jsonl`。
        expected_k: `--k`（総候補数の期待値算出に使用）。
        stats: 生成中に集計したカウンタ（例外件数・成功step数の事前チェックに使用）。

    Raises:
        RuntimeError: いずれかの整合性条件を満たさない場合。
    """
    if stats.official_exceptions > 0:
        raise RuntimeError(f"official validator exceptions occurred: {stats.official_exceptions}")
    if stats.successful_env_steps == 0:
        raise RuntimeError("successful_env_steps == 0 across all episodes; fixture insufficient")

    lines = verdicts_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("geo_verdicts staging file is empty")

    verdicts = [json.loads(line) for line in lines]
    required_verdict_keys = {"state_ref", "cand", "official_ok", "ng_type"}
    required_cand_keys = {
        "candidate_index", "item_idx", "container_idx", "ems_id",
        "ems_min_rel", "ems_max_rel", "orientation", "size", "osize", "pos_rel",
    }
    allowed_ng_types = {"", "inclusion", "path"}

    by_state_ref: dict[str, list[dict]] = {}
    for verdict in verdicts:
        missing = required_verdict_keys - verdict.keys()
        if missing:
            raise RuntimeError(f"verdict missing keys {missing}: {verdict}")
        if not isinstance(verdict["official_ok"], bool):
            raise RuntimeError(f"official_ok not bool: {verdict}")
        if verdict["ng_type"] not in allowed_ng_types:
            raise RuntimeError(f"unexpected ng_type in final fixture: {verdict}")
        cand = verdict["cand"]
        missing_c = required_cand_keys - cand.keys()
        if missing_c:
            raise RuntimeError(f"cand missing keys {missing_c}: {cand}")
        by_state_ref.setdefault(verdict["state_ref"], []).append(verdict)

    states_with_packed_items = 0
    for state_ref, group in by_state_ref.items():
        state_path = staging_dir / state_ref
        if not state_path.is_file():
            raise RuntimeError(f"state_ref does not resolve to a file: {state_ref}")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        init = payload["init"]
        observation = payload["observation"]
        if any(len(c["packed_items"]) > 0 for c in observation["container_list"]):
            states_with_packed_items += 1

        state = build_state(observation, init)
        for verdict in group:
            cand = verdict["cand"]
            ci = cand["container_idx"]
            if not (0 <= ci < len(state.containers)):
                raise RuntimeError(f"container_idx out of range: {cand}")
            ems_list = state.ems[ci]
            ems_id = cand["ems_id"]
            if not (0 <= ems_id < len(ems_list)):
                raise RuntimeError(f"ems_id out of range: {cand}")
            ems = ems_list[ems_id]
            if not (
                np.allclose(ems.min_rel, cand["ems_min_rel"], atol=1e-9)
                and np.allclose(ems.max_rel, cand["ems_max_rel"], atol=1e-9)
            ):
                raise RuntimeError(f"ems geometry mismatch on rebuild: {cand}")
            item_idx = cand["item_idx"]
            if not (0 <= item_idx < len(state.pool)):
                raise RuntimeError(f"item_idx out of range: {cand}")
            if state.pool[item_idx].idx != item_idx:
                raise RuntimeError(
                    f"pool idx mismatch: pool[{item_idx}].idx={state.pool[item_idx].idx}"
                )
            # Candidate 再構築可能性の確認（構築が例外を投げなければOK）
            Candidate(
                item_idx=item_idx, container_idx=ci, ems_id=ems_id,
                orientation=cand["orientation"],
                pos_rel=np.asarray(cand["pos_rel"], dtype=np.float64),
                osize=np.asarray(cand["osize"], dtype=np.float64),
            )

    if states_with_packed_items == 0:
        raise RuntimeError("states_with_packed_items == 0; fixture insufficient")

    total_candidates = len(verdicts)
    n_states = len(by_state_ref)
    if total_candidates != expected_k * n_states:
        raise RuntimeError(
            f"candidate count mismatch: total={total_candidates}, expected={expected_k}*{n_states}"
        )
    state_files = sorted(staging_states_dir.glob("*.json"))
    if len(state_files) != n_states:
        raise RuntimeError(f"state file count {len(state_files)} != referenced states {n_states}")

    logger.info(
        "self-check passed: states=%d candidates=%d states_with_packed_items=%d",
        n_states, total_candidates, states_with_packed_items,
    )


# ---------------------------------------------------------------------------
# atomic publish（計画§10）
# ---------------------------------------------------------------------------
def _replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    """ディレクトリを原子的に置換する。既存の同名ディレクトリがあれば退避してから置換する。"""
    try:
        os.replace(tmp_dir, final_dir)
    except OSError:
        stale = final_dir.with_name(final_dir.name + ".stale")
        if stale.exists():
            shutil.rmtree(stale)
        if final_dir.exists():
            os.replace(final_dir, stale)
        os.replace(tmp_dir, final_dir)
        if stale.exists():
            shutil.rmtree(stale)


def publish(out_root: Path, staging_dir: Path, dataset_id: str) -> None:
    """staging出力を正式パスへ公開する（新世代dir公開 → geo_verdicts.jsonl置換 → 旧世代削除の順）。

    Args:
        out_root: `--out`（例: `datasets/fixtures/`）。
        staging_dir: 全件生成・self-check済みのstagingルート。
        dataset_id: このdataset_idの世代を `states/t018_<dataset_id>/` として公開する。
    """
    states_root = out_root / "states"
    states_root.mkdir(parents=True, exist_ok=True)
    gen_name = f"t018_{dataset_id}"
    staged_gen_dir = staging_dir / "states" / gen_name
    final_gen_dir = states_root / gen_name

    tmp_gen_dir = states_root / f"{gen_name}.incoming"
    if tmp_gen_dir.exists():
        shutil.rmtree(tmp_gen_dir)
    shutil.move(str(staged_gen_dir), str(tmp_gen_dir))
    _replace_dir(tmp_gen_dir, final_gen_dir)

    staged_verdicts = staging_dir / "geo_verdicts.jsonl"
    final_verdicts_tmp = out_root / "geo_verdicts.jsonl.tmp"
    shutil.copyfile(staged_verdicts, final_verdicts_tmp)
    os.replace(final_verdicts_tmp, out_root / "geo_verdicts.jsonl")

    # 旧世代の削除は新geo_verdicts公開成功後に行う。削除失敗は整合性エラーにしない。
    for old_dir in states_root.glob("t018_*"):
        if old_dir.name == gen_name:
            continue
        try:
            shutil.rmtree(old_dir)
        except OSError:
            logger.warning("failed to remove stale generation dir %s (non-fatal)", old_dir)


def log_summary(stats: Stats, dataset_id: str, elapsed_s: float) -> None:
    """DoD要求のログ集計（生成件数・NG種別分布等）を出力する。"""
    logger.info("=== gen_fixtures summary (dataset_id=%s) ===", dataset_id)
    logger.info("total_episodes=%d", len(stats.episode_depths))
    logger.info("total_states=%d total_candidates=%d", stats.total_states, stats.total_candidates)
    logger.info(
        "official_ok: True=%d False=%d",
        stats.official_ok_counts.get(True, 0), stats.official_ok_counts.get(False, 0),
    )
    logger.info("ng_type distribution: %s", dict(sorted(stats.ng_type_counts.items())))
    logger.info("official_exceptions=%d", stats.official_exceptions)
    logger.info("successful_env_steps=%d", stats.successful_env_steps)
    logger.info("states_with_packed_items=%d", stats.states_with_packed_items)
    logger.info("episode_end_reasons: %s", dict(sorted(stats.episode_end_reasons.items())))
    if stats.episode_depths:
        logger.info("max_episode_depth=%d", max(stats.episode_depths))
        logger.info(
            "episode_depth_distribution: %s", dict(sorted(Counter(stats.episode_depths).items()))
        )
    logger.info("elapsed=%.1fs", elapsed_s)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20, help="生成するエピソード数（既定20）")
    parser.add_argument("--k", type=int, default=20, help="各stepで生成するランダム候補数（既定20）")
    parser.add_argument("--seed", type=int, default=0, help="決定論的生成の起点seed（既定0）")
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT_RELPATH,
        help=f"出力先ディレクトリ（既定 {DEFAULT_OUT_RELPATH}）",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_RELPATH,
        help=f"公式configへのパス（既定 {DEFAULT_CONFIG_RELPATH}、simulator/基準）",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.episodes < 1 or args.k < 1:
        logger.error("--episodes and --k must be >= 1 (episodes=%d, k=%d)", args.episodes, args.k)
        return 1

    script_path = Path(__file__).resolve()
    simulator_root = script_path.parents[1]
    repo_root = simulator_root.parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()

    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    full_config = json.loads(config_bytes)
    task_ids = list(full_config.keys())
    if not task_ids:
        logger.error("config has no tasks: %s", config_path)
        return 1

    try:
        config_relpath = str(config_path.resolve().relative_to(repo_root))
    except ValueError:
        config_relpath = str(config_path)

    dataset_id = compute_dataset_id(config_sha256, args.seed, args.episodes, args.k)
    logger.info(
        "dataset_id=%s config=%s config_sha256=%s...", dataset_id, config_relpath, config_sha256[:16]
    )

    t_start = time.monotonic()
    staging_dir = Path(tempfile.mkdtemp(prefix="t018_staging_"))
    staging_states_dir = staging_dir / "states" / f"t018_{dataset_id}"
    staging_states_dir.mkdir(parents=True)
    staging_verdicts_path = staging_dir / "geo_verdicts.jsonl"

    stats = Stats()
    try:
        with staging_verdicts_path.open("w", encoding="utf-8") as verdicts_f:
            for run_idx in range(args.episodes):
                task_id = task_ids[run_idx % len(task_ids)]
                logger.info("episode %d/%d task_id=%s", run_idx + 1, args.episodes, task_id)
                run_episode(
                    run_idx=run_idx,
                    task_id=task_id,
                    task_config=full_config[task_id],
                    base_seed=args.seed,
                    k=args.k,
                    config_relpath=config_relpath,
                    config_sha256=config_sha256,
                    dataset_id=dataset_id,
                    staging_states_dir=staging_states_dir,
                    verdicts_f=verdicts_f,
                    stats=stats,
                )
    except Exception:
        logger.exception("generation failed; discarding staging output")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 1

    try:
        self_check(
            staging_dir, staging_states_dir, staging_verdicts_path, expected_k=args.k, stats=stats
        )
    except Exception:
        logger.exception("self-check failed; discarding staging output")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 1

    try:
        publish(out_root, staging_dir, dataset_id)
    except Exception:
        logger.exception("publish failed")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 1

    shutil.rmtree(staging_dir, ignore_errors=True)

    elapsed_s = time.monotonic() - t_start
    log_summary(stats, dataset_id, elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
