"""観測→内部状態のステートレス再構築（詳細仕様書 §4.4、T-012〜T-013）。

state.py は毎ステップ observation から `PackingState` をゼロから組み立てる。世界↔相対座標変換
（`world_to_rel`/`rel_to_world`）と行動生成（`make_action`）の唯一の置き場所でもある（§3.1/§3.4）。

コンテナ index は `container_list` の列挙位置（`enumerate`）として扱う。`cdict["index"]` が
公式キーとして存在することを前提とせず参照しない（人間承認、2026-07-18）。`init` は
`optimize`/`lookahead_k` と静的なコンテナ形状（`container_list` の形状・center 部分）の
取得元、`observation` は現在の `packed_items` と `pool_list` の取得元として役割を分離する
（同セッションでの人間確定事項）。`init`/`observation` の `container_list` は同一台数・
同一列挙順を前提とし、仕様外の fallback・推測処理は行わない。
"""
from dataclasses import dataclass

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import GridParams, StageParams
from src.packing_core.container_space import ContainerSpace, bake_placed, build_container_space
from src.packing_core.ems import generate_ems, select_topn
from src.packing_core.types import EMSBox, ItemSpec, PlacedItem, Vec3


@dataclass
class PackingState:
    """observation/init から再構築した現在の内部状態（詳細仕様書 §4.4）。

    Attributes:
        containers: コンテナ index（`container_list` の列挙位置）順の `ContainerSpace` 一覧。
        placed: `container_idx` → 既配置荷物一覧。全コンテナ index をキーに持つ（空でも `[]`）。
        pool: プール内荷物の `ItemSpec` 一覧（`idx` はプール内列挙位置、仮定 A11）。
        ems: `container_idx` → `select_topn` 済み EMS 一覧。全コンテナ index をキーに持つ。
        ems_truncation: `container_idx` → `select_topn` が返した打切り率。
        meta: `optimize`/`lookahead_k` を含む付帯情報（`spacing` は含めない、§I-8）。
    """

    containers: list[ContainerSpace]
    placed: dict[int, list[PlacedItem]]
    pool: list[ItemSpec]
    ems: dict[int, list[EMSBox]]
    ems_truncation: dict[int, float]
    meta: dict


def world_to_rel(pos_world: Vec3, space: ContainerSpace) -> Vec3:
    """世界座標をコンテナ相対座標へ変換する（§3.1: X成分のみ `offset_x` を減算）。

    Args:
        pos_world: 世界座標。shape (3,), float64。
        space: 対象コンテナの `ContainerSpace`。

    Returns:
        コンテナ相対座標。shape (3,), float64。
    """
    origin_world = np.array([space.offset_x, 0.0, 0.0], dtype=np.float64)
    return np.asarray(pos_world, dtype=np.float64) - origin_world


def rel_to_world(pos_rel: Vec3, space: ContainerSpace) -> Vec3:
    """コンテナ相対座標を世界座標へ変換する（§3.1: X成分のみ `offset_x` を加算）。

    Args:
        pos_rel: コンテナ相対座標。shape (3,), float64。
        space: 対象コンテナの `ContainerSpace`。

    Returns:
        世界座標。shape (3,), float64。
    """
    origin_world = np.array([space.offset_x, 0.0, 0.0], dtype=np.float64)
    return np.asarray(pos_rel, dtype=np.float64) + origin_world


def make_action(item_idx: int, container_idx: int, pos_rel: Vec3, orientation: int) -> dict:
    """`policy()` の戻り値を生成する唯一の関数（§3.4）。

    Args:
        item_idx: プール内 index（仮定 A11）。
        container_idx: 配置先コンテナ index。
        pos_rel: 配置中心（コンテナ相対）。shape (3,), float64。
        orientation: §3.2 の orientation コード（0..5）。

    Returns:
        `item_idx`/`container_idx`/`place_pos`/`orientation` の4キーのみを持つ辞書。
        `place_pos` は `np.float32`、他は `int`（`np.int64` 等を `int()` で変換）。
    """
    return {
        "item_idx": int(item_idx),
        "container_idx": int(container_idx),
        "place_pos": np.asarray(pos_rel, dtype=np.float32),
        "orientation": int(orientation),
    }


def _packed_item_to_placed(raw_item: dict, space: ContainerSpace) -> PlacedItem:
    """`packed_items` の1要素を `PlacedItem` へ変換する（§4.4手順3）。

    `pos_world`→`pos_rel`、`rotated_aabb(quat)` で回転後AABBを算出しコンテナ相対へ変換する
    （仮定 A9: quat は `(x,y,z,w)`。仮定 A10: `pos` は幾何中心、T-012 時点の作業仮定として使用）。

    Args:
        raw_item: `packed_items` 要素（§E item キー）。
        space: 所属先コンテナの `ContainerSpace`。

    Returns:
        変換済みの `PlacedItem`。
    """
    # np.array(...) は既定でコピーする（呼び出し元の配列とエイリアスさせず、observation/init
    # を非改変に保つため）。
    pos_world = np.array(raw_item["pos"], dtype=np.float64)
    orn_quat = np.array(raw_item["orn"], dtype=np.float64)
    size = np.array(
        [raw_item["length"], raw_item["width"], raw_item["height"]], dtype=np.float64
    )

    world_min, world_max = geometry.rotated_aabb(pos_world, size, orn_quat)

    return PlacedItem(
        pos_world=pos_world,
        orn_quat=orn_quat,
        size=size,
        weight=float(raw_item["mass"]),
        is_soft=bool(raw_item["is_soft"]),
        is_priority=bool(raw_item["is_prioritized"]),
        aabb_min_rel=world_to_rel(world_min, space),
        aabb_max_rel=world_to_rel(world_max, space),
    )


def build_state(observation: dict, init: dict) -> PackingState:
    """observation/init から `PackingState` をゼロから組み立てる（§4.4）。

    コンテナ index は `container_list` の列挙位置として扱う（`cdict["index"]` は参照しない）。
    `init` はコンテナ形状と `optimize`/`lookahead_k` の取得元、`observation` は現在の
    `packed_items` と `pool_list` の取得元として役割を分離する（`init`/`observation` の
    `container_list` は同一台数・同一列挙順を前提とする）。

    処理順序（この順序で実装・コメントに番号を残す）：
        1) meta を init から取得する（`optimize`/`lookahead_k` のみ。`spacing` は含めない）。
        2) 各コンテナを `build_container_space(cdict, container_idx, GridParams().cell)` で
           構築する（`container_idx` は `init["container_list"]` の列挙位置。`offset_x` は
           `build_container_space` が `cdict["center"][0]` から取得する）。
        3) `observation` 側の `packed_items` を `belongs_to` で列挙 index へ振り分け、
           `PlacedItem` 化する（`pos_world`→`pos_rel`、`rotated_aabb` で回転後AABBを算出）。
        4) 各コンテナについて `bake_placed` で `height` を全再構築する。
        5) 各コンテナについて `generate_ems`（全再構築）→
           `select_topn(ems_list, n=StageParams().ems_top_n_per_container)` を実行し、
           選択後EMSと打切り率を `container_idx` キーで格納する。
        6) `observation["pool_list"]` を `ItemSpec` 化する（`idx` はプール内列挙位置、
           仮定 A11。`kind` は observation に相当キーが無いため常に `None`）。

    Args:
        observation: `policy()` へ渡る観測（§E: `optimize`/`lookahead_k`/`depth_map`/
            `container_list`/`pool_list`）。現在の `packed_items` と `pool_list` の取得元。
        init: `get_init_states()` の戻り値（§E: `optimize`/`lookahead_k`/`container_list`）。
            コンテナ形状と `optimize`/`lookahead_k` の取得元。

    Returns:
        再構築された `PackingState`。`placed`/`ems`/`ems_truncation` は全コンテナ index を
        キーに持つ（空コンテナでも `placed[idx] == []`、`ems[idx]` は空リストで代用しない）。
    """
    # 1) meta（spacing は agent I/F に存在しないため含めない、§I-8）。
    meta = {
        "optimize": init["optimize"],
        "lookahead_k": init["lookahead_k"],
    }

    # 2) コンテナ形状は init 側の container_list から、列挙位置を index として構築する。
    cell = GridParams().cell
    containers: list[ContainerSpace] = [
        build_container_space(cdict, container_idx, cell)
        for container_idx, cdict in enumerate(init["container_list"])
    ]
    n_containers = len(containers)

    # 3) 現在の packed_items は observation 側の container_list から取得し、belongs_to で
    #    振り分ける（位置からの推測はしない。belongs_to を正とする）。
    placed: dict[int, list[PlacedItem]] = {i: [] for i in range(n_containers)}
    for obs_cdict in observation["container_list"]:
        for raw_item in obs_cdict["packed_items"]:
            container_idx = int(raw_item["belongs_to"])
            placed[container_idx].append(
                _packed_item_to_placed(raw_item, containers[container_idx])
            )

    # 4) height を全再構築する（増分キャッシュは行わない、T-013の範囲）。
    for container_idx, space in enumerate(containers):
        bake_placed(space, placed[container_idx])

    # 5) EMS をステートレスに全再構築し、コンテナごとの予算で上位選択する。
    ems: dict[int, list[EMSBox]] = {}
    ems_truncation: dict[int, float] = {}
    ems_top_n = StageParams().ems_top_n_per_container
    for container_idx, space in enumerate(containers):
        placed_aabbs = [
            (item.aabb_min_rel, item.aabb_max_rel) for item in placed[container_idx]
        ]
        ems_list = generate_ems(space, placed_aabbs)
        selected, truncation = select_topn(ems_list, ems_top_n)
        ems[container_idx] = selected
        ems_truncation[container_idx] = truncation

    # 6) pool は observation 側の pool_list から構築する。idx はプール内列挙位置（仮定A11）。
    pool: list[ItemSpec] = [
        ItemSpec(
            idx=pool_idx,
            size=np.array(
                [raw_item["length"], raw_item["width"], raw_item["height"]],
                dtype=np.float64,
            ),
            weight=float(raw_item["mass"]),
            kind=None,
            is_soft=bool(raw_item["is_soft"]),
            is_priority=bool(raw_item["is_prioritized"]),
        )
        for pool_idx, raw_item in enumerate(observation["pool_list"])
    ]

    return PackingState(
        containers=containers,
        placed=placed,
        pool=pool,
        ems=ems,
        ems_truncation=ems_truncation,
        meta=meta,
    )
