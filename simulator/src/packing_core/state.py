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
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import GridParams, StageParams
from src.packing_core.container_space import (
    ContainerSpace,
    bake_placed,
    build_container_space,
    cache_bake_placed,
)
from src.packing_core.ems import generate_ems, select_topn, update_ems
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


def _build_meta(init: dict) -> dict:
    """meta を init から取得する（§4.4手順1。`optimize`/`lookahead_k` のみ、`spacing` は含めない）。"""
    return {
        "optimize": init["optimize"],
        "lookahead_k": init["lookahead_k"],
    }


def _build_containers(init: dict) -> list[ContainerSpace]:
    """init の `container_list` から列挙位置を index として `ContainerSpace` を構築する（§4.4手順2）。"""
    cell = GridParams().cell
    return [
        build_container_space(cdict, container_idx, cell)
        for container_idx, cdict in enumerate(init["container_list"])
    ]


def _route_placed(
    observation: dict, containers: list[ContainerSpace]
) -> dict[int, list[PlacedItem]]:
    """observation の `packed_items` を `belongs_to` で列挙 index へ振り分ける（§4.4手順3）。"""
    placed: dict[int, list[PlacedItem]] = {i: [] for i in range(len(containers))}
    for obs_cdict in observation["container_list"]:
        for raw_item in obs_cdict["packed_items"]:
            container_idx = int(raw_item["belongs_to"])
            placed[container_idx].append(
                _packed_item_to_placed(raw_item, containers[container_idx])
            )
    return placed


def _build_pool(observation: dict) -> list[ItemSpec]:
    """observation の `pool_list` を `ItemSpec` 化する（§4.4手順6、仮定A11）。"""
    return [
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
    meta = _build_meta(init)

    # 2) コンテナ形状は init 側の container_list から、列挙位置を index として構築する。
    containers = _build_containers(init)

    # 3) 現在の packed_items は observation 側の container_list から取得し、belongs_to で
    #    振り分ける（位置からの推測はしない。belongs_to を正とする）。
    placed = _route_placed(observation, containers)

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
    pool = _build_pool(observation)

    return PackingState(
        containers=containers,
        placed=placed,
        pool=pool,
        ems=ems,
        ems_truncation=ems_truncation,
        meta=meta,
    )


@dataclass(frozen=True)
class StateCache:
    """`build_state_cached` の増分キャッシュ（詳細仕様書 §4.4「増分キャッシュ」節）。

    Attributes:
        geometry_key: `(global_settings_key, tuple(container_geometry_key, ...))`。キャッシュ
            有効性判定専用の丸めない厳密キー。1e-6丸めはテストのEMS等価性比較にのみ用い、
            本キーには一切導入しない。
        placed_signatures: `container_idx` → 既配置AABB署名（丸めない float64 の6要素tuple）
            のソート済みtuple（多重集合。重複個数を保持するため `frozenset` は使わない）。
        heights: `container_idx` → 全再構築一致の height（独立コピー、`writeable=False`）。
        ems_full: `container_idx` → `select_topn` 前の `generate_ems` 相当EMS（tuple）。各
            `EMSBox` の `min_rel`/`max_rel` も返却状態とは独立な read-only コピー。
    """

    geometry_key: tuple
    placed_signatures: dict[int, tuple[tuple, ...]]
    heights: dict[int, np.ndarray]
    ems_full: dict[int, tuple[EMSBox, ...]]


def _global_settings_key() -> tuple:
    """全コンテナ共通設定のキー（丸めない）。`geometry_key` の第1要素（§4.4）。"""
    return (StageParams().ems_top_n_per_container,)


def _container_geometry_key(space: ContainerSpace) -> tuple:
    """コンテナ単位の geometry キー（丸めない）。`geometry_key` 第2要素の各要素（§4.4）。

    T-016B（A15、interface_notes.md §L.5）: `height`/`buffer` の個別値は既存キー要素
    （`inner_min_rel`/`inner_max_rel`/クリップ後`shelf_boxes`/`cut_planes`）から一般に
    一意復元できることを証明できなかった（クリップ後`shelf_boxes`が退化・空集合になり得る
    edge caseで分離情報が失われるため）。安全側の最小修正として、`path_*` 6フィールドを
    直接キーへ追加する（間接的な復元可能性に依拠しない）。
    """

    def _vec(a: np.ndarray) -> tuple:
        return tuple(float(x) for x in a)

    return (
        float(space.offset_x),
        _vec(space.inner_min_rel),
        _vec(space.inner_max_rel),
        float(space.cell),
        tuple((_vec(bmin), _vec(bmax)) for bmin, bmax in space.shelf_boxes),
        tuple((_vec(normal_rel), float(d)) for normal_rel, d in space.cut_planes),
        float(space.path_entry_y_rel),
        float(space.path_lane_x_min_geom_rel),
        float(space.path_lane_x_max_geom_rel),
        float(space.path_mid_resting_z_rel),
        float(space.path_mid_ceiling_z_rel),
        tuple((_vec(bmin), _vec(bmax)) for bmin, bmax in space.path_obstacle_boxes_rel),
    )


def _parse_cache_geometry(cache: StateCache) -> tuple | None:
    """`cache.geometry_key` を `(global_key, container_keys)` へ安全に分解する（§4.4）。

    構造が不正・欠損している場合は `None` を返す（検出可能な cache-miss として扱う）。
    """
    geometry_key = cache.geometry_key
    if not (isinstance(geometry_key, tuple) and len(geometry_key) == 2):
        return None
    global_key, container_keys = geometry_key
    if not isinstance(container_keys, tuple):
        return None
    return global_key, container_keys


def _aabb_signature(item: PlacedItem) -> tuple[float, ...]:
    """既配置AABBのキャッシュ有効性判定用署名（丸めない6要素tuple、§4.4）。

    1e-6丸めはテストのEMS等価性比較専用であり、キャッシュ有効性判定には使用しない
    （物理沈降等による1e-6未満の変化でも canonical な `height` は変化し得るため）。
    """
    return tuple(float(v) for v in (*item.aabb_min_rel, *item.aabb_max_rel))


def _multiset_subset(sub: Counter, sup: Counter) -> bool:
    """`sub` が `sup` に多重集合として包含されるかを判定する（追加のみ／削除・移動なし判定）。"""
    return all(count <= sup[key] for key, count in sub.items())


def _select_added(
    cur_placed: list[PlacedItem], cur_sigs: list[tuple], added: Counter
) -> list[PlacedItem]:
    """追加多重集合 `added` に対応する `PlacedItem` を `cur_placed` から必要個数だけ抽出する。

    同一AABBの重複は幾何的に同一であるため、対応する複数実体のうちどれを選んでも同値。
    """
    remaining = Counter(added)
    result: list[PlacedItem] = []
    for item, sig in zip(cur_placed, cur_sigs):
        if remaining[sig] > 0:
            result.append(item)
            remaining[sig] -= 1
    return result


def _copy_ems_box(box: EMSBox, *, readonly: bool) -> EMSBox:
    """`EMSBox.min_rel`/`max_rel` をそれぞれ独立コピーした新インスタンスを返す（§4.4）。

    `EMSBox` は `frozen=True` の dataclass だが、内部の NumPy 配列自体は可変であるため、
    フィールドの再代入不可というdataclassの性質だけでは配列内容の非共有を保証しない。
    返却 `PackingState` と `StateCache` が同一配列を共有しないよう、呼び出し側ごとに
    独立コピーする（`readonly=True` でキャッシュ側配列を `writeable=False` にする）。

    Args:
        box: コピー元の `EMSBox`。
        readonly: `True` の場合、コピー後の `min_rel`/`max_rel` を `writeable=False` にする。

    Returns:
        `min_rel`/`max_rel` が独立コピーされた新しい `EMSBox`。
    """
    min_rel = np.array(box.min_rel, dtype=np.float64)
    max_rel = np.array(box.max_rel, dtype=np.float64)
    if readonly:
        min_rel.flags.writeable = False
        max_rel.flags.writeable = False
    return EMSBox(min_rel=min_rel, max_rel=max_rel)


def _incremental_container(
    space: ContainerSpace,
    prev_sigs: tuple[tuple, ...],
    cur_placed: list[PlacedItem],
    cur_sigs: list[tuple],
    prev_height: np.ndarray,
    prev_ems_full: tuple[EMSBox, ...],
    ems_top_n: int,
) -> tuple[np.ndarray, list[EMSBox], float, list[EMSBox]] | None:
    """1コンテナ分の増分更新を試みる（§4.4）。

    検出可能な cache-miss（多重集合非包含・shape不一致・非有限値・署名対応の不整合）だけ
    `None` を返し、呼び出し側にそのコンテナの全再構築を行わせる。これらのチェックを通過した
    後の増分プリミティブ（`cache_bake_placed`/`update_ems`/`select_topn`）の呼び出しは例外を
    捕捉しない。プログラミングバグ・想定外例外はここで握りつぶさず伝播させ、テストを
    失敗させる（`Agent.policy()` の最終的な例外捕捉とは別レイヤー）。

    Args:
        space: 対象コンテナの `ContainerSpace`（セル判定・EMS更新に使用、書き換えない）。
        prev_sigs: キャッシュ側の前回 `placed_signatures[idx]`（丸めない署名の多重集合）。
        cur_placed: 現在のそのコンテナの `PlacedItem` 一覧。
        cur_sigs: `cur_placed` と同じ順序の現在の署名リスト。
        prev_height: キャッシュ側の前回 height（read-only）。
        prev_ems_full: キャッシュ側の前回 `ems_full[idx]`（select_topn 前の全EMS、read-only）。
        ems_top_n: `StageParams().ems_top_n_per_container`。

    Returns:
        `None`（検出可能な cache-miss）、または
        `(height_ro, selected_ems, truncation, ems_full_list)`。`ems_full_list` は
        select_topn 前の全EMS（コピーは呼び出し側の `_copy_ems_box` が行う）。
    """
    prev_counts = Counter(prev_sigs)
    cur_counts = Counter(cur_sigs)
    if not _multiset_subset(prev_counts, cur_counts):
        return None

    if prev_height.shape != space.height.shape or not np.all(np.isfinite(prev_height)):
        return None

    for box in prev_ems_full:
        if not (np.all(np.isfinite(box.min_rel)) and np.all(np.isfinite(box.max_rel))):
            return None

    added_counts = cur_counts - prev_counts
    added = _select_added(cur_placed, cur_sigs, added_counts)
    if len(added) != sum(added_counts.values()):
        return None

    for item in added:
        if not (np.all(np.isfinite(item.aabb_min_rel)) and np.all(np.isfinite(item.aabb_max_rel))):
            return None

    height_ro = cache_bake_placed(space, prev_height, added)

    ems_full_list = list(prev_ems_full)
    for item in added:
        ems_full_list = update_ems(ems_full_list, (item.aabb_min_rel, item.aabb_max_rel), space)

    selected, truncation = select_topn(ems_full_list, ems_top_n)
    return height_ro, selected, truncation, ems_full_list


def build_state_cached(
    observation: dict, init: dict, cache: StateCache | None = None,
) -> tuple[PackingState, StateCache]:
    """`build_state` と同じ入出力契約の `PackingState` を、増分キャッシュを用いて構築する（§4.4）。

    `build_state` は全再構築の唯一の canonical 経路として不変のまま維持する。本関数は
    コンテナ単位でキャッシュのヒット判定・増分更新・無効化（→全再構築フォールバック）を
    行い、更新後の `StateCache` も返す。可変グローバル状態・永続 `ContainerSpace` は
    使用しない（呼び出し側がキャッシュを明示的に保持し、引数と戻り値で受け渡す）。

    ヒット/無効化の別によらず、返却する `PackingState` は同一入力に対する `build_state` の
    結果と height（`np.array_equal` 完全一致）・EMS集合（§4.3と同じ1e-6丸め、テスト上の
    近似比較）の両方で一致する（この1e-6丸めはテスト専用であり、キャッシュ有効性判定
    ——AABB署名・`geometry_key`——には一切使用しない）。

    Args:
        observation: `build_state` と同じ observation（現在の `packed_items`/`pool_list` の
            取得元）。
        init: `build_state` と同じ init（コンテナ形状・`optimize`/`lookahead_k` の取得元）。
        cache: 前回の `StateCache`。`None` の場合は全コンテナを全再構築する。

    Returns:
        `(再構築された PackingState, 更新後の StateCache)`。
    """
    meta = _build_meta(init)
    containers = _build_containers(init)
    placed = _route_placed(observation, containers)

    cur_global_key = _global_settings_key()
    cur_container_keys = tuple(_container_geometry_key(space) for space in containers)

    parsed = _parse_cache_geometry(cache) if cache is not None else None

    ems_top_n = StageParams().ems_top_n_per_container

    ems: dict[int, list[EMSBox]] = {}
    ems_truncation: dict[int, float] = {}
    placed_signatures: dict[int, tuple[tuple, ...]] = {}
    heights_cache: dict[int, np.ndarray] = {}
    ems_full_cache: dict[int, tuple[EMSBox, ...]] = {}

    for container_idx, space in enumerate(containers):
        cur_sigs = [_aabb_signature(item) for item in placed[container_idx]]

        incremental = None
        if parsed is not None:
            cache_global_key, cache_container_keys = parsed
            if (
                cache_global_key == cur_global_key
                and container_idx < len(cache_container_keys)
                and cache_container_keys[container_idx] == cur_container_keys[container_idx]
                and container_idx in cache.placed_signatures
                and container_idx in cache.heights
                and container_idx in cache.ems_full
            ):
                incremental = _incremental_container(
                    space,
                    cache.placed_signatures[container_idx],
                    placed[container_idx],
                    cur_sigs,
                    cache.heights[container_idx],
                    cache.ems_full[container_idx],
                    ems_top_n,
                )

        if incremental is not None:
            height_ro, selected, truncation, ems_full_list = incremental
            space.height[...] = height_ro
        else:
            bake_placed(space, placed[container_idx])
            placed_aabbs = [
                (item.aabb_min_rel, item.aabb_max_rel) for item in placed[container_idx]
            ]
            ems_full_list = generate_ems(space, placed_aabbs)
            selected, truncation = select_topn(ems_full_list, ems_top_n)

        # height/EMS の非共有: 返却側は独立コピー（EMSは書込可）、cache側も独立コピー（read-only）。
        ems[container_idx] = [_copy_ems_box(box, readonly=False) for box in selected]
        ems_truncation[container_idx] = truncation
        placed_signatures[container_idx] = tuple(sorted(cur_sigs))

        height_copy = space.height.copy()
        height_copy.flags.writeable = False
        heights_cache[container_idx] = height_copy
        ems_full_cache[container_idx] = tuple(
            _copy_ems_box(box, readonly=True) for box in ems_full_list
        )

    pool = _build_pool(observation)

    state = PackingState(
        containers=containers,
        placed=placed,
        pool=pool,
        ems=ems,
        ems_truncation=ems_truncation,
        meta=meta,
    )
    new_cache = StateCache(
        geometry_key=(cur_global_key, cur_container_keys),
        placed_signatures=placed_signatures,
        heights=heights_cache,
        ems_full=ems_full_cache,
    )
    return state, new_cache
