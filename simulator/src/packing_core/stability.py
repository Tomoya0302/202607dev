"""幾何安定性プロキシ（詳細仕様書 §4.6、T-020〜T-021）。

T-020 時点では `support_ratio`／`max_step_below`／`soft_below_ratio` の3関数のみを実装する。
`support_polygon`／`cg_margin`（monotone chain・凸包）は T-021 の責務であり、本モジュールでは
未定義のまま据え置く（stub・暫定固定値は置かない）。

いずれの関数も候補底面AABBのXY footprint（`container_space.cells_of_aabb` が返す格子セル範囲）
を対象に、`space.height`（現在の積み上げ上端。床は `floor_z` で初期化済み、§4.2 T-008）を
参照する。格子は絞り込み専用であり最終判定には使わないという§2規約のもと、本モジュールは
あくまで特徴量・スコア用のプロキシ値を返す。
"""
from typing import TYPE_CHECKING

import numpy as np

from src.packing_core.constants import EPS_GEOM, TOL_CONTACT
from src.packing_core.container_space import cells_of_aabb
from src.packing_core.geometry import aabb_from_center
from src.packing_core.types import Candidate

if TYPE_CHECKING:
    # 循環import回避（state.py は本モジュールをimportしないが、逆方向の実行時依存を
    # 作らないよう型チェック専用importに留める。masks.py と同方針）。
    from src.packing_core.state import PackingState


def _footprint(state: "PackingState", cand: Candidate) -> tuple:
    """候補底面のXY footprintと `space.height` の対象領域を求める（3関数共通の私有ヘルパ）。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        `(space, x_slice, y_slice, sub, bottom_z)` のタプル。`space` は
        `cand.container_idx` が指す `ContainerSpace`、`x_slice`/`y_slice` は
        `cells_of_aabb` が返す格子スライス、`sub` はその領域の `space.height`
        （float64、空範囲なら size 0）、`bottom_z` は候補底面のz座標。

    Raises:
        ValueError: `cand.container_idx` が `0 <= idx < len(state.containers)` の
            範囲外の場合。`cand.osize` の要素が非正の場合（`aabb_from_center` 由来）。
    """
    cidx = cand.container_idx
    if cidx < 0 or cidx >= len(state.containers):
        raise ValueError(
            f"container_idx が範囲外です: {cidx}（コンテナ数={len(state.containers)}）"
        )
    space = state.containers[cidx]

    pos_rel = np.asarray(cand.pos_rel, dtype=np.float64)
    osize = np.asarray(cand.osize, dtype=np.float64)
    amin, amax = aabb_from_center(pos_rel, osize)
    bottom_z = float(pos_rel[2] - osize[2] / 2.0)

    x_slice, y_slice = cells_of_aabb(space, amin, amax)
    sub = space.height[x_slice, y_slice]
    return space, x_slice, y_slice, sub, bottom_z


def support_ratio(state: "PackingState", cand: Candidate) -> float:
    """候補底面footprintのうち支持されているセルの面積比を返す（詳細仕様書 §4.6）。

    セル面積は一様なため、面積比はセル数比として計算する。`candidate_bottom_z =
    cand.pos_rel[2] - cand.osize[2]/2` に対し `space.height[i,j] >= candidate_bottom_z -
    TOL_CONTACT` を満たすセルを支持セルとする。床（`floor_z` で初期化済み）も同条件で
    支持として扱う。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        支持セル数 / 対象セル総数。対象セルが0件（空footprint）の場合は0.0。
        有限値・範囲 `0.0 <= result <= 1.0` のPython float。

    Raises:
        ValueError: `cand.container_idx` が範囲外の場合。
    """
    _, _, _, sub, bottom_z = _footprint(state, cand)
    if sub.size == 0:
        return 0.0
    supported = sub >= (bottom_z - TOL_CONTACT)
    return float(np.count_nonzero(supported)) / float(sub.size)


def max_step_below(state: "PackingState", cand: Candidate) -> float:
    """候補底面footprint全セルにおける `space.height` の最大段差を返す（詳細仕様書 §4.6）。

    対象は `support_ratio` と異なり支持セルに限定せず、候補底面AABBのXY footprint
    （`cells_of_aabb`）に含まれる全セルとする。値は `max(height) - min(height)`
    （footprint内の高低差）であり、`candidate_bottom_z` との差でも隣接セル間差でもない。
    `ceil_z`/`candidate_bottom_z` による補正・クリップは行わない。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        footprint内 `space.height` の `max - min`。対象セルが0件の場合は0.0。
        非負の有限Python float。

    Raises:
        ValueError: `cand.container_idx` が範囲外の場合。
    """
    _, _, _, sub, _ = _footprint(state, cand)
    if sub.size == 0:
        return 0.0
    return float(sub.max() - sub.min())


def soft_below_ratio(state: "PackingState", cand: Candidate) -> float:
    """候補底面の支持セルのうち、ソフト荷物上面が支えているセルの比率を返す（詳細仕様書 §4.6）。

    分母は `support_ratio` と同じ全支持セル（床支持セルも含む）。分子は、支持セルの
    うち次の3条件すべてを満たすセル（＝ソフト支持セル）：
    (1) `is_soft=True` の `PlacedItem` のXY投影（`cells_of_aabb`）がそのセルを含む、
    (2) その `PlacedItem.aabb_max_rel[2]` が、そのセルの `space.height` と `EPS_GEOM`
    以内で一致（＝そのソフト荷物が `space.height` を形成している最上面）、
    (3) そのセルが支持セル条件を満たす。単なるソフトAABBのXY投影内では数えず、
    `space.height` を実際に形成している最上面であることを要求する。同一高さにソフト・
    ハードが並存する退化ケースは、ソフトが1つでも存在すれば保守側にソフト支持として
    数える。床はソフト支持に数えない（対応する `PlacedItem` が無いため条件(1)で除外）。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        ソフト支持セル数 / 全支持セル数。支持セルが0件の場合は0.0。
        有限値・範囲 `0.0 <= result <= 1.0` のPython float。

    Raises:
        ValueError: `cand.container_idx` が範囲外の場合。
    """
    space, x_slice, y_slice, sub, bottom_z = _footprint(state, cand)
    if sub.size == 0:
        return 0.0

    supported = sub >= (bottom_z - TOL_CONTACT)
    denom = int(np.count_nonzero(supported))
    if denom == 0:
        return 0.0

    # コンテナ格子全体でソフト最上面マスクを構築し、footprint部分だけ切り出す。
    soft_top = np.zeros(space.height.shape, dtype=bool)
    for item in state.placed[cand.container_idx]:
        if not item.is_soft:
            continue
        sx, sy = cells_of_aabb(space, item.aabb_min_rel, item.aabb_max_rel)
        if sx.stop <= sx.start or sy.stop <= sy.start:
            continue
        region_height = space.height[sx, sy]
        is_top = np.abs(region_height - float(item.aabb_max_rel[2])) <= EPS_GEOM
        soft_top[sx, sy] |= is_top

    numer = int(np.count_nonzero(supported & soft_top[x_slice, y_slice]))
    return float(numer) / float(denom)
