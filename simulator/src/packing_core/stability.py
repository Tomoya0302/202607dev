"""幾何安定性プロキシ（詳細仕様書 §4.6、T-020〜T-021）。

T-020 では `support_ratio`／`max_step_below`／`soft_below_ratio` の3関数を実装した。
T-021 では `support_polygon`／`cg_margin`（monotone chain・凸包）を追加する。

いずれの関数も候補底面AABBのXY footprint（`container_space.cells_of_aabb` が返す格子セル範囲）
を対象に、`space.height`（現在の積み上げ上端。床は `floor_z` で初期化済み、§4.2 T-008）を
参照する。格子は絞り込み専用であり最終判定には使わないという§2規約のもと、本モジュールは
あくまで特徴量・スコア用のプロキシ値を返す。

`support_polygon` は仕様書 §4.6 v1.14 追補のとおり、支持セルが表す矩形接触パッチ（各セルの
生矩形を候補底面AABBおよびコンテナ内壁XYでクリップしたもの）の全頂点から凸包を構成する
（セル中心点のみの凸包では候補底面の実境界より半セル内側になり、平床・半分支持のテスト
期待値を満たせないため。詳細は `docs/実装詳細仕様書.md` §4.6 更新履歴 v1.14 参照）。
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


def expected_settled_pos_rel(state: "PackingState", cand: Candidate) -> np.ndarray:
    """候補の想定沈降後中心（コンテナ相対）を返す（詳細仕様書 §4.6、HF-001でv1.27新設）。

    `Candidate.pos_rel` はHF-001（§4.12）以降「公式へ実際に出力する沈降前のaction目標位置」
    であり、EMS支持面から`z_generation_clearance`だけ浮いている。安定性プロキシ・
    `heuristic_score`のZ関連項は、この浮いたaction位置ではなく、荷物が実際に沈降した後の
    想定位置（EMS支持面へ底面一致する位置）を参照する必要があるため、本関数がその変換を担う。

    X/Yはaction位置と想定沈降後位置で同一のため変更しない。Zのみ、`cand.ems_id`が指す
    EMSの支持面（`ems.min_rel[2]`）から `ems.min_rel[2] + cand.osize[2]/2.0` として
    再計算する。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補（変更しない。戻り値は新規配列のコピー）。

    Returns:
        想定沈降後中心。shape (3,), float64。`cand.pos_rel`・`cand`自体は変更しない。

    Raises:
        ValueError: `cand.container_idx` が `0 <= idx < len(state.containers)` の
            範囲外の場合。`cand.ems_id` が `0 <= ems_id < len(state.ems[container_idx])`
            の範囲外の場合。
    """
    cidx = cand.container_idx
    if cidx < 0 or cidx >= len(state.containers):
        raise ValueError(
            f"container_idx が範囲外です: {cidx}（コンテナ数={len(state.containers)}）"
        )
    ems_list = state.ems[cidx]
    if cand.ems_id < 0 or cand.ems_id >= len(ems_list):
        raise ValueError(
            f"ems_id が範囲外です: {cand.ems_id}（container_idx={cidx}のEMS数={len(ems_list)}）"
        )
    ems = ems_list[cand.ems_id]

    result = np.array(cand.pos_rel, dtype=np.float64, copy=True)
    result[2] = float(ems.min_rel[2]) + float(cand.osize[2]) / 2.0
    return result


def _footprint(state: "PackingState", cand: Candidate) -> tuple:
    """候補底面のXY footprintと `space.height` の対象領域を求める（3関数共通の私有ヘルパ）。

    HF-001（v1.27）：底面AABBのZ成分・`bottom_z`は`cand.pos_rel`ではなく
    `expected_settled_pos_rel(state, cand)`（想定沈降後位置）から計算する。X/Yは
    `cand.pos_rel`/`cand.osize`から計算する（action位置と沈降後位置でX/Yは同一のため）。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        `(space, x_slice, y_slice, sub, bottom_z)` のタプル。`space` は
        `cand.container_idx` が指す `ContainerSpace`、`x_slice`/`y_slice` は
        `cells_of_aabb` が返す格子スライス、`sub` はその領域の `space.height`
        （float64、空範囲なら size 0）、`bottom_z` は候補底面（想定沈降後位置）のz座標。

    Raises:
        ValueError: `cand.container_idx`/`cand.ems_id` が範囲外の場合
            （`expected_settled_pos_rel` 由来）。`cand.osize` の要素が非正の場合
            （`aabb_from_center` 由来）。
    """
    settled_pos_rel = expected_settled_pos_rel(state, cand)
    space = state.containers[cand.container_idx]

    osize = np.asarray(cand.osize, dtype=np.float64)
    amin, amax = aabb_from_center(settled_pos_rel, osize)
    bottom_z = float(settled_pos_rel[2] - osize[2] / 2.0)

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


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """点集合から Andrew monotone chain で凸包を構築する（T-021の私有ヘルパ）。

    Args:
        points: 点集合。shape (N, 2), float64。(x, y) 辞書順ソート済み・重複なしを前提とする
            （`np.unique(pts, axis=0)` の出力はこの前提を満たす）。

    Returns:
        反時計回りの凸包頂点。shape (K, 2), float64。始点は末尾に重複させない。辞書順最小の
        頂点が先頭に来る。`N==0` なら shape (0,2)、`N==1` なら shape (1,2)、`N==2` または
        全点共線なら両端2点。cross積が `EPS_GEOM` 以下の共線中間点は除去し端点のみ残す。

    Raises:
        なし。
    """
    n = points.shape[0]
    if n <= 2:
        return np.array(points, dtype=np.float64).reshape(n, 2)

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= EPS_GEOM:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in points[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= EPS_GEOM:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64).reshape(len(hull), 2)


def _cg_margin_from_polygon(polygon: np.ndarray, point: np.ndarray) -> float:
    """凸多角形境界までの符号付き距離を返す（T-021の私有ヘルパ）。

    Args:
        polygon: 反時計回りの凸包頂点。shape (K, 2), float64。
        point: 対象点（重心XY投影）。shape (2,), float64。

    Returns:
        `K < 3` のとき `float("-inf")`。それ以外は、全辺で `cross(edge, point - vertex) >=
        -EPS_GEOM` を満たせば内側/境界、いずれかの辺で満たさなければ外側と判定する。境界までの
        距離は全辺の有限線分への最短ユークリッド距離（射影係数は `[0,1]` にクランプ）。
        最短距離が `EPS_GEOM` 以下なら `0.0`、内側なら `+距離`、外側なら `-距離`。Python `float`。

    Raises:
        なし。
    """
    k = polygon.shape[0]
    if k < 3:
        return float("-inf")

    p = np.asarray(point, dtype=np.float64)
    inside = True
    min_dist = float("inf")
    for idx in range(k):
        a = polygon[idx]
        b = polygon[(idx + 1) % k]
        edge = b - a
        cross_val = float(edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]))
        if cross_val < -EPS_GEOM:
            inside = False

        seg_len_sq = float(np.dot(edge, edge))
        if seg_len_sq <= EPS_GEOM:
            dist = float(np.linalg.norm(p - a))
        else:
            t = float(np.dot(p - a, edge) / seg_len_sq)
            t = min(1.0, max(0.0, t))
            proj = a + t * edge
            dist = float(np.linalg.norm(p - proj))
        min_dist = min(min_dist, dist)

    if min_dist <= EPS_GEOM:
        return 0.0
    return min_dist if inside else -min_dist


def support_polygon(state: "PackingState", cand: Candidate) -> np.ndarray:
    """候補を支持するセルから支持凸包を生成する（詳細仕様書 §4.6 v1.14追補）。

    支持セル条件は `support_ratio` と完全に同じ
    （`height[i, j] >= candidate_bottom_z - TOL_CONTACT`）。各支持セルについて、格子上の
    生矩形 `[inner_min_rel + i*cell, inner_min_rel + (i+1)*cell]`（j軸も同様）を、候補底面
    AABBのXY範囲とコンテナ内壁XY範囲（`inner_min_rel`/`inner_max_rel`）の両方でクリップし、
    クリップ後に正の面積を持つセルの4隅を点集合へ加える。同一点は `np.unique(..., axis=0)`
    で除去し、(x, y) 辞書順に並べたうえで `_convex_hull` により決定論的に凸包を構築する。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        支持凸包の頂点。反時計回り、辞書順最小の頂点が先頭、始点を末尾に重複させない。
        shape (K, 2), float64。支持セルが0件（空footprint含む）の場合は
        `np.empty((0, 2), dtype=np.float64)`。scipy・shapely等の外部幾何ライブラリは使わない。

    Raises:
        ValueError: `cand.container_idx` が範囲外の場合。`cand.osize` の要素が非正の場合。
    """
    space, x_slice, y_slice, sub, bottom_z = _footprint(state, cand)
    if sub.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    supported = sub >= (bottom_z - TOL_CONTACT)
    local_i, local_j = np.nonzero(supported)
    if local_i.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    pos_rel = np.asarray(cand.pos_rel, dtype=np.float64)
    osize = np.asarray(cand.osize, dtype=np.float64)
    amin, amax = aabb_from_center(pos_rel, osize)

    cell = space.cell
    inner_min = space.inner_min_rel
    inner_max = space.inner_max_rel

    global_i = x_slice.start + local_i
    global_j = y_slice.start + local_j

    rx0 = inner_min[0] + global_i * cell
    rx1 = inner_min[0] + (global_i + 1) * cell
    ry0 = inner_min[1] + global_j * cell
    ry1 = inner_min[1] + (global_j + 1) * cell

    clip_x_min = max(float(amin[0]), float(inner_min[0]))
    clip_x_max = min(float(amax[0]), float(inner_max[0]))
    clip_y_min = max(float(amin[1]), float(inner_min[1]))
    clip_y_max = min(float(amax[1]), float(inner_max[1]))

    x0 = np.maximum(rx0, clip_x_min)
    x1 = np.minimum(rx1, clip_x_max)
    y0 = np.maximum(ry0, clip_y_min)
    y1 = np.minimum(ry1, clip_y_max)

    valid = (x1 - x0 > EPS_GEOM) & (y1 - y0 > EPS_GEOM)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    x0, x1 = x0[valid], x1[valid]
    y0, y1 = y0[valid], y1[valid]
    pts = np.concatenate(
        [
            np.stack([x0, y0], axis=1),
            np.stack([x1, y0], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x0, y1], axis=1),
        ],
        axis=0,
    ).astype(np.float64)

    pts = np.unique(pts, axis=0)
    return _convex_hull(pts)


def cg_margin(state: "PackingState", cand: Candidate) -> float:
    """候補重心XYから支持凸包境界までの符号付き距離を返す（詳細仕様書 §4.6）。

    候補重心XYは `cand.pos_rel[:2]`。`support_polygon` の頂点数が `K < 3`（点・線支持）
    のときは `float("-inf")`（=不安定）。凸包内側では正、境界上では0、外側では負。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。

    Returns:
        符号付き距離。内側 `+距離`、境界 `0.0`、外側 `-距離`、`K<3` は `float("-inf")`。
        Python `float`。

    Raises:
        ValueError: `cand.container_idx` が範囲外の場合。`cand.osize` の要素が非正の場合。
    """
    polygon = support_polygon(state, cand)
    point = np.asarray(cand.pos_rel[:2], dtype=np.float64)
    return _cg_margin_from_polygon(polygon, point)
