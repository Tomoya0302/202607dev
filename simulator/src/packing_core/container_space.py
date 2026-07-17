"""有効空間と格子（詳細仕様書 §4.2、T-006〜T-008）。

本モジュールは cut・棚を含む有効空間（連続幾何）と絞り込み用格子を構築する（T-006: 直方体、
T-007: cut plane と棚のAABBを反映した floor_z/ceil_z・contains_oriented_box 拡張、
T-008: `cells_of_aabb` によるAABB→格子sliceの変換と `bake_placed` による `height` 全再構築）。
"""
from dataclasses import dataclass

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM
from src.packing_core.types import PlacedItem, Vec3


@dataclass
class ContainerSpace:
    """コンテナ1台分の有効空間と絞り込み用格子。

    Attributes:
        index: コンテナ index（`cdict["index"]` と同一）。
        offset_x: `index * spacing`。コンテナ原点の世界座標X（詳細仕様書 §3.1）。
        inner_min_rel: 内壁AABBの最小点（コンテナ相対）。shape (3,), float64。
        inner_max_rel: 内壁AABBの最大点（コンテナ相対）。shape (3,), float64。
        cut_planes: 軸整列でない内壁面の半空間 `(normal_rel, d)` のリスト。
            有効空間 = 内壁AABB ∩ {x | normal_rel・x <= d for all}。cut なしでは空。
        shelf_boxes: 棚＝障害物AABB（相対）のリスト。`(min, max)` のタプル。
            小棚（常時計算・クリップ後正体積のみ）→ 大棚（`cdict["shelf"] is True` のみ）の
            順で決定的に格納する（interface_notes.md §K.2/K.3）。棚なしでは空。
        cell: 絞り込み格子のセル一辺の長さ [m]（`GridParams.cell`）。
        floor_z: 格子セルごとの床高さ。cut plane で持ち上がった床を反映する。
            shape (nx, ny), float64。
        ceil_z: 格子セルごとの天井高さ。内壁天井と棚下面の min を反映する。
            shape (nx, ny), float64。
        height: 現在の積み上げ上端。shape (nx, ny), float64。`bake_placed`（T-008）で更新。
    """

    index: int
    offset_x: float
    inner_min_rel: Vec3
    inner_max_rel: Vec3
    cut_planes: list
    shelf_boxes: list[tuple[Vec3, Vec3]]
    cell: float
    floor_z: np.ndarray
    ceil_z: np.ndarray
    height: np.ndarray


def _axis_alignment(normal_rel: np.ndarray) -> tuple[int, float] | None:
    """法線が軸整列単位ベクトルなら `(axis, sign)` を返し、そうでなければ `None`。

    Args:
        normal_rel: 単位法線ベクトル（コンテナ相対、平行移動不変）。shape (3,), float64。

    Returns:
        軸整列なら `(axis, sign)`（`axis` は 0..2, `sign` は +1.0 か -1.0）。非軸整列なら `None`。
    """
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        if abs(abs(normal_rel[axis]) - 1.0) < EPS_GEOM and all(
            abs(normal_rel[k]) < EPS_GEOM for k in others
        ):
            return axis, float(np.sign(normal_rel[axis]))
    return None


def _recover_buffer(cdict: dict) -> float:
    """`buffer` を `cdict["center"][2]`/`cdict["height"]` から復元する（interface_notes.md §K.1）。

    `buffer` は `cdict` に存在しないキーであり、固定値で補ってはならない
    （`docs/interface_notes.md` §I-7）。
    """
    return float(cdict["center"][2]) - float(cdict["height"]) / 2.0


def _small_shelf_raw_aabb(cdict: dict, buffer: float) -> tuple[np.ndarray, np.ndarray]:
    """小棚（`_create_small_shelf` 相当）のクリップ前AABB（interface_notes.md §K.2）。"""
    length = float(cdict["length"])
    width = float(cdict["width"])
    height = float(cdict["height"])
    thickness = float(cdict["thickness"])
    cut_x = float(cdict["cut_x"])

    center = np.array(
        [-length / 2.0 + cut_x / 2.0 + thickness, 0.0, height / 2.0 + thickness / 2.0 + buffer],
        dtype=np.float64,
    )
    half = np.array([cut_x / 2.0, width / 2.0 - thickness, thickness / 2.0], dtype=np.float64)
    return center - half, center + half


def _main_shelf_raw_aabb(cdict: dict, buffer: float) -> tuple[np.ndarray, np.ndarray]:
    """大棚（`_create_shelf` 相当）のクリップ前AABB（interface_notes.md §K.3）。"""
    length = float(cdict["length"])
    width = float(cdict["width"])
    height = float(cdict["height"])
    thickness = float(cdict["thickness"])

    center = np.array(
        [0.0, width / 4.0, height / 2.0 + thickness / 2.0 + buffer], dtype=np.float64
    )
    half = np.array(
        [length / 2.0 - thickness / 2.0, width / 4.0 - thickness, thickness / 2.0],
        dtype=np.float64,
    )
    return center - half, center + half


def _clip_shelf_aabb(
    raw_min: np.ndarray, raw_max: np.ndarray, inner_min_rel: np.ndarray, inner_max_rel: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """内壁AABBへクリップし、全軸長が `EPS_GEOM` を超える場合のみ返す（§K.4）。

    ゼロ・負寸法になる場合は `None`（棚として追加しない）。
    """
    clip_min = np.maximum(raw_min, inner_min_rel)
    clip_max = np.minimum(raw_max, inner_max_rel)
    if np.all(clip_max - clip_min > EPS_GEOM):
        return clip_min, clip_max
    return None


def _build_shelf_boxes(
    cdict: dict, buffer: float, inner_min_rel: np.ndarray, inner_max_rel: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """小棚→大棚の順で `shelf_boxes` を決定的に構築する（interface_notes.md §K.2/K.3）。

    小棚は `cdict["shelf"]`・`cut_planes` の有無に関わらず常に計算する（公式コード上、
    `require_shelf` の分岐によらず無条件生成されるため）。大棚は `cdict["shelf"] is True`
    のときのみ計算する。生成条件を `cut_planes` の有無で判定してはならない。
    """
    boxes: list[tuple[np.ndarray, np.ndarray]] = []

    small_clipped = _clip_shelf_aabb(*_small_shelf_raw_aabb(cdict, buffer), inner_min_rel, inner_max_rel)
    if small_clipped is not None:
        boxes.append(small_clipped)

    if cdict["shelf"] is True:
        main_clipped = _clip_shelf_aabb(*_main_shelf_raw_aabb(cdict, buffer), inner_min_rel, inner_max_rel)
        if main_clipped is not None:
            boxes.append(main_clipped)

    return boxes


def _cell_centers(
    inner_min_rel: np.ndarray, inner_max_rel: np.ndarray, cell: float
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """格子添字・セル中心座標を返す（interface_notes.md §K.7 のセル中心規約）。

    `x_c[i] = inner_min_rel[0] + (i+0.5)*cell`（`i=0,...,nx-1`、`y` 軸も同様）。
    `nx = max(1, round(size[0]/cell))` は T-006 の既存式と同一。
    """
    size = inner_max_rel - inner_min_rel
    nx = max(1, round(size[0] / cell))
    ny = max(1, round(size[1] / cell))
    xs = inner_min_rel[0] + (np.arange(nx) + 0.5) * cell
    ys = inner_min_rel[1] + (np.arange(ny) + 0.5) * cell
    return nx, ny, xs, ys


def _build_floor_ceil(
    inner_min_rel: np.ndarray,
    inner_max_rel: np.ndarray,
    cut_planes: list,
    shelf_boxes: list[tuple[np.ndarray, np.ndarray]],
    cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`floor_z`/`ceil_z` をセル単位ループなしで構築する（interface_notes.md §K.7/K.8）。

    セル中心ごとに `cut_planes` を Z について解き、`normal_z<-EPS_GEOM` の面は `floor_z` の
    下限候補（`max`で反映）、`normal_z>EPS_GEOM` の面は `ceil_z` の上限候補（`min`で反映）と
    する。`|normal_z|<=EPS_GEOM` の面は Z を拘束しないため寄与させない。`shelf_boxes` は
    XY範囲が交差するセルについて `ceil_z` を棚下面と `min` で反映する（境界は
    `X>=box_min[0] and X<=box_max[0]` の閉区間、golden fixture と同一規約）。
    `floor_z > ceil_z` となるセルはそのまま保持し、修正しない（絞り込み専用の格子であり
    最終包含判定には使わない。`effective_volume` は `max(ceil_z-floor_z, 0)` で吸収する）。
    """
    nx, ny, xs, ys = _cell_centers(inner_min_rel, inner_max_rel, cell)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")

    floor_z = np.full((nx, ny), inner_min_rel[2], dtype=np.float64)
    ceil_z = np.full((nx, ny), inner_max_rel[2], dtype=np.float64)

    for normal_rel, d in cut_planes:
        nz = normal_rel[2]
        if nz < -EPS_GEOM:
            candidate = (d - normal_rel[0] * x_grid - normal_rel[1] * y_grid) / nz
            floor_z = np.maximum(floor_z, candidate)
        elif nz > EPS_GEOM:
            candidate = (d - normal_rel[0] * x_grid - normal_rel[1] * y_grid) / nz
            ceil_z = np.minimum(ceil_z, candidate)
        # |nz| <= EPS_GEOM: Z を拘束しないため floor_z/ceil_z には寄与させない（§K.8）。

    for box_min, box_max in shelf_boxes:
        covers = (
            (x_grid >= box_min[0])
            & (x_grid <= box_max[0])
            & (y_grid >= box_min[1])
            & (y_grid <= box_max[1])
        )
        ceil_z = np.where(covers, np.minimum(ceil_z, box_min[2]), ceil_z)

    return floor_z, ceil_z


def build_container_space(cdict: dict, index: int, spacing: float, cell: float) -> ContainerSpace:
    """`container_list[i]` の辞書から有効空間と絞り込み格子を構築する（cut・棚対応）。

    `cdict["points"]`（世界座標の代表点）と `cdict["n_vecs"]`（外向き単位法線）を正として
    内壁形状（`inner_min_rel`/`inner_max_rel`/`cut_planes`）を復元する。`buffer` キーは
    `cdict` に存在しないため参照せず、`_recover_buffer` で都度復元する
    （`docs/interface_notes.md` §I-7/§K.1）。

    Args:
        cdict: `container_list` の1要素。`constants.OBS_KEYS["container"]` のキーを持つ。
        index: コンテナ index。
        spacing: コンテナ間隔 [m]（`init_states` から取得済みの値）。
        cell: 絞り込み格子のセル一辺の長さ [m]（`GridParams.cell`）。

    Returns:
        `ContainerSpace`。cut なしの入力では `cut_planes` は空リストになる。

    Raises:
        ValueError: 軸整列面が6面（±X/±Y/±Z）揃わず内壁AABBが確定しない場合、
            内壁AABBの一辺が0以下になる場合、または `cdict["n_vecs"]` にゼロ長
            （`EPS_GEOM` 以下）の法線が含まれる場合。
    """
    offset_x = index * spacing
    origin_world = np.array([offset_x, 0.0, 0.0], dtype=np.float64)

    inner_min_rel = np.full(3, -np.inf, dtype=np.float64)
    inner_max_rel = np.full(3, np.inf, dtype=np.float64)
    cut_planes: list = []

    for point_world, normal_world in zip(cdict["points"], cdict["n_vecs"]):
        point_rel = np.asarray(point_world, dtype=np.float64) - origin_world
        normal_rel = np.asarray(normal_world, dtype=np.float64)

        # cut_planes の半空間厳格化（margin付き判定）は normal_rel が単位ベクトルである
        # ことを前提とする。公式 `write_open_cut_corner_cup_obj`（ground_handling/utils.py:
        # 206-207 `n_vec=[ey/ne,-ex/ne,0]` で ne=hypot(ex,ey) により正規化済み、および
        # 211行目の軸単位ベクトル `[0,0,±1]`）が生成した法線は単位ベクトルであり、
        # `containers.py::create()` の `aff_n_vecs` はこれに回転行列（直交行列、ノルム保存）
        # のみを適用するため、`cdict["n_vecs"]` は単位ベクトルのまま渡ってくる
        # （interface_notes.md §K.7 追記）。ゼロ長・退化した法線を黙って cut_planes へ
        # 混入させないよう、ここで明示的に検出する。
        norm = float(np.linalg.norm(normal_rel))
        if norm <= EPS_GEOM:
            raise ValueError(
                f"cdict['n_vecs'] にゼロ長（EPS_GEOM以下）の法線が含まれます: {normal_world}"
            )

        d = float(np.dot(normal_rel, point_rel))

        alignment = _axis_alignment(normal_rel)
        if alignment is None:
            cut_planes.append((normal_rel, d))
            continue

        axis, sign = alignment
        if sign > 0:
            inner_max_rel[axis] = min(inner_max_rel[axis], d)
        else:
            inner_min_rel[axis] = max(inner_min_rel[axis], -d)

    if np.any(np.isinf(inner_min_rel)) or np.any(np.isinf(inner_max_rel)):
        raise ValueError("cdict の points/n_vecs から内壁AABBの全軸を復元できません")

    size = inner_max_rel - inner_min_rel
    if np.any(size <= 0):
        raise ValueError(f"内壁AABBの一辺が0以下です: size={size}")

    buffer = _recover_buffer(cdict)
    shelf_boxes = _build_shelf_boxes(cdict, buffer, inner_min_rel, inner_max_rel)
    floor_z, ceil_z = _build_floor_ceil(inner_min_rel, inner_max_rel, cut_planes, shelf_boxes, cell)
    height = floor_z.copy()

    return ContainerSpace(
        index=index,
        offset_x=float(offset_x),
        inner_min_rel=inner_min_rel,
        inner_max_rel=inner_max_rel,
        cut_planes=cut_planes,
        shelf_boxes=shelf_boxes,
        cell=float(cell),
        floor_z=floor_z,
        ceil_z=ceil_z,
        height=height,
    )


def _aabb_corners(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """AABBの8頂点を返す。shape (8, 3), float64。"""
    xs = (box_min[0], box_max[0])
    ys = (box_min[1], box_max[1])
    zs = (box_min[2], box_max[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def contains_oriented_box(
    space: ContainerSpace, center_rel: Vec3, osize: Vec3, margin: float
) -> bool:
    """回転後の箱が有効空間に完全に収まるかを判定する（cut・棚を含む連続幾何判定）。

    `margin` の符号規約は `geometry.aabb_contains` と統一する：正で厳格化（必要な
    クリアランスを増やす）、0で面接触は許容、負で緩和する。cut plane 判定・棚判定にも
    同じ規約を適用する（cut plane は許容半空間を `margin` だけ内側に縮める。棚は
    `geometry.aabb_intersects(..., tol=-margin)` により、`margin>0` で棚との必要離隔を
    増やし、`margin==0` で面接触を交差とみなさず、`margin<0` で許容領域を緩和する）。

    格子（`floor_z`/`ceil_z`）は絞り込み専用であり、本関数の最終判定には使用しない。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        center_rel: 箱の中心（コンテナ相対）。shape (3,), float64。
        osize: 回転後寸法。shape (3,), float64。
        margin: 判定を縮める量 [m]（`geometry.aabb_contains` と同じ符号規約）。

    Returns:
        内壁AABBに収まり、全 `cut_planes` の半空間を満たし、`shelf_boxes` のいずれとも
        交差しなければ True。
    """
    box_min, box_max = geometry.aabb_from_center(center_rel, osize)

    if not geometry.aabb_contains(
        space.inner_min_rel, space.inner_max_rel, box_min, box_max, margin=margin
    ):
        return False

    if space.cut_planes:
        corners = _aabb_corners(box_min, box_max)
        for normal_rel, d in space.cut_planes:
            # normal_rel は単位ベクトル（build_container_space で検証済み）である前提で、
            # 距離 margin をそのまま d から減じて許容半空間を縮める。
            if np.any(corners @ normal_rel > d - margin):
                return False

    for shelf_min, shelf_max in space.shelf_boxes:
        if geometry.aabb_intersects(box_min, box_max, shelf_min, shelf_max, tol=-margin):
            return False

    return True


def effective_volume(space: ContainerSpace) -> float:
    """`floor_z`/`ceil_z` 単一区間格子上の近似・診断用体積を返す（モンテカルロ不可）。

    `sum(max(ceil_z-floor_z, 0)) * cell * cell` で計算する（境界セルの面積補正はしない、
    詳細仕様書 §4.2 T-007節）。以下の既知制約に注意すること：

    * 棚のあるセルでは `ceil_z` を棚下面に設定するため、棚上方空間を保守的に無視する
      （実際にはその空間へも荷物を配置できる場合があるが、本関数は過小評価する）。
    * cut形状では、格子離散化と公式体積式（`cdict["volume"]`）の形状定義差
      （入口面の壁厚オフセットの有無など）により誤差が生じる
      （`docs/interface_notes.md` §K.10）。
    * `cdict["volume"]` との一致を目的にしない。公式fillスコアの分母としても使用しない。
      fill分母は `evaluator.py`（または転記済みの公式値）を正とする。
    * 包含の最終判定は本関数ではなく `contains_oriented_box()` の連続幾何を使用すること。

    Args:
        space: 対象コンテナの `ContainerSpace`。

    Returns:
        格子表現上の近似体積 [m^3]。
    """
    return float(np.sum(np.maximum(space.ceil_z - space.floor_z, 0.0)) * space.cell * space.cell)


def cells_of_aabb(
    space: ContainerSpace, bmin: Vec3, bmax: Vec3
) -> tuple[slice, slice]:
    """AABBのXY範囲を格子インデックスの `(x_slice, y_slice)` へ変換する（詳細仕様書 §4.2 T-008節）。

    セルはセル中心（`_cell_centers` と同一の生成式）で代表し、半開区間
    `bmin[k] <= center < bmax[k]`（下端を含み上端を含まない）を満たすセルを選ぶ。
    2つのAABBが同一平面で面接触し、境界がちょうどセル中心と一致する場合、境界セルは
    `bmax` 側のAABBにのみ帰属し、重複帰属しない。

    XY範囲は先にコンテナ内壁のXY範囲へclampし、`EPS_GEOM` によるセル占有範囲の外側拡張は
    行わない。clamp後にいずれかの軸で正の幅を持たない場合、および `searchsorted` 後の
    インデックスがいずれかの軸で `start >= stop` となる場合は、canonicalな空表現
    `(slice(0, 0), slice(0, 0))` を返す（片方の軸だけ実範囲を残さない）。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        bmin: AABBの最小点（コンテナ相対）。shape (3,), float64。
        bmax: AABBの最大点（コンテナ相対）。shape (3,), float64。

    Returns:
        `(x_slice, y_slice)`。交差なし・空範囲の場合は `(slice(0, 0), slice(0, 0))`。

    Raises:
        ValueError: `bmin`/`bmax` が shape (3,) でない場合、有限値でない要素を含む場合、
            またはいずれかの軸で `bmax[k] < bmin[k] - EPS_GEOM` となる場合。
    """
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)

    if bmin.shape != (3,) or bmax.shape != (3,):
        raise ValueError(
            f"bmin/bmax は shape (3,) である必要があります: bmin.shape={bmin.shape}, "
            f"bmax.shape={bmax.shape}"
        )
    if not (np.all(np.isfinite(bmin)) and np.all(np.isfinite(bmax))):
        raise ValueError(f"bmin/bmax は全要素が有限値である必要があります: bmin={bmin}, bmax={bmax}")
    if np.any(bmax < bmin - EPS_GEOM):
        raise ValueError(f"bmax が bmin を下回っています（EPS_GEOM超過）: bmin={bmin}, bmax={bmax}")

    clipped_min = np.maximum(bmin[:2], space.inner_min_rel[:2])
    clipped_max = np.minimum(bmax[:2], space.inner_max_rel[:2])

    # 積ではなく軸ごとに判定する（両軸が負の場合に積が正になる誤判定を避けるため）。
    if np.any(clipped_max <= clipped_min):
        return slice(0, 0), slice(0, 0)

    _, _, xs, ys = _cell_centers(space.inner_min_rel, space.inner_max_rel, space.cell)
    nx, ny = space.height.shape

    x_start = int(np.clip(np.searchsorted(xs, clipped_min[0], side="left"), 0, nx))
    x_stop = int(np.clip(np.searchsorted(xs, clipped_max[0], side="left"), 0, nx))
    y_start = int(np.clip(np.searchsorted(ys, clipped_min[1], side="left"), 0, ny))
    y_stop = int(np.clip(np.searchsorted(ys, clipped_max[1], side="left"), 0, ny))

    if x_start >= x_stop or y_start >= y_stop:
        return slice(0, 0), slice(0, 0)

    return slice(x_start, x_stop), slice(y_start, y_stop)


def bake_placed(space: ContainerSpace, placed: list[PlacedItem]) -> None:
    """既配置荷物のAABBから `height` を全再構築する（詳細仕様書 §4.2 T-008節）。

    呼び出しごとに `floor_z` から作り直すため、過去の `height` は引き継がない
    （増分キャッシュ・`cache_*` 関数は本関数のスコープ外）。各 `PlacedItem` の
    `aabb_min_rel`/`aabb_max_rel`（回転後AABB、コンテナ相対）をそのまま `cells_of_aabb` へ
    渡し、空sliceが返れば当該荷物をスキップする。対象セルは現在値と `aabb_max_rel[2]` の
    最大値で更新し（`ceil_z` によるclampはしない）、`floor_z`/`ceil_z` は変更しない。

    Args:
        space: 対象コンテナの `ContainerSpace`。`height` をその場で書き換える。
        placed: 対象コンテナに既に配置された荷物のリスト。
    """
    space.height[...] = space.floor_z

    for item in placed:
        x_slice, y_slice = cells_of_aabb(space, item.aabb_min_rel, item.aabb_max_rel)
        if x_slice.stop <= x_slice.start or y_slice.stop <= y_slice.start:
            continue
        space.height[x_slice, y_slice] = np.maximum(
            space.height[x_slice, y_slice], item.aabb_max_rel[2]
        )
