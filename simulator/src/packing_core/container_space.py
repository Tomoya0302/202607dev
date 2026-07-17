"""有効空間と格子（詳細仕様書 §4.2、T-006〜T-008）。

本モジュール（T-006）は cut なし・棚なしの直方体ケースのみを対象とする。
cut plane を用いた floor_z/ceil_z の変形、shelf 障害物の構築、`cells_of_aabb`、
`bake_placed` は後続チケット（T-007/T-008）で追加する。
"""
from dataclasses import dataclass

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM
from src.packing_core.types import Vec3


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
        shelf_boxes: 棚＝障害物AABB（相対）のリスト。棚なしでは空（T-006では常に空）。
        cell: 絞り込み格子のセル一辺の長さ [m]（`GridParams.cell`）。
        floor_z: 格子セルごとの床高さ。shape (nx, ny), float64。
        ceil_z: 格子セルごとの天井高さ。shape (nx, ny), float64。
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


def build_container_space(cdict: dict, index: int, spacing: float, cell: float) -> ContainerSpace:
    """`container_list[i]` の辞書から有効空間と絞り込み格子を構築する（cut・棚なし）。

    `cdict["points"]`（世界座標の代表点）と `cdict["n_vecs"]`（外向き単位法線）を正として
    内壁形状を復元する。`buffer` キーは `cdict` に存在しないため参照しない
    （`docs/interface_notes.md` §I-7）。

    Args:
        cdict: `container_list` の1要素。`constants.OBS_KEYS["container"]` のキーを持つ。
        index: コンテナ index。
        spacing: コンテナ間隔 [m]（`init_states` から取得済みの値）。
        cell: 絞り込み格子のセル一辺の長さ [m]（`GridParams.cell`）。

    Returns:
        `ContainerSpace`。cut なしの入力では `cut_planes` は空リストになる。

    Raises:
        ValueError: 軸整列面が6面（±X/±Y/±Z）揃わず内壁AABBが確定しない場合、
            または内壁AABBの一辺が0以下になる場合。
    """
    offset_x = index * spacing
    origin_world = np.array([offset_x, 0.0, 0.0], dtype=np.float64)

    inner_min_rel = np.full(3, -np.inf, dtype=np.float64)
    inner_max_rel = np.full(3, np.inf, dtype=np.float64)
    cut_planes: list = []

    for point_world, normal_world in zip(cdict["points"], cdict["n_vecs"]):
        point_rel = np.asarray(point_world, dtype=np.float64) - origin_world
        normal_rel = np.asarray(normal_world, dtype=np.float64)
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

    nx = max(1, round(size[0] / cell))
    ny = max(1, round(size[1] / cell))

    floor_z = np.full((nx, ny), inner_min_rel[2], dtype=np.float64)
    ceil_z = np.full((nx, ny), inner_max_rel[2], dtype=np.float64)
    height = floor_z.copy()

    return ContainerSpace(
        index=index,
        offset_x=float(offset_x),
        inner_min_rel=inner_min_rel,
        inner_max_rel=inner_max_rel,
        cut_planes=cut_planes,
        shelf_boxes=[],
        cell=float(cell),
        floor_z=floor_z,
        ceil_z=ceil_z,
        height=height,
    )


def contains_oriented_box(
    space: ContainerSpace, center_rel: Vec3, osize: Vec3, margin: float
) -> bool:
    """回転後の箱が有効空間に完全に収まるかを判定する（cut・棚なし）。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        center_rel: 箱の中心（コンテナ相対）。shape (3,), float64。
        osize: 回転後寸法。shape (3,), float64。
        margin: 内壁AABBを縮める量 [m]（`geometry.aabb_contains` と同じ符号規約）。

    Returns:
        内壁AABBに収まっていれば True。
    """
    box_min, box_max = geometry.aabb_from_center(center_rel, osize)
    return geometry.aabb_contains(
        space.inner_min_rel, space.inner_max_rel, box_min, box_max, margin=margin
    )


def effective_volume(space: ContainerSpace) -> float:
    """格子積分による有効体積の近似値を返す。

    XY方向は `space.cell` 四方のセルに離散化し、Z方向は `floor_z`/`ceil_z` の連続値を
    そのまま積分する（モンテカルロ不可、詳細仕様書 §4.2）。fill スコアの分母は
    公式 Evaluator 移植値（`cdict["volume"]`）を正とし、本関数の戻り値はそれに対する
    近似（絞り込み・特徴量計算用）として扱う。

    Args:
        space: 対象コンテナの `ContainerSpace`。

    Returns:
        近似体積 [m^3]。
    """
    return float(np.sum(space.ceil_z - space.floor_z) * space.cell * space.cell)
