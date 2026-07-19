"""T-007 golden fixture・独立期待値計算（docs/interface_notes.md §K, 実装詳細仕様書.md §4.2）。

Fixture A/B の `points`/`n_vecs` は公式 `ground_handling/utils.py::write_open_cut_corner_cup_obj`
（PyBullet非依存の純関数）と公式 `aff()` を直接呼び出して取得する（読解のみ・コピー禁止の対象は
`containers.py`/`utils.py` 本体であり、本モジュールはそれらを呼び出すだけで内容は変更しない）。

このモジュールは `src.packing_core.container_space`（`build_container_space`/
`contains_oriented_box`/`effective_volume`/非公開関数）を一切 import しない。期待値はすべて
`interface_notes.md` §K の式をこのモジュール内で独立に再実装して計算する。
"""
import math
import os
import tempfile

import numpy as np

from src.ground_handling.utils import aff, write_open_cut_corner_cup_obj
from src.packing_core.constants import EPS_GEOM, GridParams

CELL = GridParams().cell  # 0.02

# --- Fixture A / B: 公式cut形状（cut_x=cut_y=0.20, shelf=False/True） -----------------

FIXTURE_AB_LENGTH = 1.00
FIXTURE_AB_WIDTH = 1.00
FIXTURE_AB_HEIGHT = 1.00
FIXTURE_AB_THICKNESS = 0.02
FIXTURE_AB_CUT_X = 0.20
FIXTURE_AB_CUT_Y = 0.20
FIXTURE_AB_BUFFER = 0.02


def _official_points_n_vecs(length, height, width, thickness, cut_x, cut_y):
    """公式 `write_open_cut_corner_cup_obj` を直接呼び出す（PyBulletクライアント不要）。"""
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "container.obj")
        points, n_vecs = write_open_cut_corner_cup_obj(
            fname,
            width=length,
            height=height,
            cut_x=cut_x,
            cut_y=cut_y,
            depth=width,
            wall=thickness,
            bottom=thickness,
        )
    return points, n_vecs


def _container_relative_points_n_vecs(length, height, width, thickness, cut_x, cut_y, buffer):
    """`containers.py::create()` と同じ回転・並進（公式 `aff()` を使用）でコンテナ相対座標へ変換する。"""
    points, n_vecs = _official_points_n_vecs(length, height, width, thickness, cut_x, cut_y)
    rot = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(math.pi / 2), -math.sin(math.pi / 2)],
        [0.0, math.sin(math.pi / 2), math.cos(math.pi / 2)],
    ]
    pos = (0.0, 0.0, height / 2.0 + buffer)
    rel_points = aff(points, rot, pos)
    rel_n_vecs = aff(n_vecs, rot, intercept=(0.0, 0.0, 0.0))
    return rel_points, rel_n_vecs


def build_cdict_from_raw_config(raw_config: dict, offset_x: float, index: int = 0) -> dict:
    """任意の `container_raw_config`（T-016A l_path_golden_v1 の `length`/`width`/`height`/
    `thickness`/`cut_x`/`cut_y`/`buffer`/`require_shelf`）から cdict を構築する
    （T-016Bゴールデン照合テスト専用。`build_fixture_ab_cdict` の一般化）。

    `points`/`n_vecs` は公式 `write_open_cut_corner_cup_obj`/`aff` の直接呼び出しにより
    取得する（`container_space.py` は未使用）。ゴールデンの `container_raw_config` は
    `cut_x`/`cut_y` を常に正で保持するため（`safety_margin_boundary`/`random_scene` を含む
    全カテゴリで生成時に保証済み）、公式関数の `cut_x<=0` ガードには抵触しない。
    """
    length = float(raw_config["length"])
    width = float(raw_config["width"])
    height = float(raw_config["height"])
    thickness = float(raw_config["thickness"])
    cut_x = float(raw_config["cut_x"])
    cut_y = float(raw_config["cut_y"])
    buffer = float(raw_config["buffer"])
    shelf = bool(raw_config["require_shelf"])

    rel_points, rel_n_vecs = _container_relative_points_n_vecs(
        length, height, width, thickness, cut_x, cut_y, buffer
    )
    points_world = [(px + offset_x, py, pz) for px, py, pz in rel_points]

    return {
        "index": index,
        "length": length,
        "width": width,
        "height": height,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "thickness": thickness,
        "center": (offset_x, 0.0, height / 2.0 + buffer),
        "n_vecs": rel_n_vecs,
        "points": points_world,
        "volume": official_volume_formula(
            length, width, height, thickness, cut_x, cut_y, buffer, shelf
        ),
        "shelf": shelf,
        "is_prioritized": False,
        "packed_items": [],
    }


def build_fixture_ab_cdict(index: int = 0, spacing: float = 2.0, shelf: bool = False) -> dict:
    """Fixture A（shelf=False）／ Fixture B（shelf=True）の cdict を構築する。

    `points`/`n_vecs` は公式関数の直接呼び出しにより取得する（`container_space.py` は未使用）。
    """
    length, height, width = FIXTURE_AB_LENGTH, FIXTURE_AB_HEIGHT, FIXTURE_AB_WIDTH
    thickness, cut_x, cut_y, buffer = (
        FIXTURE_AB_THICKNESS,
        FIXTURE_AB_CUT_X,
        FIXTURE_AB_CUT_Y,
        FIXTURE_AB_BUFFER,
    )
    rel_points, rel_n_vecs = _container_relative_points_n_vecs(
        length, height, width, thickness, cut_x, cut_y, buffer
    )
    offset_x = index * spacing
    points_world = [(px + offset_x, py, pz) for px, py, pz in rel_points]

    return {
        "index": index,
        "length": length,
        "width": width,
        "height": height,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "thickness": thickness,
        "center": (offset_x, 0.0, height / 2.0 + buffer),
        "n_vecs": rel_n_vecs,
        "points": points_world,
        "volume": official_volume_formula(
            length, width, height, thickness, cut_x, cut_y, buffer, shelf
        ),
        "shelf": shelf,
        "is_prioritized": False,
        "packed_items": [],
    }


# --- 公式体積式（interface_notes.md §D、診断値専用。effective_volumeのDoDには使わない） -----

def official_volume_formula(
    length: float,
    width: float,
    height: float,
    thickness: float,
    cut_x: float,
    cut_y: float,
    buffer: float,
    shelf: bool,
) -> float:
    """`containers.py::create()` の有効体積式（base - cut - small_shelf - shelf）を独立再実装する。"""
    inner_length = length - 2 * thickness
    inner_width = width - 2 * thickness
    inner_height = height - thickness - buffer
    base_volume = inner_length * inner_width * inner_height
    cut_volume = 0.5 * (cut_x - thickness) * (cut_y - thickness) * inner_width
    small_shelf_volume = cut_x * thickness * inner_width
    shelf_volume = 0.0
    if shelf:
        shelf_width = width / 2.0 - 2 * thickness
        shelf_volume = inner_length * thickness * shelf_width
    return base_volume - cut_volume - small_shelf_volume - shelf_volume


# --- cut plane構築の一般化（interface_notes.md §K.6、T-006ロジックとは独立実装） -------------

def classify_planes(points, n_vecs, origin_world=(0.0, 0.0, 0.0)):
    """`(points, n_vecs)` を軸整列面／非軸整列面（cut_planes）に分類する。

    `container_space.py::_axis_alignment`/`build_container_space` とは独立の実装。
    """
    origin = np.asarray(origin_world, dtype=np.float64)
    inner_min = np.full(3, -np.inf, dtype=np.float64)
    inner_max = np.full(3, np.inf, dtype=np.float64)
    cut_planes = []
    for point_world, normal_world in zip(points, n_vecs):
        point_rel = np.asarray(point_world, dtype=np.float64) - origin
        normal_rel = np.asarray(normal_world, dtype=np.float64)
        d = float(np.dot(normal_rel, point_rel))

        axis = None
        for ax in range(3):
            others = [k for k in range(3) if k != ax]
            if abs(abs(normal_rel[ax]) - 1.0) < EPS_GEOM and all(
                abs(normal_rel[k]) < EPS_GEOM for k in others
            ):
                axis = (ax, float(np.sign(normal_rel[ax])))
                break

        if axis is None:
            cut_planes.append((normal_rel, d))
            continue

        ax, sign = axis
        if sign > 0:
            inner_max[ax] = min(inner_max[ax], d)
        else:
            inner_min[ax] = max(inner_min[ax], -d)

    return inner_min, inner_max, cut_planes


# --- 小棚・大棚AABB（interface_notes.md §K.2/K.3/K.4） ---------------------------------------

def expected_small_shelf_raw(length, width, height, thickness, cut_x, buffer):
    center = np.array(
        [-length / 2.0 + cut_x / 2.0 + thickness, 0.0, height / 2.0 + thickness / 2.0 + buffer],
        dtype=np.float64,
    )
    half = np.array([cut_x / 2.0, width / 2.0 - thickness, thickness / 2.0], dtype=np.float64)
    return center - half, center + half


def expected_main_shelf_raw(length, width, height, thickness, buffer):
    center = np.array(
        [0.0, width / 4.0, height / 2.0 + thickness / 2.0 + buffer], dtype=np.float64
    )
    half = np.array(
        [length / 2.0 - thickness / 2.0, width / 4.0 - thickness, thickness / 2.0],
        dtype=np.float64,
    )
    return center - half, center + half


def clip_and_filter(raw_min, raw_max, inner_min, inner_max):
    """内壁AABBへクリップし、全軸長が `EPS_GEOM` を超える場合のみ (min, max) を返す。"""
    clip_min = np.maximum(raw_min, inner_min)
    clip_max = np.minimum(raw_max, inner_max)
    extents = clip_max - clip_min
    if np.all(extents > EPS_GEOM):
        return clip_min, clip_max
    return None


def expected_shelf_boxes(length, width, height, thickness, cut_x, buffer, shelf, inner_min, inner_max):
    """小棚（常時計算）・大棚（`shelf=True`のみ）のクリップ後AABBリストを返す。"""
    boxes = []

    small_raw_min, small_raw_max = expected_small_shelf_raw(length, width, height, thickness, cut_x, buffer)
    small_clipped = clip_and_filter(small_raw_min, small_raw_max, inner_min, inner_max)
    if small_clipped is not None:
        boxes.append(small_clipped)

    if shelf:
        main_raw_min, main_raw_max = expected_main_shelf_raw(length, width, height, thickness, buffer)
        main_clipped = clip_and_filter(main_raw_min, main_raw_max, inner_min, inner_max)
        if main_clipped is not None:
            boxes.append(main_clipped)

    return boxes


# --- セル格子・floor_z/ceil_z（interface_notes.md §K.7、セル中心規約） ------------------------

def cell_grid(inner_min, inner_max, cell):
    size = inner_max - inner_min
    nx = max(1, round(size[0] / cell))
    ny = max(1, round(size[1] / cell))
    xs = inner_min[0] + (np.arange(nx) + 0.5) * cell
    ys = inner_min[1] + (np.arange(ny) + 0.5) * cell
    return nx, ny, xs, ys


def expected_floor_ceil(inner_min, inner_max, cut_planes, shelf_boxes, cell):
    nx, ny, xs, ys = cell_grid(inner_min, inner_max, cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    floor_z = np.full((nx, ny), inner_min[2], dtype=np.float64)
    ceil_z = np.full((nx, ny), inner_max[2], dtype=np.float64)

    for normal_rel, d in cut_planes:
        nx_, ny_, nz_ = normal_rel
        if nz_ < -EPS_GEOM:
            candidate = (d - nx_ * X - ny_ * Y) / nz_
            floor_z = np.maximum(floor_z, candidate)
        elif nz_ > EPS_GEOM:
            candidate = (d - nx_ * X - ny_ * Y) / nz_
            ceil_z = np.minimum(ceil_z, candidate)
        # |nz_| <= EPS_GEOM: z を拘束しないため寄与させない（§K.8）

    for box_min, box_max in shelf_boxes:
        covers = (X >= box_min[0]) & (X <= box_max[0]) & (Y >= box_min[1]) & (Y <= box_max[1])
        ceil_z = np.where(covers, np.minimum(ceil_z, box_min[2]), ceil_z)

    return floor_z, ceil_z


def expected_grid_volume(floor_z, ceil_z, cell):
    return float(np.sum(np.maximum(ceil_z - floor_z, 0.0)) * cell * cell)


def expected_analytic_box_volume(inner_min, inner_max):
    return float(np.prod(inner_max - inner_min))
