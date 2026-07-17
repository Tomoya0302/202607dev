"""T-009: packing_core.ems の全再構築（generate_ems）検証
（詳細仕様書 §2・§3・§4.3・§6 T-009。依存: T-005 geometry AABB系、T-007 cut/棚対応）。

ems.py は本チケット時点では未実装のため、収集(--collect-only)を成功させるべく
各テスト関数の内部で対象モジュールを import する（モジュール直下 import は禁止、
test_geometry.py・test_container_space.py と同方針）。

本ファイルは T-009（generate_ems の全再構築、既知3EMSケース）のみを対象とし、
T-010（update_ems 差分・一致性質テスト）・T-011（select_topn/normalize_descriptors/
打切り率）のテストは含めない。
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.container_space import build_container_space

ROUND_NDIGITS = 6

# 内壁寸法 1.5 x 2.0 x 1.6（cut・棚なし）。座標は inner_min を原点として表記する
# （実装詳細仕様書.md §4.3 のテスト例と同一）。
INNER_MIN_REL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
INNER_MAX_REL = np.array([1.5, 2.0, 1.6], dtype=np.float64)
CELL = constants.GridParams().cell


def _empty_cdict(inner_min_rel: np.ndarray, inner_max_rel: np.ndarray) -> dict:
    """cutなし・棚なしの軸整列直方体コンテナの最小cdict（本ファイル専用、他所非依存）。

    `build_container_space` は `points`/`n_vecs`（6面の代表点・外向き法線）の半空間交差
    から内壁AABBを復元するため、6面すべて軸整列に設定すれば `cut_planes` は空になる。
    `cut_x=0.0` により小棚のクリップ後体積は常に0になるため `shelf_boxes` も空になる
    （`container_space._small_shelf_raw_aabb` の半径が `cut_x/2` のため）。
    """
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
    points = [(px, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]

    thickness = 0.02
    buffer = 0.02
    height = float(imax[2]) + buffer

    return {
        "index": 0,
        "length": float(imax[0] - imin[0]) + 2 * thickness,
        "width": float(imax[1] - imin[1]) + 2 * thickness,
        "height": height,
        "cut_x": 0.0,
        "cut_y": 0.0,
        "thickness": thickness,
        "center": (0.0, 0.0, height / 2.0 + buffer),
        "n_vecs": n_vecs,
        "points": points,
        "volume": float(np.prod(imax - imin)),
        "shelf": False,
        "is_prioritized": False,
        "packed_items": [],
    }


def _empty_space(inner_min_rel: np.ndarray = INNER_MIN_REL, inner_max_rel: np.ndarray = INNER_MAX_REL):
    """cut・棚なしの `ContainerSpace` を構築する（`build_container_space` 経由）。"""
    cdict = _empty_cdict(inner_min_rel, inner_max_rel)
    return build_container_space(cdict, index=0, spacing=2.0, cell=CELL)


def _round_bounds(min_rel: np.ndarray, max_rel: np.ndarray, ndigits: int = ROUND_NDIGITS) -> tuple:
    """min_rel/max_rel を順序非依存比較用の丸め済みtupleへ変換する。"""
    min_r = tuple(round(float(v), ndigits) for v in min_rel)
    max_r = tuple(round(float(v), ndigits) for v in max_rel)
    return min_r + max_r


def _rounded_box_set(boxes, ndigits: int = ROUND_NDIGITS) -> set:
    return {_round_bounds(box.min_rel, box.max_rel, ndigits) for box in boxes}


# --- 空コンテナ: cut・棚なし ------------------------------------------------------

def test_generate_ems_empty_container_yields_full_inner_space():
    from src.packing_core import ems

    space = _empty_space()
    result = ems.generate_ems(space, placed_aabbs=[])

    assert len(result) == 1
    np.testing.assert_allclose(result[0].min_rel, INNER_MIN_REL, atol=1e-6)
    np.testing.assert_allclose(result[0].max_rel, INNER_MAX_REL, atol=1e-6)


# --- FLB角に0.5^3の箱を1個配置: EMSちょうど3個 -------------------------------------

def test_generate_ems_single_flb_corner_box_yields_three_ems():
    from src.packing_core import ems

    space = _empty_space()
    box_min = INNER_MIN_REL.copy()
    box_max = INNER_MIN_REL + np.array([0.5, 0.5, 0.5], dtype=np.float64)

    result = ems.generate_ems(space, placed_aabbs=[(box_min, box_max)])

    assert len(result) == 3

    expected = {
        # X側空間: x∈[0.5,1.5]、他軸は全範囲
        _round_bounds(
            np.array([0.5, INNER_MIN_REL[1], INNER_MIN_REL[2]]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
        # Y側空間: y∈[0.5,2.0]、他軸は全範囲
        _round_bounds(
            np.array([INNER_MIN_REL[0], 0.5, INNER_MIN_REL[2]]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
        # Z側空間: z∈[0.5,1.6]、他軸は全範囲
        _round_bounds(
            np.array([INNER_MIN_REL[0], INNER_MIN_REL[1], 0.5]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
    }

    assert _rounded_box_set(result) == expected


# --- 棚(shelf_boxes)も障害物として扱われる ------------------------------------------

def test_generate_ems_treats_shelf_as_obstacle():
    """generate_ems は placed_aabbs だけでなく space.shelf_boxes も障害物として処理する
    （詳細仕様書 §4.3: 「障害物（placed + shelf）ごとに update_ems を適用」）。

    同じAABBを shelf_boxes 経由と placed_aabbs 経由で与えた場合、結果のEMS集合が
    一致するはず。
    """
    from src.packing_core import ems

    box_min = INNER_MIN_REL.copy()
    box_max = INNER_MIN_REL + np.array([0.5, 0.5, 0.5], dtype=np.float64)

    space_via_shelf = _empty_space()
    space_via_shelf.shelf_boxes = [(box_min, box_max)]
    result_via_shelf = ems.generate_ems(space_via_shelf, placed_aabbs=[])

    space_via_placed = _empty_space()
    result_via_placed = ems.generate_ems(space_via_placed, placed_aabbs=[(box_min, box_max)])

    assert len(result_via_shelf) == len(result_via_placed) == 3
    assert _rounded_box_set(result_via_shelf) == _rounded_box_set(result_via_placed)
