"""T-020: packing_core.stability の support_ratio／max_step_below／soft_below_ratio 検証
（詳細仕様書 §2・§3・§4.2・§4.4・§4.6・§6 T-020）。

stability.py は本チケット時点では未実装ではなく本セッションで実装するが、収集(--collect-only)を
安全に保つため、対象モジュール `src.packing_core.stability` の import は各テスト関数の内部で行う
（tests/test_geometry.py・tests/test_container_space.py・tests/test_masks.py と同方針）。

T-020 では `support_ratio`／`max_step_below`／`soft_below_ratio` の3関数のみを検証する。
`support_polygon`／`cg_margin`（T-021の責務、monotone chain・凸包）はここではテストしない。

確定した仕様（人間確認済み。詳細は本チケットの計画ファイル参照）:
    D1. max_step_below: 対象は `cells_of_aabb` が返す候補底面footprintの全セル（支持セルに
        限定しない）。値は `max(height) - min(height)`。bottom_zとの差でも隣接セル間差でもない。
    D2. soft_below_ratio の分子（ソフト支持セル）は次の3条件すべてを満たす支持セル：
        (1) is_soft=True の PlacedItem のXY投影がそのセル中心を覆う
        (2) その PlacedItem の aabb_max_rel[2] がそのセルの space.height と EPS_GEOM 以内で一致
        (3) そのセルが support_ratio と同じ支持セル条件を満たす
        同一高さにソフト・ハードが並存する退化ケースは、ソフトが1つでもあれば保守側に数える。
    D3. soft_below_ratio の分母は support_ratio と同じ全支持セル（床支持セルも含む）。
        支持セル0件は0.0を返す。

fixture: 軸整列直方体コンテナ（cut_x=cut_y=0、shelf=False）。
    inner_min_rel=[-0.40,-0.40,0.02], inner_max_rel=[0.40,0.40,1.02], cell=0.02（nx=ny=40）。
    候補は中心原点・osize=(0.4,0.4,0.3) を既定とし、footprintは20x20=400セル
    （x/y方向とも候補境界 -0.2/0.2 はセル中心に一致しないため端数なく厳密に20セルずつ選ばれる）。
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.container_space import bake_placed, build_container_space
from src.packing_core.state import PackingState
from src.packing_core.types import Candidate, PlacedItem

TOL_CONTACT = constants.TOL_CONTACT
EPS_GEOM = constants.EPS_GEOM

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
CELL = 0.02
FLOOR_Z = float(INNER_MIN_REL[2])  # = thickness、bake_placed前のheight初期値


def _box_cdict() -> dict:
    """軸整列6面（cut_x=cut_y=0、shelf=False）の合成 cdict を組み立てる。

    `tests/test_container_space.py::_box_cdict` と同方針（本ファイルはそちらを import せず
    自己完結させる）。offset_x=0（center[0]=0）とし、world座標=rel座標として扱えるようにする。
    """
    imin, imax = INNER_MIN_REL, INNER_MAX_REL
    mid = (imin + imax) / 2.0
    thickness = 0.02
    buffer = 0.02
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer  # ceil_z(=inner_max.z) = height - buffer と自己無矛盾
    assert abs(imin[2] - thickness) < 1e-12  # floor_z(=inner_min.z) = thickness と自己無矛盾

    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px, py, pz) for (px, py, pz), _ in face_specs]  # offset_x=0
    n_vecs = [n for _, n in face_specs]

    return {
        "index": 0,
        "length": length,
        "width": width,
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


def _flat_box_space():
    """テスト用の軸整列直方体 `ContainerSpace`（floor_z=0.02一様）を返す。"""
    return build_container_space(_box_cdict(), index=0, cell=CELL)


def _placed_item(
    xmin: float, xmax: float, ymin: float, ymax: float,
    z_bottom: float, z_top: float, is_soft: bool = False,
) -> PlacedItem:
    """テスト用の `PlacedItem`（AABBのみが本質。ゼロ厚は禁止=常に z_bottom < z_top）。"""
    assert z_bottom < z_top, "ゼロ厚のPlacedItemは使用しない"
    aabb_min = np.array([xmin, ymin, z_bottom], dtype=np.float64)
    aabb_max = np.array([xmax, ymax, z_top], dtype=np.float64)
    return PlacedItem(
        pos_world=(aabb_min + aabb_max) / 2.0,
        orn_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        size=aabb_max - aabb_min,
        weight=1.0,
        is_soft=is_soft,
        is_priority=False,
        aabb_min_rel=aabb_min,
        aabb_max_rel=aabb_max,
    )


def _state_with_placed(placed: list) -> PackingState:
    """`placed` を焼き込んだ1コンテナの `PackingState` を返す。"""
    space = _flat_box_space()
    bake_placed(space, placed)
    return PackingState(
        containers=[space],
        placed={0: placed},
        pool=[],
        ems={0: []},
        ems_truncation={0: 0.0},
        meta={},
    )


def _candidate(
    x: float = 0.0, y: float = 0.0, bottom_z: float = FLOOR_Z,
    osize=(0.4, 0.4, 0.3), container_idx: int = 0,
) -> Candidate:
    """テスト用の `Candidate`（`pos_rel[2]` は `bottom_z + osize[2]/2` から逆算）。"""
    osize_arr = np.asarray(osize, dtype=np.float64)
    pos_z = bottom_z + osize_arr[2] / 2.0
    return Candidate(
        item_idx=0,
        container_idx=container_idx,
        ems_id=0,
        orientation=0,
        pos_rel=np.array([x, y, pos_z], dtype=np.float64),
        osize=osize_arr,
    )


# --- support_ratio ---------------------------------------------------------

def test_support_ratio_flat_floor_is_one():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    assert support_ratio(state, cand) == pytest.approx(1.0)


def test_support_ratio_half_on_platform_is_half():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    assert support_ratio(state, cand) == pytest.approx(0.5)


def test_support_ratio_floating_is_zero():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=0.50)
    assert support_ratio(state, cand) == pytest.approx(0.0)


def test_support_ratio_outside_footprint_is_zero():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(x=10.0, bottom_z=FLOOR_Z)
    assert support_ratio(state, cand) == pytest.approx(0.0)


def test_support_ratio_tol_contact_boundary():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])

    inside_tol = _candidate(bottom_z=0.30 + 0.8 * TOL_CONTACT)
    assert support_ratio(state, inside_tol) == pytest.approx(1.0)

    outside_tol = _candidate(bottom_z=0.30 + 1.2 * TOL_CONTACT)
    assert support_ratio(state, outside_tol) == pytest.approx(0.0)


def test_support_ratio_return_type_is_float_and_bounded():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    result = support_ratio(state, _candidate(bottom_z=0.30))
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


# --- max_step_below ----------------------------------------------------------

def test_max_step_below_flat_is_zero():
    from src.packing_core.stability import max_step_below

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    assert max_step_below(state, cand) == pytest.approx(0.0)


def test_max_step_below_platform_step_includes_nonsupport():
    from src.packing_core.stability import max_step_below

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    # footprint内: 左半分 height=0.30、右半分 height=floor_z=0.02（非支持だが対象に含む）
    assert max_step_below(state, cand) == pytest.approx(0.30 - FLOOR_Z)


def test_max_step_below_empty_footprint_is_zero():
    from src.packing_core.stability import max_step_below

    state = _state_with_placed([])
    cand = _candidate(x=10.0, bottom_z=FLOOR_Z)
    assert max_step_below(state, cand) == pytest.approx(0.0)


def test_max_step_below_is_nonnegative_float():
    from src.packing_core.stability import max_step_below

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    result = max_step_below(state, _candidate(bottom_z=0.30))
    assert isinstance(result, float)
    assert result >= 0.0


# --- soft_below_ratio ----------------------------------------------------------

def test_soft_below_ratio_all_soft_is_one():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=0.30)
    assert soft_below_ratio(state, cand) == pytest.approx(1.0)


def test_soft_below_ratio_half_soft_half_hard_is_half():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(0.0, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    assert soft_below_ratio(state, cand) == pytest.approx(0.5)


def test_soft_below_ratio_half_floor_half_soft_is_half():
    from src.packing_core.stability import soft_below_ratio

    soft_top_z = FLOOR_Z + 0.8 * TOL_CONTACT
    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=soft_top_z, is_soft=True)
    # 右半分は床のまま（PlacedItemを置かない）。
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=soft_top_z)
    # 床は candidate_bottom_z との差 < TOL_CONTACT のため支持セルとなり分母に入るが、
    # ソフト最上面ではないため分子には入らない。
    assert soft_below_ratio(state, cand) == pytest.approx(0.5)


def test_soft_below_ratio_hard_only_is_zero():
    from src.packing_core.stability import soft_below_ratio

    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([hard])
    cand = _candidate(bottom_z=0.30)
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_no_support_is_zero():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=0.50)  # 支持セル0件
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_tie_soft_and_hard_counts_soft():
    from src.packing_core.stability import soft_below_ratio

    # §4.6の退化ケース: 同一最上面(z=0.30)にソフト・ハードが並存 → 保守側にソフトとして数える。
    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    assert soft_below_ratio(state, cand) == pytest.approx(1.0)


def test_soft_below_ratio_soft_below_top_not_counted():
    from src.packing_core.stability import soft_below_ratio

    # ソフト(z=[0.02,0.20])の上にハード(z=[0.20,0.30])を体積非重複で物理的に積み重ねる。
    # space.heightを形成する最上面はハードのため、ソフトは支持として数えない。
    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.20, is_soft=True)
    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.20, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_return_type_is_float_and_bounded():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(0.0, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    result = soft_below_ratio(state, _candidate(bottom_z=0.30))
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


# --- container_idx 検証（3公開関数共通の _footprint 契約。代表関数で検証） -----------------

def test_container_idx_negative_raises_value_error():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z, container_idx=-1)
    with pytest.raises(ValueError):
        support_ratio(state, cand)


def test_container_idx_out_of_range_raises_value_error():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z, container_idx=len(state.containers))
    with pytest.raises(ValueError):
        support_ratio(state, cand)
