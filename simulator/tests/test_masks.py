"""T-014: packing_core.masks の check_inclusion／check_ceiling 検証
（詳細仕様書 §3.5・§3.6・§4.5・§6 T-014、更新履歴 v1.9）。

masks.py は本チケット時点では未実装のため、収集(--collect-only)を成功させるべく
各テスト関数の内部で対象モジュールを import する（モジュール直下 import は禁止、
test_geometry.py・test_container_space.py と同方針）。

確定した判定式（v1.9、§4.5）:
    check_inclusion: contains_oriented_box(margin = -pp.inclusion_margin + pp.internal_extra)
        （符号はA14に準拠：inclusion_margin は負で厳格・正で緩和）
    check_ceiling:
        box_top = cand.pos_rel[2] + cand.osize[2] / 2.0
        required_clearance = pp.ceiling_margin + pp.internal_extra
        local_ceiling_z = min(内壁上面, 候補とXY投影が正の面積で重なりかつ
            棚下面が候補上端以上（shelf_min_z >= box_top - EPS_GEOM）である棚の下面の最小)
        （cut_planes は対象外で check_inclusion に委譲。格子 ceil_z は最終判定に使わない）
        return box_top + required_clearance <= local_ceiling_z + EPS_GEOM
"""
import numpy as np
import pytest

from fixtures import container_space_golden as golden
from src.packing_core import constants

PP0 = constants.PlacementParams()  # inclusion_margin=-0.005, internal_extra=0.005,
                                    # ceiling_margin=0.018, start_z=0.08
DEFAULT_OSIZE = np.array([0.2, 0.2, 0.2], dtype=np.float64)


def _box_space(inner_min, inner_max, cell):
    """cut無・棚無の軸整列直方体 space（inclusion 全般／ceiling 無棚ケース用）。"""
    from src.packing_core.container_space import ContainerSpace

    imin = np.asarray(inner_min, dtype=np.float64)
    imax = np.asarray(inner_max, dtype=np.float64)
    size = imax - imin
    nx = max(1, round(size[0] / cell))
    ny = max(1, round(size[1] / cell))
    floor_z = np.full((nx, ny), imin[2], dtype=np.float64)
    ceil_z = np.full((nx, ny), imax[2], dtype=np.float64)
    return ContainerSpace(
        index=0,
        offset_x=0.0,
        inner_min_rel=imin,
        inner_max_rel=imax,
        cut_planes=[],
        shelf_boxes=[],
        cell=float(cell),
        floor_z=floor_z,
        ceil_z=ceil_z,
        height=floor_z.copy(),
    )


def _unit_cube_space():
    return _box_space([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], constants.GridParams().cell)


def _make_candidate(pos_rel, osize):
    from src.packing_core.types import Candidate

    return Candidate(
        item_idx=0,
        container_idx=0,
        ems_id=0,
        orientation=0,
        pos_rel=np.asarray(pos_rel, dtype=np.float64),
        osize=np.asarray(osize, dtype=np.float64),
    )


def _select_main_shelf(shelf_boxes):
    """XY投影面積が最大の棚を選ぶ（決定的選択。golden fixtureでは主棚に対応）。"""

    def _xy_area(box):
        smin, smax = box
        return float((smax[0] - smin[0]) * (smax[1] - smin[1]))

    return max(shelf_boxes, key=_xy_area)


# --- check_inclusion --------------------------------------------------------------


def test_inclusion_fully_inside_passes():
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is True


def test_inclusion_at_effective_margin_boundary_passes():
    from src.packing_core.masks import check_inclusion

    # 有効margin = -(-0.005)+0.005 = 0.010。x上端 = 0.89+0.1 = 0.99 = 1.0-0.010（境界）。
    space = _unit_cube_space()
    cand = _make_candidate([0.89, 0.5, 0.5], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is True


def test_inclusion_touching_raw_wall_rejected():
    from src.packing_core.masks import check_inclusion

    # x上端 = 0.90+0.1 = 1.00（内壁ぴったり）。有効margin境界 0.990 を超えるため不合格。
    space = _unit_cube_space()
    cand = _make_candidate([0.90, 0.5, 0.5], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is False


def test_inclusion_protrusion_x_rejected():
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    cand = _make_candidate([0.91, 0.5, 0.5], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is False


def test_inclusion_protrusion_y_rejected():
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.91, 0.5], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is False


def test_inclusion_below_floor_rejected():
    from src.packing_core.masks import check_inclusion

    # z下端 = 0.09-0.1 = -0.01（床下1cm）。
    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.09], DEFAULT_OSIZE)

    assert check_inclusion(space, cand, PP0) is False


def test_inclusion_margin_sign_semantics():
    """A14準拠の符号（負=厳格・正=緩和）とinternal_extraの厳格化方向を検証する。"""
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    # x上端 = 0.87+0.1 = 0.97（内側30mm）。
    cand = _make_candidate([0.87, 0.5, 0.5], DEFAULT_OSIZE)

    # 既定（有効margin=0.010）では合格する（対比の基準点）。
    assert check_inclusion(space, cand, PP0) is True

    # inclusion_margin をより負にする→厳格化（有効margin=0.055、要x上端<=0.945）→不合格。
    pp_deeper_negative = constants.PlacementParams(inclusion_margin=-0.05, internal_extra=0.005)
    assert check_inclusion(space, cand, pp_deeper_negative) is False

    # internal_extra を増やす→厳格化（有効margin=0.055、同上）→不合格。
    pp_extra_up = constants.PlacementParams(inclusion_margin=-0.005, internal_extra=0.05)
    assert check_inclusion(space, cand, pp_extra_up) is False

    # inclusion_margin を正にする→緩和（有効margin=-0.02、10mmはみ出しまで許容）→合格。
    pp_relaxed = constants.PlacementParams(inclusion_margin=0.02, internal_extra=0.0)
    cand_protruding = _make_candidate([0.91, 0.5, 0.5], DEFAULT_OSIZE)  # x上端=1.01
    assert check_inclusion(space, cand_protruding, pp_relaxed) is True


def test_inclusion_matches_contains_oriented_box():
    """check_inclusion は margin=-inclusion_margin+internal_extra の委譲であること（同値性）。"""
    from src.packing_core.container_space import contains_oriented_box
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    cases = [
        ([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], PP0),
        ([0.89, 0.5, 0.5], [0.2, 0.2, 0.2], PP0),
        ([0.91, 0.5, 0.5], [0.2, 0.2, 0.2], PP0),
        ([0.5, 0.5, 0.09], [0.2, 0.2, 0.2], PP0),
        ([0.3, 0.7, 0.4], [0.1, 0.3, 0.5], constants.PlacementParams(inclusion_margin=-0.05)),
        ([0.91, 0.5, 0.5], [0.2, 0.2, 0.2], constants.PlacementParams(inclusion_margin=0.02, internal_extra=0.0)),
    ]

    for pos_rel, osize, pp in cases:
        cand = _make_candidate(pos_rel, osize)
        expected = contains_oriented_box(
            space, cand.pos_rel, cand.osize, margin=-pp.inclusion_margin + pp.internal_extra
        )
        assert check_inclusion(space, cand, pp) == expected


def test_inclusion_does_not_mutate_inputs():
    from src.packing_core.masks import check_inclusion

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE)

    pos_rel_before = cand.pos_rel.copy()
    osize_before = cand.osize.copy()
    inner_min_before = space.inner_min_rel.copy()
    inner_max_before = space.inner_max_rel.copy()
    floor_z_before = space.floor_z.copy()
    ceil_z_before = space.ceil_z.copy()
    height_before = space.height.copy()

    result = check_inclusion(space, cand, PP0)

    assert isinstance(result, bool)
    assert np.array_equal(cand.pos_rel, pos_rel_before)
    assert np.array_equal(cand.osize, osize_before)
    assert np.array_equal(space.inner_min_rel, inner_min_before)
    assert np.array_equal(space.inner_max_rel, inner_max_before)
    assert np.array_equal(space.floor_z, floor_z_before)
    assert np.array_equal(space.ceil_z, ceil_z_before)
    assert np.array_equal(space.height, height_before)


# --- check_ceiling -----------------------------------------------------------------


def test_ceiling_ample_clearance_passes():
    from src.packing_core.masks import check_ceiling

    # box_top=0.6、local_ceiling(無棚)=1.0、必要余裕=0.023 → 0.623<=1.0。
    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE)

    assert check_ceiling(space, cand, PP0) is True


def test_ceiling_at_margin_boundary_passes():
    from src.packing_core.masks import check_ceiling

    # box_top=0.977、0.977+0.023=1.000=local_ceiling（境界、等号=合格）。
    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.877], DEFAULT_OSIZE)

    assert check_ceiling(space, cand, PP0) is True


def test_ceiling_slight_violation_rejected():
    from src.packing_core.masks import check_ceiling

    # box_top=0.978、0.978+0.023=1.001>1.0。
    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.878], DEFAULT_OSIZE)

    assert check_ceiling(space, cand, PP0) is False


def test_ceiling_internal_extra_tightens():
    from src.packing_core.masks import check_ceiling

    # box_top=0.95。既定(必要余裕0.023)では0.973<=1.0で合格。
    # internal_extra=0.05に増やすと必要余裕0.068、1.018>1.0で不合格に転じる。
    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.85], DEFAULT_OSIZE)

    assert check_ceiling(space, cand, PP0) is True

    pp_extra_up = constants.PlacementParams(ceiling_margin=0.018, internal_extra=0.05)
    assert check_ceiling(space, cand, pp_extra_up) is False


def test_ceiling_shelf_underside_is_local_ceiling():
    """候補とXY投影が正の面積で重なり、棚下面が候補上端以上（上方の棚）なら局所天井は棚下面。"""
    from src.packing_core.container_space import build_container_space
    from src.packing_core.masks import check_ceiling

    cdict = golden.build_fixture_ab_cdict(shelf=True)
    space = build_container_space(cdict, index=0, cell=golden.CELL)
    shelf_min, shelf_max = _select_main_shelf(space.shelf_boxes)
    shelf_min_z = float(shelf_min[2])

    center_xy = (shelf_min[:2] + shelf_max[:2]) / 2.0
    osize = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    # XY正面積重なりを保証（棚フットプリント内に候補フットプリントを収める）。
    assert np.all(shelf_max[:2] - shelf_min[:2] > osize[:2])

    required_clearance = PP0.ceiling_margin + PP0.internal_extra

    box_top_pass = shelf_min_z - required_clearance - 1e-4
    cand_pass = _make_candidate(
        [center_xy[0], center_xy[1], box_top_pass - osize[2] / 2.0], osize
    )
    assert check_ceiling(space, cand_pass, PP0) is True

    box_top_fail = shelf_min_z - required_clearance + 1e-4
    cand_fail = _make_candidate(
        [center_xy[0], center_xy[1], box_top_fail - osize[2] / 2.0], osize
    )
    assert check_ceiling(space, cand_fail, PP0) is False


def test_ceiling_shelf_below_candidate_is_not_ceiling():
    """棚とXY投影は重なるが棚下面が候補上端未満（下方の棚）なら天井扱いしない。"""
    from src.packing_core.container_space import build_container_space
    from src.packing_core.masks import check_ceiling

    cdict = golden.build_fixture_ab_cdict(shelf=True)
    space = build_container_space(cdict, index=0, cell=golden.CELL)
    shelf_min, shelf_max = _select_main_shelf(space.shelf_boxes)
    shelf_min_z = float(shelf_min[2])
    inner_top = float(space.inner_max_rel[2])

    required_clearance = PP0.ceiling_margin + PP0.internal_extra
    gap = inner_top - shelf_min_z
    assert gap >= 2 * required_clearance + 1e-3  # 十分な余白がある前提（golden fixtureで成立）

    center_xy = (shelf_min[:2] + shelf_max[:2]) / 2.0
    osize = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    assert np.all(shelf_max[:2] - shelf_min[:2] > osize[:2])

    box_top = shelf_min_z + gap / 2.0  # 候補上端 > 棚下面（棚は候補より下方）
    assert box_top > shelf_min_z

    cand = _make_candidate([center_xy[0], center_xy[1], box_top - osize[2] / 2.0], osize)

    # 下方の棚は無視され、局所天井は内壁上面のまま（box_top+required_clearance<=inner_top）。
    assert check_ceiling(space, cand, PP0) is True


def test_ceiling_returns_bool_and_does_not_mutate_inputs():
    from src.packing_core.masks import check_ceiling

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.3], DEFAULT_OSIZE)

    pos_rel_before = cand.pos_rel.copy()
    osize_before = cand.osize.copy()
    inner_min_before = space.inner_min_rel.copy()
    inner_max_before = space.inner_max_rel.copy()
    floor_z_before = space.floor_z.copy()
    ceil_z_before = space.ceil_z.copy()
    height_before = space.height.copy()

    result = check_ceiling(space, cand, PP0)

    assert isinstance(result, bool)
    assert np.array_equal(cand.pos_rel, pos_rel_before)
    assert np.array_equal(cand.osize, osize_before)
    assert np.array_equal(space.inner_min_rel, inner_min_before)
    assert np.array_equal(space.inner_max_rel, inner_max_before)
    assert np.array_equal(space.floor_z, floor_z_before)
    assert np.array_equal(space.ceil_z, ceil_z_before)
    assert np.array_equal(space.height, height_before)
