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
        # T-016B: inclusion/ceiling/overlap 系テストは path_* を参照しないため、
        # 軸整列直方体（cut_x=0, shelf無し）相当の自己無矛盾な値で埋める。
        path_entry_y_rel=float(imin[1]),
        path_lane_x_min_geom_rel=float(imin[0]),
        path_lane_x_max_geom_rel=float(imax[0]),
        path_mid_resting_z_rel=float(imin[2]),
        path_mid_ceiling_z_rel=float(imax[2]),
        path_obstacle_boxes_rel=(),
    )


def _unit_cube_space():
    return _box_space([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], constants.GridParams().cell)


def _make_candidate(pos_rel, osize, container_idx=0, ems_id=0):
    from src.packing_core.types import Candidate

    return Candidate(
        item_idx=0,
        container_idx=container_idx,
        ems_id=ems_id,
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


# --- T-015: MaskStage / evaluate_stage / prefilter_dims / check_overlap 契約 --------------
#
# T-015 時点では対象APIが masks.py に未実装のため、本セクションの各テストは
# --collect-only では収集に成功しつつ、実行時は ImportError（未実装関数の import 失敗）で
# 失敗することを許容する（詳細仕様書 §6 T-015〜T-017責務境界、T-015開発フロー）。
# 本番コードは変更しない。


def _make_ems(min_rel, max_rel):
    from src.packing_core.types import EMSBox

    return EMSBox(
        min_rel=np.asarray(min_rel, dtype=np.float64),
        max_rel=np.asarray(max_rel, dtype=np.float64),
    )


def _make_placed(aabb_min, aabb_max):
    """check_overlap 用の最小 PlacedItem（aabb_min_rel/aabb_max_rel のみ意味を持つ）。"""
    from src.packing_core.types import PlacedItem

    aabb_min = np.asarray(aabb_min, dtype=np.float64)
    aabb_max = np.asarray(aabb_max, dtype=np.float64)
    return PlacedItem(
        pos_world=np.zeros(3, dtype=np.float64),
        orn_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        size=aabb_max - aabb_min,
        weight=1.0,
        is_soft=False,
        is_priority=False,
        aabb_min_rel=aabb_min,
        aabb_max_rel=aabb_max,
    )


def _make_state(containers, placed, ems):
    from src.packing_core.state import PackingState

    return PackingState(
        containers=containers,
        placed=placed,
        pool=[],
        ems=ems,
        ems_truncation={idx: 0.0 for idx in ems},
        meta={},
    )


# --- MaskStage --------------------------------------------------------------------------


def test_mask_stage_values():
    from enum import IntEnum

    from src.packing_core.masks import MaskStage

    assert issubclass(MaskStage, IntEnum)
    assert MaskStage.DIMS == 0
    assert MaskStage.INCLUSION == 1
    assert MaskStage.OVERLAP == 2
    assert MaskStage.CEILING == 3
    assert MaskStage.L_PATH == 4


# --- evaluate_stage: 単一段階ディスパッチ・引数契約 ----------------------------------------


def test_evaluate_stage_dispatches_single_stage_only(monkeypatch):
    """指定した1段階の判定関数だけが1回呼ばれ、他は呼ばれないこと。加えて、各判定関数へ
    渡される引数が §4.5 の公開API契約（プレースホルダ引数名は公開シグネチャに一致）どおり
    であること（呼び出し回数だけでなく引数も検証する）。"""
    from src.packing_core import masks
    from src.packing_core.masks import MaskStage

    space = _unit_cube_space()
    ems_box = _make_ems([0.0, 0.0, 0.0], [0.6, 0.6, 0.6])
    cand = _make_candidate([0.3, 0.3, 0.3], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state = _make_state([space], {0: []}, {0: [ems_box]})

    calls = {"dims": [], "inclusion": [], "overlap": [], "ceiling": []}

    # スパイの引数名は masks.py 公開API契約（§4.5）のシグネチャに一致させる。呼び出し側が
    # 位置引数／キーワード引数のどちらで呼んでも束縛される（内部実装構造は固定しない）。
    def spy_dims(cand, ems):
        calls["dims"].append((cand, ems))
        return True

    def spy_inclusion(space, cand, pp):
        calls["inclusion"].append((space, cand, pp))
        return True

    def spy_overlap(state, cand, tol):
        calls["overlap"].append((state, cand, tol))
        return True

    def spy_ceiling(space, cand, pp):
        calls["ceiling"].append((space, cand, pp))
        return True

    monkeypatch.setattr(masks, "prefilter_dims", spy_dims)
    monkeypatch.setattr(masks, "check_inclusion", spy_inclusion)
    monkeypatch.setattr(masks, "check_overlap", spy_overlap)
    monkeypatch.setattr(masks, "check_ceiling", spy_ceiling)

    masks.evaluate_stage(state, cand, PP0, MaskStage.DIMS)
    assert len(calls["dims"]) == 1
    assert len(calls["inclusion"]) == 0
    assert len(calls["overlap"]) == 0
    assert len(calls["ceiling"]) == 0
    dims_cand, dims_ems = calls["dims"][0]
    assert dims_cand is cand
    assert dims_ems is ems_box  # state.ems[cand.container_idx][cand.ems_id] を渡す契約
    for key in calls:
        calls[key].clear()

    masks.evaluate_stage(state, cand, PP0, MaskStage.INCLUSION)
    assert len(calls["inclusion"]) == 1
    assert len(calls["dims"]) == 0
    assert len(calls["overlap"]) == 0
    assert len(calls["ceiling"]) == 0
    incl_space, incl_cand, incl_pp = calls["inclusion"][0]
    assert incl_space is space  # state.containers[cand.container_idx]
    assert incl_cand is cand
    assert incl_pp is PP0
    for key in calls:
        calls[key].clear()

    masks.evaluate_stage(state, cand, PP0, MaskStage.OVERLAP)
    assert len(calls["overlap"]) == 1
    assert len(calls["dims"]) == 0
    assert len(calls["inclusion"]) == 0
    assert len(calls["ceiling"]) == 0
    ov_state, ov_cand, ov_tol = calls["overlap"][0]
    assert ov_state is state
    assert ov_cand is cand
    assert ov_tol == -PP0.internal_extra  # tol=-pp.internal_extra
    for key in calls:
        calls[key].clear()

    masks.evaluate_stage(state, cand, PP0, MaskStage.CEILING)
    assert len(calls["ceiling"]) == 1
    assert len(calls["dims"]) == 0
    assert len(calls["inclusion"]) == 0
    assert len(calls["overlap"]) == 0
    ceil_space, ceil_cand, ceil_pp = calls["ceiling"][0]
    assert ceil_space is space  # state.containers[cand.container_idx]
    assert ceil_cand is cand
    assert ceil_pp is PP0


def test_evaluate_stage_returns_same_candidate_object():
    from src.packing_core.masks import MaskStage, evaluate_stage

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state = _make_state(
        [space], {0: []}, {0: [_make_ems([0.0, 0.0, 0.0], [0.6, 0.6, 0.6])]}
    )

    result = evaluate_stage(state, cand, PP0, MaskStage.INCLUSION)

    assert result is cand


def test_evaluate_stage_pass_sets_feasible_true_and_empty_reason():
    from src.packing_core.masks import MaskStage, evaluate_stage

    space = _unit_cube_space()
    osize = DEFAULT_OSIZE.copy()

    # (stage, pos_rel, cand_osize, ems_list, placed_list)
    cases = [
        (MaskStage.DIMS, [0.5, 0.5, 0.5], osize, [_make_ems([0.0, 0.0, 0.0], list(osize))], []),
        (MaskStage.INCLUSION, [0.5, 0.5, 0.5], osize, [], []),
        (MaskStage.OVERLAP, [0.5, 0.5, 0.5], osize, [], []),
        (MaskStage.CEILING, [0.5, 0.5, 0.5], osize, [], []),
    ]

    for stage, pos_rel, cand_osize, ems_list, placed_list in cases:
        cand = _make_candidate(pos_rel, cand_osize, container_idx=0, ems_id=0)
        state = _make_state([space], {0: placed_list}, {0: ems_list})

        result = evaluate_stage(state, cand, PP0, stage)

        assert result is cand
        assert cand.feasible is True
        assert cand.reject_reason == ""


def test_evaluate_stage_fail_sets_reject_reason_per_stage():
    from src.packing_core.masks import MaskStage, evaluate_stage

    space = _unit_cube_space()

    # DIMS: x軸のみ osize が EMS 寸法を超える。
    cand_dims = _make_candidate([0.5, 0.5, 0.5], [0.3, 0.2, 0.2], container_idx=0, ems_id=0)
    state_dims = _make_state(
        [space], {0: []}, {0: [_make_ems([0.0, 0.0, 0.0], [0.2, 0.2, 0.2])]}
    )

    # INCLUSION: 1cm はみ出し（既存 test_inclusion_protrusion_x_rejected と同じ幾何）。
    cand_inclusion = _make_candidate([0.91, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state_inclusion = _make_state([space], {0: []}, {0: []})

    # OVERLAP: 既配置と全面的に重なる候補。
    placed = _make_placed([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
    cand_overlap = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state_overlap = _make_state([space], {0: [placed]}, {0: []})

    # CEILING: わずかな違反（既存 test_ceiling_slight_violation_rejected と同じ幾何）。
    cand_ceiling = _make_candidate([0.5, 0.5, 0.878], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state_ceiling = _make_state([space], {0: []}, {0: []})

    cases = [
        (MaskStage.DIMS, cand_dims, state_dims, "dims"),
        (MaskStage.INCLUSION, cand_inclusion, state_inclusion, "inclusion"),
        (MaskStage.OVERLAP, cand_overlap, state_overlap, "overlap"),
        (MaskStage.CEILING, cand_ceiling, state_ceiling, "ceiling"),
    ]

    for stage, cand, state, reason in cases:
        result = evaluate_stage(state, cand, PP0, stage)

        assert result is cand
        assert cand.feasible is False
        assert cand.reject_reason == reason


def test_evaluate_stage_invalid_stage_raises_value_error():
    from src.packing_core.masks import evaluate_stage

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state = _make_state([space], {0: []}, {0: []})

    with pytest.raises(ValueError):
        evaluate_stage(state, cand, PP0, 99)


def test_evaluate_stage_l_path_raises_not_implemented():
    """T-015時点では L_PATH の推測実装・暫定合格を行わず NotImplementedError を送出する
    （T-016で l_path_sweep_boxes/check_l_path を実装、T-017で evaluate_stage へ接続）。"""
    from src.packing_core.masks import MaskStage, evaluate_stage

    space = _unit_cube_space()
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)
    state = _make_state([space], {0: []}, {0: []})

    with pytest.raises(NotImplementedError):
        evaluate_stage(state, cand, PP0, MaskStage.L_PATH)


def test_evaluate_stage_dims_resolves_ems_via_ems_id():
    """DIMS段階が state.ems[cand.container_idx][cand.ems_id] のindex契約でEMSを解決する
    こと。複数コンテナ・複数EMSを用意し、別コンテナまたは別indexのEMSでは合格しない
    fixtureで、正しいindexの組み合わせでのみ合格することを確認する。"""
    from src.packing_core.masks import MaskStage, evaluate_stage

    space0 = _unit_cube_space()
    space1 = _unit_cube_space()
    small = _make_ems([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    big = _make_ems([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])

    # container 0: [small, small] / container 1: [small, big]。big は [1][1] のみに存在。
    ems = {0: [small, small], 1: [small, big]}
    placed = {0: [], 1: []}
    state = _make_state([space0, space1], placed, ems)

    osize = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    cand_correct = _make_candidate([0.5, 0.5, 0.5], osize, container_idx=1, ems_id=1)
    result_correct = evaluate_stage(state, cand_correct, PP0, MaskStage.DIMS)
    assert result_correct is cand_correct
    assert cand_correct.feasible is True
    assert cand_correct.reject_reason == ""

    # 誤った container_idx（big が存在しない container 0）では osize=0.5 が small(0.1) に
    # 収まらず不合格になる。fixture が index 契約を実際に判別できることの担保。
    cand_wrong_container = _make_candidate([0.5, 0.5, 0.5], osize, container_idx=0, ems_id=1)
    result_wrong_container = evaluate_stage(state, cand_wrong_container, PP0, MaskStage.DIMS)
    assert result_wrong_container.feasible is False
    assert result_wrong_container.reject_reason == "dims"

    # 誤った ems_id（container 1 の index 0 は small）でも同様に不合格になる。
    cand_wrong_ems = _make_candidate([0.5, 0.5, 0.5], osize, container_idx=1, ems_id=0)
    result_wrong_ems = evaluate_stage(state, cand_wrong_ems, PP0, MaskStage.DIMS)
    assert result_wrong_ems.feasible is False
    assert result_wrong_ems.reject_reason == "dims"


def test_evaluate_stage_dims_out_of_range_ems_id_raises_index_error():
    """範囲外の ems_id に対し、evaluate_stage が独自に握りつぶさず自然な IndexError を
    送出すること（別の例外への変換や暫定合格は行わない）。"""
    from src.packing_core.masks import MaskStage, evaluate_stage

    space = _unit_cube_space()
    ems_list = [
        _make_ems([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
        _make_ems([0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
    ]
    state = _make_state([space], {0: []}, {0: ems_list})
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=5)

    with pytest.raises(IndexError):
        evaluate_stage(state, cand, PP0, MaskStage.DIMS)


# --- prefilter_dims（T-015確定） ----------------------------------------------------------


def test_prefilter_dims_fits_exactly_passes():
    """osize が EMS 寸法と各軸一致（境界一致）なら合格。"""
    from src.packing_core.masks import prefilter_dims

    ems = _make_ems([0.0, 0.0, 0.0], [0.3, 0.4, 0.5])
    cand = _make_candidate([0.15, 0.2, 0.25], [0.3, 0.4, 0.5])

    assert prefilter_dims(cand, ems) is True


def test_prefilter_dims_exceeds_ems_rejected():
    """いずれかの軸で osize が EMS 寸法を超えれば不合格（ここでは x 軸のみ超過）。"""
    from src.packing_core.masks import prefilter_dims

    ems = _make_ems([0.0, 0.0, 0.0], [0.3, 0.4, 0.5])
    cand = _make_candidate([0.155, 0.2, 0.25], [0.31, 0.4, 0.5])

    assert prefilter_dims(cand, ems) is False


# --- check_overlap（T-015確定） -----------------------------------------------------------


def test_check_overlap_no_placed_passes():
    """同一コンテナに既配置がなければ合格。他コンテナの配置物と重なっていても対象外
    （同一コンテナの placed だけを対象とする契約）。"""
    from src.packing_core.masks import check_overlap

    space0 = _unit_cube_space()
    space1 = _unit_cube_space()
    other_container_placed = _make_placed([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
    state = _make_state(
        [space0, space1], {0: [], 1: [other_container_placed]}, {0: [], 1: []}
    )
    cand = _make_candidate([0.5, 0.5, 0.5], DEFAULT_OSIZE, container_idx=0, ems_id=0)

    assert check_overlap(state, cand, tol=-PP0.internal_extra) is True


def test_check_overlap_gap_within_internal_extra_rejected():
    """既配置との隙間が internal_extra(5mm) 未満なら tol=-internal_extra で交差扱いとなり
    不合格（3mm の重なり、および 3mm の隙間のいずれも不合格）。"""
    from src.packing_core.masks import check_overlap

    space = _unit_cube_space()
    placed = _make_placed([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
    state = _make_state([space], {0: [placed]}, {0: []})
    osize = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    # 3mm重なり: center_x=0.697 → x∈[0.597,0.797]（placed x上端0.6と3mm重なり）。
    cand_overlap_3mm = _make_candidate([0.697, 0.5, 0.5], osize, container_idx=0, ems_id=0)
    assert check_overlap(state, cand_overlap_3mm, tol=-PP0.internal_extra) is False

    # 隙間3mm（<internal_extra=5mm）: center_x=0.703 → x∈[0.603,0.803]、隙間0.003。
    cand_gap_3mm = _make_candidate([0.703, 0.5, 0.5], osize, container_idx=0, ems_id=0)
    assert check_overlap(state, cand_gap_3mm, tol=-PP0.internal_extra) is False


def test_check_overlap_sufficient_gap_passes():
    """隙間が internal_extra(5mm) 以上なら合格（8mm 隙間、および 5mm 境界のいずれも合格）。"""
    from src.packing_core.masks import check_overlap

    space = _unit_cube_space()
    placed = _make_placed([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
    state = _make_state([space], {0: [placed]}, {0: []})
    osize = np.array([0.2, 0.2, 0.2], dtype=np.float64)

    # 隙間8mm: center_x=0.708 → x∈[0.608,0.808]、隙間0.008。
    cand_gap_8mm = _make_candidate([0.708, 0.5, 0.5], osize, container_idx=0, ems_id=0)
    assert check_overlap(state, cand_gap_8mm, tol=-PP0.internal_extra) is True

    # 境界: 隙間5mm=internal_extra: center_x=0.705 → x∈[0.605,0.805]、隙間0.005。
    cand_gap_5mm = _make_candidate([0.705, 0.5, 0.5], osize, container_idx=0, ems_id=0)
    assert check_overlap(state, cand_gap_5mm, tol=-PP0.internal_extra) is True


# --- T-016B: l_path_sweep_boxes / check_l_path -----------------------------------------
#
# 確定した判定式（v1.13、§4.5「L字経路：プロキシ確定仕様」、出典
# validator.py::PlacementValidator.check_transport_path L96-99,101-137, 転記元HEAD abf630f）:
#     lane_x = clamp(target_x, path_lane_x_min_geom_rel+half_x+start_margin,
#                              path_lane_x_max_geom_rel-half_x-start_margin)
#     resting_surfaces = (inner_min_rel[2], path_mid_resting_z_rel)
#     ceiling_surfaces = (path_mid_ceiling_z_rel, inner_max_rel[2])
#     effective_start_z: 直置き面直上0〜0.05m(RESTING_SNAP_BAND)なら0、天井直前ならクリップ
#     rel_z = min(inner_max_rel[2]-half_z-start_margin, target_z+effective_start_z)
#     Yレグ: x=lane_x,z=rel_z固定でy: path_entry_y_rel -> target_y
#     Xレグ: y=target_y,z=rel_z固定でx: lane_x -> target_x
#     check_l_path (P4): 全レグ×全障害物で distance_sq=Σgap_i² > (safety_margin+EPS_GEOM)**2


def _path_space(
    inner_min, inner_max, cell,
    entry_y, lane_x_min_geom, lane_x_max_geom, mid_resting_z, mid_ceiling_z,
    obstacle_boxes=(),
):
    """`path_*` 6フィールドを明示指定できる `ContainerSpace`（l_path 専用テストヘルパ）。

    `build_container_space` を経由せず、公式式の各分岐（直置きスナップ・天井クリップ・
    レーンクランプ）を手計算で検証できるよう `path_mid_resting_z_rel`/`path_mid_ceiling_z_rel`
    を内壁境界から独立に指定する。
    """
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
        path_entry_y_rel=float(entry_y),
        path_lane_x_min_geom_rel=float(lane_x_min_geom),
        path_lane_x_max_geom_rel=float(lane_x_max_geom),
        path_mid_resting_z_rel=float(mid_resting_z),
        path_mid_ceiling_z_rel=float(mid_ceiling_z),
        path_obstacle_boxes_rel=tuple(
            (np.asarray(bmin, dtype=np.float64), np.asarray(bmax, dtype=np.float64))
            for bmin, bmax in obstacle_boxes
        ),
    )


def _baseline_path_space():
    """遠方に直置き/天井面を置き、スナップ・クリップが発火しない基準空間。"""
    return _path_space(
        inner_min=[-1.0, -1.0, 0.0], inner_max=[1.0, 1.0, 2.0], cell=0.1,
        entry_y=-1.0, lane_x_min_geom=-1.0, lane_x_max_geom=1.0,
        mid_resting_z=10.0, mid_ceiling_z=10.0,
    )


def test_l_path_sweep_boxes_baseline_y_then_x_leg_geometry():
    """クランプ・スナップ・クリップいずれも発火しない基準ケースでYレグ→Xレグの形状を検証する。"""
    from src.packing_core.masks import l_path_sweep_boxes

    space = _baseline_path_space()
    pp = constants.PlacementParams(start_margin=0.0, start_z=0.08, ceiling_margin=0.018)
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])

    # lane_x = clamp(0.3, -1.0+0.1+0, 1.0-0.1-0) = 0.3（クランプなし）。
    # 直置き・天井いずれのsurfaceも遠方(10.0)のためeffective_start_z=start_z=0.08。
    # rel_z = min(2.0-0.1-0, 1.0+0.08) = 1.08。
    (y_min, y_max), (x_min, x_max) = l_path_sweep_boxes(space, cand, pp)

    np.testing.assert_allclose(y_min, [0.2, -1.1, 0.98])
    np.testing.assert_allclose(y_max, [0.4, 0.6, 1.18])
    # Xレグ: lane_x(0.3)==target_x(0.3) → スイープ長ゼロだが候補半幅を持つ非退化AABB。
    np.testing.assert_allclose(x_min, [0.2, 0.4, 0.98])
    np.testing.assert_allclose(x_max, [0.4, 0.6, 1.18])


def test_l_path_sweep_boxes_lane_x_clamped_to_geom_bounds():
    """target_x がレーン範囲外なら lane_x はクランプされ、Xレグはクランプ位置から実target_xまで伸びる。"""
    from src.packing_core.masks import l_path_sweep_boxes

    space = _baseline_path_space()
    pp = constants.PlacementParams(start_margin=0.0, start_z=0.08, ceiling_margin=0.018)
    cand = _make_candidate([5.0, 0.5, 1.0], [0.2, 0.2, 0.2])

    # lane_x = clamp(5.0, -0.9, 0.9) = 0.9（上限クランプ）。z計算は基準ケースと同一。
    (y_min, y_max), (x_min, x_max) = l_path_sweep_boxes(space, cand, pp)

    np.testing.assert_allclose(y_min, [0.8, -1.1, 0.98])
    np.testing.assert_allclose(y_max, [1.0, 0.6, 1.18])
    np.testing.assert_allclose(x_min, [0.8, 0.4, 0.98])
    np.testing.assert_allclose(x_max, [5.1, 0.6, 1.18])


def test_l_path_sweep_boxes_resting_snap_sets_rel_z_to_target_z():
    """直置き面直上0〜0.05m(RESTING_SNAP_BAND)ならeffective_start_z=0となりrel_z=target_zになる。"""
    from src.packing_core.masks import l_path_sweep_boxes

    space = _path_space(
        inner_min=[-1.0, -1.0, 0.0], inner_max=[1.0, 1.0, 2.0], cell=0.1,
        entry_y=-1.0, lane_x_min_geom=-1.0, lane_x_max_geom=1.0,
        mid_resting_z=1.0, mid_ceiling_z=10.0,
    )
    pp = constants.PlacementParams(start_margin=0.0, start_z=0.08, ceiling_margin=0.018)
    # target_z=1.12, half_z=0.1 → bottom_z=1.02。bottom_z-mid_resting_z(1.0)=0.02∈[0,0.05]→snap。
    cand = _make_candidate([0.3, 0.5, 1.12], [0.2, 0.2, 0.2])

    (y_min, y_max), _ = l_path_sweep_boxes(space, cand, pp)

    assert y_min[2] == pytest.approx(1.12 - 0.1)
    assert y_max[2] == pytest.approx(1.12 + 0.1)


def test_l_path_sweep_boxes_ceiling_clip_reduces_effective_start_z():
    """天井面直前ではeffective_start_zが頭打ち回避のためクリップされる。"""
    from src.packing_core.masks import l_path_sweep_boxes

    space = _path_space(
        inner_min=[-1.0, -1.0, 0.0], inner_max=[1.0, 1.0, 2.0], cell=0.1,
        entry_y=-1.0, lane_x_min_geom=-1.0, lane_x_max_geom=1.0,
        mid_resting_z=0.5, mid_ceiling_z=1.5,
    )
    pp = constants.PlacementParams(start_margin=0.0, start_z=0.08, ceiling_margin=0.018)
    # target_z=1.35, half_z=0.1 → top_z=1.45。clearance=1.5-1.45=0.05∈[0, 0.08+0.018=0.098)→クリップ。
    # effective_start_z = max(0, 0.05-0.018-0.0005) = 0.0315。rel_z=min(1.9, 1.35+0.0315)=1.3815。
    cand = _make_candidate([0.3, 0.5, 1.35], [0.2, 0.2, 0.2])

    (y_min, y_max), _ = l_path_sweep_boxes(space, cand, pp)

    expected_rel_z = 1.35 + 0.0315
    assert y_min[2] == pytest.approx(expected_rel_z - 0.1, abs=1e-9)
    assert y_max[2] == pytest.approx(expected_rel_z + 0.1, abs=1e-9)


def test_l_path_sweep_boxes_does_not_mutate_inputs():
    from src.packing_core.masks import l_path_sweep_boxes

    space = _baseline_path_space()
    pp = constants.PlacementParams()
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])
    pos_rel_before = cand.pos_rel.copy()
    osize_before = cand.osize.copy()

    l_path_sweep_boxes(space, cand, pp)

    assert np.array_equal(cand.pos_rel, pos_rel_before)
    assert np.array_equal(cand.osize, osize_before)


def test_check_l_path_no_obstacles_passes():
    from src.packing_core.masks import check_l_path

    space = _baseline_path_space()
    pp = constants.PlacementParams()
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])
    state = _make_state([space], {0: []}, {0: []})

    assert check_l_path(state, cand, pp) is True


def test_check_l_path_placed_item_boundary_rejected_beyond_boundary_passes():
    """P4式の境界：gap==safety_marginは不合格、safety_margin+1e-6は合格（distance_sq比較）。"""
    from src.packing_core.masks import check_l_path, l_path_sweep_boxes

    space = _baseline_path_space()
    pp = constants.PlacementParams()
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])
    (y_leg_min, y_leg_max), _ = l_path_sweep_boxes(space, cand, pp)

    def _placed_with_gap(gap: float):
        obs_min = np.array([y_leg_max[0] + gap, y_leg_min[1], y_leg_min[2]], dtype=np.float64)
        obs_max = np.array([y_leg_max[0] + gap + 0.1, y_leg_max[1], y_leg_max[2]], dtype=np.float64)
        return _make_placed(obs_min, obs_max)

    state_boundary = _make_state([space], {0: [_placed_with_gap(pp.safety_margin)]}, {0: []})
    assert check_l_path(state_boundary, cand, pp) is False

    state_beyond = _make_state([space], {0: [_placed_with_gap(pp.safety_margin + 1e-6)]}, {0: []})
    assert check_l_path(state_beyond, cand, pp) is True


def test_check_l_path_uses_path_obstacle_boxes_rel():
    """既配置が空でも `path_obstacle_boxes_rel`（棚のraw AABB）と競合すれば不合格になる。"""
    from src.packing_core.masks import check_l_path, l_path_sweep_boxes

    space_no_obstacle = _baseline_path_space()
    pp = constants.PlacementParams()
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])
    (y_leg_min, y_leg_max), _ = l_path_sweep_boxes(space_no_obstacle, cand, pp)

    blocking_obstacle = (
        np.array([y_leg_min[0], y_leg_min[1], y_leg_min[2]], dtype=np.float64),
        np.array([y_leg_max[0], y_leg_max[1], y_leg_max[2]], dtype=np.float64),
    )
    space_with_obstacle = _path_space(
        inner_min=[-1.0, -1.0, 0.0], inner_max=[1.0, 1.0, 2.0], cell=0.1,
        entry_y=-1.0, lane_x_min_geom=-1.0, lane_x_max_geom=1.0,
        mid_resting_z=10.0, mid_ceiling_z=10.0,
        obstacle_boxes=(blocking_obstacle,),
    )
    state_no_obstacle = _make_state([space_no_obstacle], {0: []}, {0: []})
    state_with_obstacle = _make_state([space_with_obstacle], {0: []}, {0: []})

    assert check_l_path(state_no_obstacle, cand, pp) is True
    assert check_l_path(state_with_obstacle, cand, pp) is False


def test_check_l_path_returns_bool_and_does_not_mutate_inputs():
    from src.packing_core.masks import check_l_path

    space = _baseline_path_space()
    pp = constants.PlacementParams()
    cand = _make_candidate([0.3, 0.5, 1.0], [0.2, 0.2, 0.2])
    placed = _make_placed([0.9, 0.9, 0.9], [1.0, 1.0, 1.0])
    state = _make_state([space], {0: [placed]}, {0: []})

    pos_rel_before = cand.pos_rel.copy()
    osize_before = cand.osize.copy()
    inner_min_before = space.inner_min_rel.copy()
    aabb_min_before = placed.aabb_min_rel.copy()

    result = check_l_path(state, cand, pp)

    assert isinstance(result, bool)
    assert np.array_equal(cand.pos_rel, pos_rel_before)
    assert np.array_equal(cand.osize, osize_before)
    assert np.array_equal(space.inner_min_rel, inner_min_before)
    assert np.array_equal(placed.aabb_min_rel, aabb_min_before)


# --- T-016B: l_path_golden_v1（T-016Aゴールデン1,000件）照合契約テスト -----------------------
#
# DoD（§6 T-016B）: 危険な誤合格（proxy_pass and official_fail）0件、既知fixture
# （無障害/Yレグ遮蔽/Xレグ遮蔽）の合否一致、proxy_acceptance_rate（分母=全1,000件中
# official_pass 403件）が0.95以上。


def _l_path_golden_path():
    import pathlib

    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "datasets" / "fixtures" / "l_path_golden_v1.jsonl"
    )


def _load_l_path_golden_cases():
    import json

    path = _l_path_golden_path()
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _golden_case_state_and_candidate(case, space_cache):
    """ゴールデン1件から `(state, cand, pp)` を構築する（container_space_golden の
    `write_open_cut_corner_cup_obj`/`aff` 直接呼び出しでcdictを再構成、build_container_space
    は本番コードをそのまま使用）。"""
    from src.packing_core import geometry
    from src.packing_core.container_space import build_container_space
    from src.packing_core.types import Candidate, PlacedItem

    raw = case["container_raw_config"]
    offset_x = float(case["container_origin_world"])
    shape_key = (
        raw["length"], raw["width"], raw["height"], raw["thickness"],
        raw["cut_x"], raw["cut_y"], raw["buffer"], raw["require_shelf"],
    )
    space = space_cache.get((shape_key, offset_x))
    if space is None:
        cdict = golden.build_cdict_from_raw_config(raw, offset_x=offset_x)
        space = build_container_space(cdict, index=0, cell=golden.CELL)
        space_cache[(shape_key, offset_x)] = space

    pp = constants.PlacementParams(
        safety_margin=case["effective_validator_config"]["safety_margin"],
        start_z=case["effective_validator_config"]["start_z"],
        ceiling_margin=case["effective_validator_config"]["ceiling_margin"],
        start_margin=case["effective_validator_config"]["start_margin"],
    )

    orientation = case["candidate_orientation"]
    size = np.array(case["candidate_size"], dtype=np.float64)
    osize = geometry.oriented_size(size, orientation)
    target_world = np.array(case["candidate_target_pos"], dtype=np.float64)
    pos_rel = target_world - np.array([offset_x, 0.0, 0.0])
    cand = Candidate(
        item_idx=case["candidate_item_index"], container_idx=0, ems_id=0,
        orientation=orientation, pos_rel=pos_rel, osize=osize,
    )

    placed = []
    for item in case["placed_items"]:
        pos_world = np.array(item["pos_world"], dtype=np.float64)
        orn_quat = np.array(item["orn_quat"], dtype=np.float64)
        size_lwh = np.array(item["size"], dtype=np.float64)
        world_min, world_max = geometry.rotated_aabb(pos_world, size_lwh, orn_quat)
        origin = np.array([offset_x, 0.0, 0.0], dtype=np.float64)
        placed.append(PlacedItem(
            pos_world=pos_world, orn_quat=orn_quat, size=size_lwh, weight=1.0,
            is_soft=False, is_priority=False,
            aabb_min_rel=world_min - origin, aabb_max_rel=world_max - origin,
        ))

    state = _make_state([space], {0: placed}, {0: []})
    return state, cand, pp


def _run_l_path_golden():
    """全1,000件を実行し `(cases, results)` を返す（`results[i]` は `check_l_path` の bool）。
    複数テストで再利用するモジュールレベルキャッシュ。
    """
    from src.packing_core.masks import check_l_path

    cases = _load_l_path_golden_cases()
    space_cache: dict = {}
    results = [
        check_l_path(*_golden_case_state_and_candidate(case, space_cache)[:3])
        for case in cases
    ]
    return cases, results


_L_PATH_GOLDEN_CACHE: dict = {}


def _l_path_golden_results():
    if "value" not in _L_PATH_GOLDEN_CACHE:
        _L_PATH_GOLDEN_CACHE["value"] = _run_l_path_golden()
    return _L_PATH_GOLDEN_CACHE["value"]


def test_l_path_golden_dataset_has_1000_cases_with_403_official_pass():
    cases, _ = _l_path_golden_results()
    assert len(cases) == 1000
    assert sum(1 for c in cases if c["official_verdict"]) == 403


def test_l_path_golden_dangerous_false_accept_is_zero():
    """安全契約: proxy_pass and official_fail（危険な誤合格）が全1,000件で0件であること。"""
    cases, results = _l_path_golden_results()
    dangerous = [
        c["case_id"] for c, proxy_pass in zip(cases, results)
        if proxy_pass and not c["official_verdict"]
    ]
    assert dangerous == []


def test_l_path_golden_proxy_acceptance_rate_meets_floor():
    """proxy_acceptance_rate（分母=official_pass 403件）が下限0.95以上であること。"""
    cases, results = _l_path_golden_results()
    official_pass_flags = [proxy_pass for c, proxy_pass in zip(cases, results) if c["official_verdict"]]
    assert len(official_pass_flags) == 403
    acceptance_rate = sum(official_pass_flags) / len(official_pass_flags)
    assert acceptance_rate >= 0.95


def test_l_path_golden_clear_path_category_mostly_accepted():
    """既知fixture（無障害）: clear_pathカテゴリ（全件official pass）で退化した全不合格実装を排除する。"""
    cases, results = _l_path_golden_results()
    clear_path_results = [
        proxy_pass for c, proxy_pass in zip(cases, results) if c["category"] == "clear_path"
    ]
    assert len(clear_path_results) == 200
    assert all(clear_path_results)


def test_l_path_golden_y_leg_blocked_category_all_rejected():
    """既知fixture（Yレグ遮蔽）: 全件official failのため、危険な誤合格0件契約から全件不合格になる。"""
    cases, results = _l_path_golden_results()
    y_blocked_results = [
        proxy_pass for c, proxy_pass in zip(cases, results) if c["category"] == "y_leg_blocked"
    ]
    assert len(y_blocked_results) == 200
    assert not any(y_blocked_results)


def test_l_path_golden_x_leg_blocked_category_all_rejected():
    """既知fixture（Xレグ遮蔽）: 全件official failのため、危険な誤合格0件契約から全件不合格になる。"""
    cases, results = _l_path_golden_results()
    x_blocked_results = [
        proxy_pass for c, proxy_pass in zip(cases, results) if c["category"] == "x_leg_blocked"
    ]
    assert len(x_blocked_results) == 200
    assert not any(x_blocked_results)


def test_l_path_golden_performance_budget_p95_under_1ms():
    """性能予算: p95<1.0ms/candidate（既存 `PackingState`/`ContainerSpace` を1回構築しループ外で
    再利用する構成、§4.5「性能予算」）。ウォームアップ1回、GC無効化、time.perf_counter()。"""
    import gc
    import time

    from src.packing_core.masks import check_l_path

    cases = _load_l_path_golden_cases()[:300]
    space_cache: dict = {}
    prepared = [_golden_case_state_and_candidate(case, space_cache) for case in cases]

    # ウォームアップ（import・分岐予測等の一過性コストを計測対象から除外）。
    for state, cand, pp in prepared[:10]:
        check_l_path(state, cand, pp)

    gc.disable()
    try:
        durations = []
        for state, cand, pp in prepared:
            start = time.perf_counter()
            check_l_path(state, cand, pp)
            durations.append(time.perf_counter() - start)
    finally:
        gc.enable()

    durations.sort()
    p95 = durations[int(len(durations) * 0.95)]
    assert p95 < 1.0e-3, f"check_l_path p95 duration {p95 * 1000:.4f}ms exceeds 1.0ms budget"
