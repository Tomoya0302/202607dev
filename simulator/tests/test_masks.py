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
