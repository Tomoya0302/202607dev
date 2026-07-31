"""統合T-024: score.heuristic_score の契約テスト（詳細仕様書 v1.18 §4.8、契約計画 §3.F）。

`score.py` は未実装のため対象関数の import は各テスト関数内で行う（`tests/test_masks.py`と
同方針）。全20家族が分類 I。

完全式（§4.8）:
    z_top_norm    = clip((pos_z + osize_z/2 - inner_min_z) / span_z, 0, 1)
    z_center_norm = clip((pos_z - inner_min_z) / span_z, 0, 1)
    y_center_norm = clip((pos_y - inner_min_y) / span_y, 0, 1)
    x_center_norm = clip((pos_x - inner_min_x) / span_x, 0, 1)
    weight_norm   = clip(item.weight / POOL_MAX_WEIGHT, 0, 1)
    cg_height_norm = z_center_norm * weight_norm
    support = clip(support_ratio(state, cand), 0, 1)
    hard_on_soft_flag = float((not item.is_soft) and soft_below_ratio(state, cand) > 0.0)
    priority_ok_flag = 0.0  # T-024固定
    score = -w_z*z_top_norm + w_y*y_center_norm - w_x*x_center_norm + w_support*support
            - w_cg_h*cg_height_norm - w_soft*hard_on_soft_flag + w_prio*priority_ok_flag

大半のテストは「1点だけ異なる2つのCandidate/Stateのペア」を比較し、スコア差が該当項の
係数×正規化値差にちょうど一致することを確認する（他の項は両者で共通のため差分に現れず、
ゼロ化トリックなしに項を分離できる）。
"""
import math

import numpy as np
import pytest

from agents.heuristic.packing_core import constants
from agents.heuristic.packing_core.container_space import ContainerSpace, bake_placed, build_container_space
from agents.heuristic.packing_core.state import PackingState
from agents.heuristic.packing_core.types import Candidate, EMSBox, ItemSpec, PlacedItem

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
SPAN = INNER_MAX_REL - INNER_MIN_REL  # [0.8, 0.8, 1.0]
CELL = 0.02
SP0 = constants.ScoreParams()  # w_z=1.0,w_y=0.3,w_x=0.1,w_support=0.5,w_cg_h=0.3,w_soft=0.8,w_prio=0.4
POOL_MAX_WEIGHT = constants.POOL_MAX_WEIGHT  # 18.0


def _box_cdict():
    imin, imax = INNER_MIN_REL, INNER_MAX_REL
    mid = (imin + imax) / 2.0
    thickness = 0.02
    buffer = 0.02
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer

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

    return {
        "index": 0, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness,
        "center": (0.0, 0.0, height / 2.0 + buffer),
        "n_vecs": n_vecs, "points": points,
        "volume": float(np.prod(imax - imin)),
        "shelf": False, "is_prioritized": False, "packed_items": [],
    }


def _flat_space():
    return build_container_space(_box_cdict(), index=0, cell=CELL)


def _item(idx=0, size=(0.1, 0.1, 0.1), weight=5.0, is_soft=False, is_priority=False):
    return ItemSpec(
        idx=idx, size=np.array(size, dtype=np.float64), weight=weight,
        kind=None, is_soft=is_soft, is_priority=is_priority,
    )


def _cand(item_idx=0, pos_rel=(0.0, 0.0, 0.5), osize=(0.1, 0.1, 0.1), container_idx=0, ems_id=0):
    return Candidate(
        item_idx=item_idx, container_idx=container_idx, ems_id=ems_id, orientation=0,
        pos_rel=np.array(pos_rel, dtype=np.float64), osize=np.array(osize, dtype=np.float64),
    )


def _register_ems(state, cand):
    """`cand.container_idx`/`ems_id`スロットへ、想定沈降後Z（`stability.
    expected_settled_pos_rel`が読む`ems.min_rel[2]+osize[2]/2`）が`cand.pos_rel[2]`と
    一致するようなEMSBoxを登録する（HF-001でheuristic_scoreがEMS参照必須になったため、
    本ファイルの既存の期待値（z_top_norm等はcand.pos_rel[2]から直接計算）を変えずに
    テストを通すためのfixtureヘルパ。X/Yは`expected_settled_pos_rel`が使わないため
    広めの適当な値でよい）。"""
    bottom_z = float(cand.pos_rel[2]) - float(cand.osize[2]) / 2.0
    ems = EMSBox(
        min_rel=np.array([-10.0, -10.0, bottom_z], dtype=np.float64),
        max_rel=np.array([10.0, 10.0, bottom_z + 10.0], dtype=np.float64),
    )
    ems_list = state.ems[cand.container_idx]
    while len(ems_list) <= cand.ems_id:
        ems_list.append(ems)
    ems_list[cand.ems_id] = ems


def _placed(xmin, xmax, ymin, ymax, zmin, zmax, is_soft=False):
    amin = np.array([xmin, ymin, zmin], dtype=np.float64)
    amax = np.array([xmax, ymax, zmax], dtype=np.float64)
    return PlacedItem(
        pos_world=(amin + amax) / 2.0, orn_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        size=amax - amin, weight=1.0, is_soft=is_soft, is_priority=False,
        aabb_min_rel=amin, aabb_max_rel=amax,
    )


def _state(pool, placed=()):
    space = _flat_space()
    placed_list = list(placed)
    bake_placed(space, placed_list)
    return PackingState(
        containers=[space], placed={0: placed_list}, pool=pool,
        ems={0: []}, ems_truncation={0: 0.0}, meta={},
    )


def _score(state, cand, sp=SP0):
    from agents.heuristic.packing_core.score import heuristic_score
    return heuristic_score(state, cand, sp)


# --- SCORE-001..008: 単項分離（差分法） ------------------------------------------------------


def test_score_001_z_top_term_sign_and_coefficient():
    """z_top_norm: 係数-w_z。他の項が変化しないよう weight=0（cg_height=0固定）・
    floating（support=0固定、両位置とも十分高い）で位置のみ変える。"""
    item = _item(weight=0.0)
    state = _state([item])
    cand_a = _cand(pos_rel=(0.0, 0.0, 0.5), ems_id=0)  # z_top_norm=(0.5+0.05-0.02)/1.0=0.53
    cand_b = _cand(pos_rel=(0.0, 0.0, 0.7), ems_id=1)  # z_top_norm=(0.7+0.05-0.02)/1.0=0.73
    _register_ems(state, cand_a)
    _register_ems(state, cand_b)

    score_a = _score(state, cand_a)
    score_b = _score(state, cand_b)
    ztop_a, ztop_b = 0.53, 0.73
    expected_diff = -SP0.w_z * (ztop_b - ztop_a)
    assert (score_b - score_a) == pytest.approx(expected_diff)


def test_score_002_y_center_term_sign_and_coefficient():
    """y_center_norm: 係数+w_y。z/weightを固定し位置yのみ変える。"""
    item = _item(weight=0.0)
    state = _state([item])
    cand_a = _cand(pos_rel=(0.0, INNER_MIN_REL[1], 0.5))  # y_center_norm=0
    cand_b = _cand(pos_rel=(0.0, 0.0, 0.5))  # y_center_norm=(0-(-0.4))/0.8=0.5
    _register_ems(state, cand_a)  # 両候補ともz=0.5で共通

    score_a = _score(state, cand_a)
    score_b = _score(state, cand_b)
    expected_diff = SP0.w_y * (0.5 - 0.0)
    assert (score_b - score_a) == pytest.approx(expected_diff)


def test_score_003_x_center_term_sign_and_coefficient():
    """x_center_norm: 係数-w_x。z/weightを固定し位置xのみ変える。"""
    item = _item(weight=0.0)
    state = _state([item])
    cand_a = _cand(pos_rel=(INNER_MIN_REL[0], 0.0, 0.5))  # x_center_norm=0
    cand_b = _cand(pos_rel=(0.0, 0.0, 0.5))  # x_center_norm=0.5
    _register_ems(state, cand_a)  # 両候補ともz=0.5で共通

    score_a = _score(state, cand_a)
    score_b = _score(state, cand_b)
    expected_diff = -SP0.w_x * (0.5 - 0.0)
    assert (score_b - score_a) == pytest.approx(expected_diff)


def test_score_004_support_term_sign_and_coefficient():
    """support: 係数+w_support。pos_rel/itemは同一、既配置の有無だけで support 0→1 にする。"""
    item = _item(weight=0.0)
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))  # bottom_z=0.45

    state_a = _state([item])  # 既配置なし → support=0
    platform = _placed(-0.4, 0.4, -0.4, 0.4, 0.02, 0.45)  # top=0.45=bottom_z → support=1
    state_b = _state([item], placed=[platform])
    _register_ems(state_a, cand)
    _register_ems(state_b, cand)

    score_a = _score(state_a, cand)
    score_b = _score(state_b, cand)
    expected_diff = SP0.w_support * (1.0 - 0.0)
    assert (score_b - score_a) == pytest.approx(expected_diff)


def test_score_005_cg_height_term_sign_and_coefficient():
    """cg_height_norm=z_center_norm*weight_norm: 係数-w_cg_h。pos_rel同一（z_center固定=0.48）で
    item.weightだけを変える。"""
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))  # z_center_norm=(0.5-0.02)/1.0=0.48
    z_center = 0.48

    item_a = _item(weight=0.0)  # weight_norm=0
    item_b = _item(weight=9.0)  # weight_norm=9/18=0.5
    state_a = _state([item_a])
    state_b = _state([item_b])
    _register_ems(state_a, cand)
    _register_ems(state_b, cand)

    score_a = _score(state_a, cand)
    score_b = _score(state_b, cand)
    cg_a, cg_b = z_center * 0.0, z_center * 0.5
    expected_diff = -SP0.w_cg_h * (cg_b - cg_a)
    assert (score_b - score_a) == pytest.approx(expected_diff)


def test_score_006_hard_on_soft_term_sign_and_coefficient():
    """hard_on_soft_flag: 係数-w_soft。hardなitemに対し、支持面がhard→softに変わるだけで
    flagが0→1になる（support=1は両ケース共通）。"""
    item = _item(weight=0.0, is_soft=False)
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))  # bottom_z=0.45

    hard_platform = _placed(-0.4, 0.4, -0.4, 0.4, 0.02, 0.45, is_soft=False)
    soft_platform = _placed(-0.4, 0.4, -0.4, 0.4, 0.02, 0.45, is_soft=True)
    state_hard = _state([item], placed=[hard_platform])
    state_soft = _state([item], placed=[soft_platform])
    _register_ems(state_hard, cand)
    _register_ems(state_soft, cand)

    score_hard = _score(state_hard, cand)
    score_soft = _score(state_soft, cand)
    expected_diff = -SP0.w_soft * (1.0 - 0.0)
    assert (score_soft - score_hard) == pytest.approx(expected_diff)


def test_score_007_priority_ok_flag_is_fixed_zero():
    """priority_ok_flag はT-024では常に0.0固定。is_priorityを変えてもスコアは不変。"""
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    item_a = _item(weight=3.0, is_priority=False)
    item_b = _item(weight=3.0, is_priority=True)
    state_a = _state([item_a])
    state_b = _state([item_b])
    _register_ems(state_a, cand)
    _register_ems(state_b, cand)

    score_a = _score(state_a, cand)
    score_b = _score(state_b, cand)
    assert score_a == pytest.approx(score_b)


def test_score_008_weight_norm_clamps_at_pool_max_weight():
    """weight_norm は POOL_MAX_WEIGHT で [0,1] にクランプされる。ちょうど上限(18.0)と
    大幅超過(100.0)で同一スコアになる（クランプの証拠）。"""
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    item_at_cap = _item(weight=POOL_MAX_WEIGHT)
    item_over_cap = _item(weight=100.0)
    state_at_cap = _state([item_at_cap])
    state_over_cap = _state([item_over_cap])
    _register_ems(state_at_cap, cand)
    _register_ems(state_over_cap, cand)

    score_at_cap = _score(state_at_cap, cand)
    score_over_cap = _score(state_over_cap, cand)
    assert score_at_cap == pytest.approx(score_over_cap)


# --- SCORE-009: 平床2候補（低い vs 高い）で低い方が高スコア（DoD例） -------------------------


def test_score_009_flat_floor_lower_candidate_scores_higher():
    item = _item(weight=3.0)
    state = _state([item])
    cand_low = _cand(pos_rel=(0.0, 0.0, 0.3), ems_id=0)
    cand_high = _cand(pos_rel=(0.0, 0.0, 0.8), ems_id=1)
    _register_ems(state, cand_low)
    _register_ems(state, cand_high)

    score_low = _score(state, cand_low)
    score_high = _score(state, cand_high)
    assert score_low > score_high


# --- SCORE-010/011: 戻り値型・有限性 ---------------------------------------------------------


def test_score_010_return_type_is_builtin_float():
    item = _item(weight=3.0)
    state = _state([item])
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)
    result = _score(state, cand)
    assert type(result) is float


def test_score_011_valid_input_returns_finite_value():
    item = _item(weight=3.0)
    state = _state([item])
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)
    result = _score(state, cand)
    assert math.isfinite(result)


# --- SCORE-012: 非破壊 ------------------------------------------------------------------------


def test_score_012_does_not_mutate_candidate_state_or_item():
    from agents.heuristic.packing_core.score import heuristic_score

    item = _item(idx=0, weight=3.0)
    state = _state([item])
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)

    pos_before = cand.pos_rel.copy()
    osize_before = cand.osize.copy()
    score_before = cand.score
    features_before = dict(cand.features)
    weight_before = item.weight

    heuristic_score(state, cand, SP0)

    np.testing.assert_array_equal(cand.pos_rel, pos_before)
    np.testing.assert_array_equal(cand.osize, osize_before)
    assert cand.score == score_before  # heuristic_score自身は代入しない（呼び出し側の責務）
    assert cand.features == features_before
    assert item.weight == weight_before


# --- SCORE-013: cg_marginを使わない（呼ばない） -----------------------------------------------


def test_score_013_does_not_call_cg_margin(monkeypatch):
    from agents.heuristic.packing_core import stability

    calls = []
    monkeypatch.setattr(stability, "cg_margin", lambda *a, **k: calls.append(1) or 0.0, raising=False)

    item = _item(weight=3.0)
    state = _state([item])
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)
    _score(state, cand)

    assert calls == []


# --- SCORE-014（訂正）: 全項非零の完全式一致 ---------------------------------------------------


def test_score_014_full_formula_exact_match_all_terms_nonzero():
    item = _item(idx=0, weight=9.0, is_soft=False, is_priority=True)
    cand = _cand(item_idx=0, pos_rel=(0.1, 0.1, 0.5), osize=(0.1, 0.1, 0.1))
    # 候補footprint x:[0.05,0.15] y:[0.05,0.15]、bottom_z=0.45。
    # 全面soft platformで覆う → support=1.0・soft_below_ratio=1.0・hard_on_soft_flag=1.0。
    soft_platform = _placed(-0.4, 0.4, -0.4, 0.4, 0.02, 0.45, is_soft=True)
    state = _state([item], placed=[soft_platform])
    _register_ems(state, cand)

    z_top_norm = (0.5 + 0.05 - 0.02) / 1.0       # 0.53
    z_center_norm = (0.5 - 0.02) / 1.0            # 0.48
    y_center_norm = (0.1 - (-0.4)) / 0.8          # 0.625
    x_center_norm = (0.1 - (-0.4)) / 0.8          # 0.625
    weight_norm = 9.0 / POOL_MAX_WEIGHT           # 0.5
    cg_height_norm = z_center_norm * weight_norm  # 0.24
    support = 1.0
    hard_on_soft_flag = 1.0
    priority_ok_flag = 0.0

    expected = (
        -SP0.w_z * z_top_norm
        + SP0.w_y * y_center_norm
        - SP0.w_x * x_center_norm
        + SP0.w_support * support
        - SP0.w_cg_h * cg_height_norm
        - SP0.w_soft * hard_on_soft_flag
        + SP0.w_prio * priority_ok_flag
    )

    result = _score(state, cand)
    assert result == pytest.approx(expected, abs=1e-9)


# --- SCORE-015: span<=0はValueError ------------------------------------------------------------


def _degenerate_space_x():
    """X軸のspanが0（inner_min_rel[0]==inner_max_rel[0]）の縮退ContainerSpace。"""
    imin = np.array([0.0, -0.4, 0.02], dtype=np.float64)
    imax = np.array([0.0, 0.4, 1.02], dtype=np.float64)  # X span = 0
    floor_z = np.zeros((1, 40), dtype=np.float64)
    ceil_z = np.full((1, 40), 1.02, dtype=np.float64)
    return ContainerSpace(
        index=0, offset_x=0.0, inner_min_rel=imin, inner_max_rel=imax,
        cut_planes=[], shelf_boxes=[], cell=CELL, floor_z=floor_z, ceil_z=ceil_z,
        height=floor_z.copy(), path_entry_y_rel=-0.4, path_lane_x_min_geom_rel=0.0,
        path_lane_x_max_geom_rel=0.0, path_mid_resting_z_rel=0.02, path_mid_ceiling_z_rel=1.02,
        path_obstacle_boxes_rel=(),
    )


def test_score_015_degenerate_span_raises_value_error():
    from agents.heuristic.packing_core.score import heuristic_score

    item = _item(weight=3.0)
    space = _degenerate_space_x()
    state = PackingState(
        containers=[space], placed={0: []}, pool=[item],
        ems={0: []}, ems_truncation={0: 0.0}, meta={},
    )
    cand = _cand(pos_rel=(0.0, 0.0, 0.5))

    with pytest.raises(ValueError):
        heuristic_score(state, cand, SP0)


# --- SCORE-016: 無効入力→ValueError（param×25） -----------------------------------------------


def _valid_fixture():
    item = _item(idx=0, weight=3.0)
    state = _state([item])
    cand = _cand(item_idx=0, pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)
    return state, cand


def _case_posrel(component, value):
    def _build():
        state, cand = _valid_fixture()
        cand.pos_rel = cand.pos_rel.copy()
        cand.pos_rel[component] = value
        return state, cand, SP0
    return _build


def _case_osize(component, value):
    def _build():
        state, cand = _valid_fixture()
        cand.osize = cand.osize.copy()
        cand.osize[component] = value
        return state, cand, SP0
    return _build


def _case_weight(value):
    def _build():
        item = _item(idx=0, weight=value)
        state = _state([item])
        cand = _cand(item_idx=0, pos_rel=(0.0, 0.0, 0.5))
        _register_ems(state, cand)
        return state, cand, SP0
    return _build


def _case_inner_min(component, value):
    def _build():
        state, cand = _valid_fixture()
        space = state.containers[0]
        space.inner_min_rel = space.inner_min_rel.copy()
        space.inner_min_rel[component] = value
        return state, cand, SP0
    return _build


def _case_inner_max(component, value):
    def _build():
        state, cand = _valid_fixture()
        space = state.containers[0]
        space.inner_max_rel = space.inner_max_rel.copy()
        space.inner_max_rel[component] = value
        return state, cand, SP0
    return _build


def _case_span_zero(axis):
    def _build():
        state, cand = _valid_fixture()
        space = state.containers[0]
        space.inner_max_rel = space.inner_max_rel.copy()
        space.inner_max_rel[axis] = space.inner_min_rel[axis]  # span=0
        return state, cand, SP0
    return _build


def _case_pool_max_weight(monkeypatch_value):
    def _build():
        state, cand = _valid_fixture()
        return state, cand, SP0, ("POOL_MAX_WEIGHT", monkeypatch_value)
    return _build


def _case_weights(field, value):
    def _build():
        state, cand = _valid_fixture()
        sp = constants.ScoreParams(**{field: value})
        return state, cand, sp
    return _build


def _patch_support_ratio(monkeypatch, fn):
    """`stability.support_ratio` を、import文の形式（`from...import`か`module.attr`か）に
    依存せず確実にスパイ化する（両方のバインディング箇所へ patch、片方はraising=False）。"""
    from agents.heuristic.packing_core import score as score_module
    from agents.heuristic.packing_core import stability
    monkeypatch.setattr(stability, "support_ratio", fn, raising=False)
    monkeypatch.setattr(score_module, "support_ratio", fn, raising=False)


def _patch_soft_below_ratio(monkeypatch, fn):
    from agents.heuristic.packing_core import score as score_module
    from agents.heuristic.packing_core import stability
    monkeypatch.setattr(stability, "soft_below_ratio", fn, raising=False)
    monkeypatch.setattr(score_module, "soft_below_ratio", fn, raising=False)


NAN, PINF, NINF = float("nan"), float("inf"), float("-inf")

_SCORE_016_CASES = [
    pytest.param(_case_posrel(0, NAN), id="T024-SCORE-016-POSREL-NAN"),
    pytest.param(_case_posrel(1, PINF), id="T024-SCORE-016-POSREL-PINF"),
    pytest.param(_case_posrel(2, NINF), id="T024-SCORE-016-POSREL-NINF"),
    pytest.param(_case_osize(0, NAN), id="T024-SCORE-016-OSIZE-NAN"),
    pytest.param(_case_osize(1, PINF), id="T024-SCORE-016-OSIZE-PINF"),
    pytest.param(_case_osize(2, NINF), id="T024-SCORE-016-OSIZE-NINF"),
    pytest.param(_case_weight(NAN), id="T024-SCORE-016-WEIGHT-NAN"),
    pytest.param(_case_weight(PINF), id="T024-SCORE-016-WEIGHT-PINF"),
    pytest.param(_case_weight(NINF), id="T024-SCORE-016-WEIGHT-NINF"),
    pytest.param(_case_inner_min(0, NAN), id="T024-SCORE-016-INNERMIN-NAN"),
    pytest.param(_case_inner_min(1, PINF), id="T024-SCORE-016-INNERMIN-PINF"),
    pytest.param(_case_inner_min(2, NINF), id="T024-SCORE-016-INNERMIN-NINF"),
    pytest.param(_case_inner_max(0, NAN), id="T024-SCORE-016-INNERMAX-NAN"),
    pytest.param(_case_inner_max(1, PINF), id="T024-SCORE-016-INNERMAX-PINF"),
    pytest.param(_case_inner_max(2, NINF), id="T024-SCORE-016-INNERMAX-NINF"),
    pytest.param(_case_span_zero(0), id="T024-SCORE-016-SPAN-ZERO-X"),
    pytest.param(_case_span_zero(1), id="T024-SCORE-016-SPAN-ZERO-Y"),
    pytest.param(_case_span_zero(2), id="T024-SCORE-016-SPAN-ZERO-Z"),
    pytest.param(_case_pool_max_weight(NAN), id="T024-SCORE-016-POOLMAX-NAN"),
    pytest.param(_case_pool_max_weight(0.0), id="T024-SCORE-016-POOLMAX-ZERO"),
    pytest.param(_case_pool_max_weight(-5.0), id="T024-SCORE-016-POOLMAX-NEG"),
    pytest.param(_case_weights("w_z", NAN), id="T024-SCORE-016-WEIGHTS-NAN"),
]

# support_ratio/soft_below_ratio 非有限（spy注入）は別途3ケース追加。
_SUPPORT_SOFT_CASES = [
    "T024-SCORE-016-SUPPORT-NAN", "T024-SCORE-016-SUPPORT-PINF", "T024-SCORE-016-SOFTBELOW-NAN",
]


@pytest.mark.parametrize("case_id", [p.id for p in _SCORE_016_CASES] + _SUPPORT_SOFT_CASES)
def test_score_016_invalid_input_raises_value_error(monkeypatch, case_id):
    from agents.heuristic.packing_core.score import heuristic_score

    by_id = {p.id: p.values[0] for p in _SCORE_016_CASES}

    if case_id in by_id:
        builder = by_id[case_id]
        built = builder()
        if len(built) == 4:
            state, cand, sp, (attr, value) = built
            monkeypatch.setattr(constants, attr, value)
        else:
            state, cand, sp = built
    elif case_id == "T024-SCORE-016-SUPPORT-NAN":
        _patch_support_ratio(monkeypatch, lambda *a, **k: NAN)
        state, cand = _valid_fixture()
        sp = SP0
    elif case_id == "T024-SCORE-016-SUPPORT-PINF":
        _patch_support_ratio(monkeypatch, lambda *a, **k: PINF)
        state, cand = _valid_fixture()
        sp = SP0
    elif case_id == "T024-SCORE-016-SOFTBELOW-NAN":
        _patch_soft_below_ratio(monkeypatch, lambda *a, **k: NAN)
        state, cand = _valid_fixture()
        sp = SP0
    else:
        raise AssertionError(f"unknown case_id {case_id}")

    with pytest.raises(ValueError):
        heuristic_score(state, cand, sp)


# --- SCORE-017..020: v1.18 item解決契約（idx一意検索） -----------------------------------------


def test_score_017_resolves_by_idx_not_pool_position():
    """pool位置0にidx=17のItemSpecを置き、Candidate.item_idx=17で解決できる
    （pool[17]という位置添字ではない）。"""
    item = _item(idx=17, weight=3.0)
    state = _state([item])  # pool位置0だがidx=17
    cand = _cand(item_idx=17, pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state, cand)

    result = _score(state, cand)
    assert math.isfinite(result)


def test_score_018_score_independent_of_pool_order():
    item_a = _item(idx=5, weight=3.0)
    item_b = _item(idx=9, weight=7.0)
    state_order1 = _state([item_a, item_b])
    state_order2 = _state([item_b, item_a])

    cand = _cand(item_idx=5, pos_rel=(0.0, 0.0, 0.5))
    _register_ems(state_order1, cand)
    _register_ems(state_order2, cand)
    score_order1 = _score(state_order1, cand)
    score_order2 = _score(state_order2, cand)
    assert score_order1 == pytest.approx(score_order2)


def test_score_019_no_matching_idx_raises_value_error():
    item = _item(idx=3, weight=3.0)
    state = _state([item])
    cand = _cand(item_idx=999, pos_rel=(0.0, 0.0, 0.5))  # 該当idxなし

    with pytest.raises(ValueError):
        _score(state, cand)


def test_score_020_duplicate_idx_raises_value_error():
    item_a = _item(idx=4, weight=3.0)
    item_b = _item(idx=4, weight=8.0)  # idx重複
    state = _state([item_a, item_b])
    cand = _cand(item_idx=4, pos_rel=(0.0, 0.0, 0.5))

    with pytest.raises(ValueError):
        _score(state, cand)
