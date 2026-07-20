"""packing_core.stability の契約検証（詳細仕様書 §2・§3・§4.2・§4.4・§4.6・§6 T-020〜T-021）。

stability.py の import は各テスト関数の内部で行う（tests/test_geometry.py・
tests/test_container_space.py・tests/test_masks.py と同方針）。

T-020 では `support_ratio`／`max_step_below`／`soft_below_ratio` の3関数を検証する（後半の
T-021 節では `support_polygon`／`cg_margin` を検証する）。

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
    D4（T-021）. support_polygon は「支持セル中心点の凸包」ではなく、支持セルが表す矩形接触
        パッチ（各セルの生矩形を候補底面AABBおよびコンテナ内壁XYでクリップしたもの）の
        全頂点の凸包（人間確定・案B′、`docs/実装詳細仕様書.md` §4.6 v1.14追補参照）。
        平床 cg_margin=0.20（許容1e-3）・半分支持 cg_margin=0.0 の期待値はこの定義で成立する。
    D5（HF-001, v1.27）. 本モジュールの全関数は`stability.expected_settled_pos_rel(state, cand)`
        経由で想定沈降後Zを参照するため、`state.ems[cand.container_idx][cand.ems_id]`が
        存在している必要がある。本ファイルの`_candidate()`は生成と同時に、
        `settled_z == bottom_z`（既存の期待値を変えない）となるEMSBoxを`state.ems`へ
        登録する私有ヘルパ`_register_ems(state, cand)`を通す（下記参照）。

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
from src.packing_core.types import Candidate, EMSBox, PlacedItem

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


def _register_ems(state: PackingState, cand: Candidate) -> None:
    """`cand.container_idx`/`ems_id`スロットへ、想定沈降後Z（`stability.
    expected_settled_pos_rel`が読む`ems.min_rel[2]+osize[2]/2`）が`cand.pos_rel[2]`
    （＝`_candidate()`のbottom_z由来）と一致するEMSBoxを登録する（HF-001、D5参照）。
    X/Yは`expected_settled_pos_rel`が使わないため広めの適当な値でよい。
    `container_idx`が`state.containers`の範囲外のテスト（container_idx検証用）でも
    `state.ems`へのキー追加自体は無害（実際の検証はそれより前のcontainer_idx範囲チェックで
    ValueErrorになる）。"""
    if cand.container_idx not in state.ems:
        state.ems[cand.container_idx] = []
    ems_list = state.ems[cand.container_idx]
    bottom_z = float(cand.pos_rel[2]) - float(cand.osize[2]) / 2.0
    ems = EMSBox(
        min_rel=np.array([-10.0, -10.0, bottom_z], dtype=np.float64),
        max_rel=np.array([10.0, 10.0, bottom_z + 10.0], dtype=np.float64),
    )
    while len(ems_list) <= cand.ems_id:
        ems_list.append(ems)
    ems_list[cand.ems_id] = ems


# --- support_ratio ---------------------------------------------------------

def test_support_ratio_flat_floor_is_one():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    assert support_ratio(state, cand) == pytest.approx(1.0)


def test_support_ratio_half_on_platform_is_half():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert support_ratio(state, cand) == pytest.approx(0.5)


def test_support_ratio_floating_is_zero():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=0.50)
    _register_ems(state, cand)
    assert support_ratio(state, cand) == pytest.approx(0.0)


def test_support_ratio_outside_footprint_is_zero():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(x=10.0, bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    assert support_ratio(state, cand) == pytest.approx(0.0)


def test_support_ratio_tol_contact_boundary():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])

    inside_tol = _candidate(bottom_z=0.30 + 0.8 * TOL_CONTACT)
    _register_ems(state, inside_tol)
    assert support_ratio(state, inside_tol) == pytest.approx(1.0)

    outside_tol = _candidate(bottom_z=0.30 + 1.2 * TOL_CONTACT)
    _register_ems(state, outside_tol)
    assert support_ratio(state, outside_tol) == pytest.approx(0.0)


def test_support_ratio_return_type_is_float_and_bounded():
    from src.packing_core.stability import support_ratio

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    result = support_ratio(state, cand)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


# --- max_step_below ----------------------------------------------------------

def test_max_step_below_flat_is_zero():
    from src.packing_core.stability import max_step_below

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    assert max_step_below(state, cand) == pytest.approx(0.0)


def test_max_step_below_platform_step_includes_nonsupport():
    from src.packing_core.stability import max_step_below

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    # footprint内: 左半分 height=0.30、右半分 height=floor_z=0.02（非支持だが対象に含む）
    assert max_step_below(state, cand) == pytest.approx(0.30 - FLOOR_Z)


def test_max_step_below_empty_footprint_is_zero():
    from src.packing_core.stability import max_step_below

    state = _state_with_placed([])
    cand = _candidate(x=10.0, bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    assert max_step_below(state, cand) == pytest.approx(0.0)


def test_max_step_below_is_nonnegative_float():
    from src.packing_core.stability import max_step_below

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    result = max_step_below(state, cand)
    assert isinstance(result, float)
    assert result >= 0.0


# --- soft_below_ratio ----------------------------------------------------------

def test_soft_below_ratio_all_soft_is_one():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(1.0)


def test_soft_below_ratio_half_soft_half_hard_is_half():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(0.0, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(0.5)


def test_soft_below_ratio_half_floor_half_soft_is_half():
    from src.packing_core.stability import soft_below_ratio

    soft_top_z = FLOOR_Z + 0.8 * TOL_CONTACT
    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=soft_top_z, is_soft=True)
    # 右半分は床のまま（PlacedItemを置かない）。
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=soft_top_z)
    _register_ems(state, cand)
    # 床は candidate_bottom_z との差 < TOL_CONTACT のため支持セルとなり分母に入るが、
    # ソフト最上面ではないため分子には入らない。
    assert soft_below_ratio(state, cand) == pytest.approx(0.5)


def test_soft_below_ratio_hard_only_is_zero():
    from src.packing_core.stability import soft_below_ratio

    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([hard])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_no_support_is_zero():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    state = _state_with_placed([soft])
    cand = _candidate(bottom_z=0.50)  # 支持セル0件
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_tie_soft_and_hard_counts_soft():
    from src.packing_core.stability import soft_below_ratio

    # §4.6の退化ケース: 同一最上面(z=0.30)にソフト・ハードが並存 → 保守側にソフトとして数える。
    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(1.0)


def test_soft_below_ratio_soft_below_top_not_counted():
    from src.packing_core.stability import soft_below_ratio

    # ソフト(z=[0.02,0.20])の上にハード(z=[0.20,0.30])を体積非重複で物理的に積み重ねる。
    # space.heightを形成する最上面はハードのため、ソフトは支持として数えない。
    soft = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.20, is_soft=True)
    hard = _placed_item(-0.4, 0.4, -0.4, 0.4, z_bottom=0.20, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert soft_below_ratio(state, cand) == pytest.approx(0.0)


def test_soft_below_ratio_return_type_is_float_and_bounded():
    from src.packing_core.stability import soft_below_ratio

    soft = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=True)
    hard = _placed_item(0.0, 0.4, -0.4, 0.4, z_bottom=0.02, z_top=0.30, is_soft=False)
    state = _state_with_placed([soft, hard])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    result = soft_below_ratio(state, cand)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


# --- container_idx 検証（3公開関数共通の _footprint 契約。代表関数で検証） -----------------

def test_container_idx_negative_raises_value_error():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z, container_idx=-1)
    _register_ems(state, cand)
    with pytest.raises(ValueError):
        support_ratio(state, cand)


def test_container_idx_out_of_range_raises_value_error():
    from src.packing_core.stability import support_ratio

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z, container_idx=len(state.containers))
    _register_ems(state, cand)
    with pytest.raises(ValueError):
        support_ratio(state, cand)


# --- T-021: support_polygon ------------------------------------------------
# 確定仕様（案B′、人間確定済み。docs/実装詳細仕様書.md §4.6 v1.14追補）：
# 支持セルが表す矩形接触パッチ（各セルの生矩形を候補底面AABB・コンテナ内壁XYでクリップ
# したもの）の全頂点の凸包。セル「中心点」のみの凸包ではない。

def test_support_polygon_flat_floor_matches_candidate_boundary():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    expected = np.array(
        [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]], dtype=np.float64
    )
    np.testing.assert_allclose(poly, expected, atol=1e-9)


def test_support_polygon_no_support_is_empty_shape():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    cand = _candidate(bottom_z=0.50)  # 浮遊、支持0件
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    assert poly.shape == (0, 2)
    assert poly.dtype == np.float64


def test_support_polygon_single_supported_cell_returns_intersection_rectangle():
    from src.packing_core.stability import support_polygon

    # セル(0,0)（生矩形 x:[-0.4,-0.38], y:[-0.4,-0.38]）だけを支持する小さな台。
    platform = _placed_item(-0.4, -0.38, -0.4, -0.38, z_bottom=FLOOR_Z, z_top=0.05)
    state = _state_with_placed([platform])
    # 候補footprintをそのセルと厳密に一致させる（osize=cellと同寸、中心をセル中心に一致）。
    cand = _candidate(x=-0.39, y=-0.39, bottom_z=0.05, osize=(CELL, CELL, 0.1))
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    expected = np.array(
        [[-0.4, -0.4], [-0.38, -0.4], [-0.38, -0.38], [-0.4, -0.38]], dtype=np.float64
    )
    np.testing.assert_allclose(poly, expected, atol=1e-9)


def test_support_polygon_clips_to_candidate_footprint_when_off_grid():
    from src.packing_core.stability import support_polygon

    # 候補中心を格子に対して非整列（x=0.011）にしても、支持パッチは候補底面AABBの外へ出ない。
    state = _state_with_placed([])
    cand = _candidate(x=0.011, y=0.0, bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    amin = cand.pos_rel[:2] - cand.osize[:2] / 2.0
    amax = cand.pos_rel[:2] + cand.osize[:2] / 2.0
    assert np.all(poly[:, 0] >= amin[0] - EPS_GEOM)
    assert np.all(poly[:, 0] <= amax[0] + EPS_GEOM)
    assert np.all(poly[:, 1] >= amin[1] - EPS_GEOM)
    assert np.all(poly[:, 1] <= amax[1] + EPS_GEOM)


def test_support_polygon_clips_to_container_inner_wall():
    from src.packing_core.stability import support_polygon

    # 候補底面が内壁(x<=0.4)をまたぐ配置（中心x=0.39→amax_x=0.59>0.4）。支持セル自体は
    # cells_of_aabb がセル中心で内壁clampする一方、セル「生矩形」は内壁を越え得るため、
    # support_polygon 自身が内壁XYでクリップする契約を単体で検証する。
    state = _state_with_placed([])
    cand = _candidate(x=0.39, y=0.0, bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    assert np.all(poly[:, 0] <= INNER_MAX_REL[0] + EPS_GEOM)
    assert np.all(poly[:, 0] >= INNER_MIN_REL[0] - EPS_GEOM)
    assert np.all(poly[:, 1] <= INNER_MAX_REL[1] + EPS_GEOM)
    assert np.all(poly[:, 1] >= INNER_MIN_REL[1] - EPS_GEOM)


def test_support_polygon_is_counterclockwise():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    # shoelace公式による符号付き面積。正なら反時計回り。
    x, y = poly[:, 0], poly[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed_area > 0.0


def test_support_polygon_lexicographically_smallest_vertex_first():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    first = tuple(poly[0])
    for row in poly[1:]:
        assert first <= tuple(row)


def test_support_polygon_first_vertex_not_duplicated_at_end():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    poly = support_polygon(state, cand)
    assert not np.array_equal(poly[0], poly[-1])


def test_support_polygon_dtype_and_shape_contract():
    from src.packing_core.stability import support_polygon

    state = _state_with_placed([])
    for bottom_z in (FLOOR_Z, 0.50):  # 支持あり／支持0件の両方
        cand = _candidate(bottom_z=bottom_z)
        _register_ems(state, cand)
        poly = support_polygon(state, cand)
        assert poly.dtype == np.float64
        assert poly.ndim == 2
        assert poly.shape[1] == 2


def test_support_polygon_independent_of_placed_list_order():
    from src.packing_core.stability import support_polygon

    plat_a = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=0.10)
    plat_b = _placed_item(0.0, 0.4, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=0.30)
    cand = _candidate(bottom_z=0.30)
    state_ab = _state_with_placed([plat_a, plat_b])
    state_ba = _state_with_placed([plat_b, plat_a])
    _register_ems(state_ab, cand)
    _register_ems(state_ba, cand)

    poly_ab = support_polygon(state_ab, cand)
    poly_ba = support_polygon(state_ba, cand)
    np.testing.assert_array_equal(poly_ab, poly_ba)


# --- T-021: _convex_hull（private helper。重複点・共線点除去を合成入力で直接検証） ----------

def test_convex_hull_removes_duplicate_points():
    from src.packing_core.stability import _convex_hull

    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    hull = _convex_hull(np.unique(pts, axis=0))
    assert hull.shape == (4, 2)
    # 重複除去後も四隅がすべて残る（面積を持つ四角形のまま縮退しない）。
    expected = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    assert {tuple(v) for v in hull} == expected


def test_convex_hull_removes_collinear_midpoints():
    from src.packing_core.stability import _convex_hull

    # ユーザー確定の合成入力：正の面積を持つ支持パッチからは通常発生しない縮退ケースを
    # _convex_hull へ直接合成点で検証する（ゼロ幅Candidateやゼロ面積セルは作らない）。
    pts = np.array(
        [[-0.2, 0.0], [-0.1, 0.0], [0.0, 0.0], [0.2, 0.0]], dtype=np.float64
    )
    hull = _convex_hull(pts)
    expected = np.array([[-0.2, 0.0], [0.2, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(hull, expected, atol=1e-9)


def test_convex_hull_empty_and_single_point_shapes():
    from src.packing_core.stability import _convex_hull

    empty = _convex_hull(np.empty((0, 2), dtype=np.float64))
    assert empty.shape == (0, 2)
    assert empty.dtype == np.float64

    single = _convex_hull(np.array([[1.0, 2.0]], dtype=np.float64))
    assert single.shape == (1, 2)
    np.testing.assert_allclose(single, [[1.0, 2.0]])


# --- T-021: cg_margin --------------------------------------------------------

def test_cg_margin_flat_floor_is_half_min_osize():
    from src.packing_core.stability import cg_margin

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    expected = min(cand.osize[0], cand.osize[1]) / 2.0
    assert cg_margin(state, cand) == pytest.approx(expected, abs=1e-3)


def test_cg_margin_half_support_is_near_zero_boundary():
    from src.packing_core.stability import cg_margin

    platform = _placed_item(-0.4, 0.0, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    assert cg_margin(state, cand) == pytest.approx(0.0, abs=1e-3)


def test_cg_margin_outside_support_hull_is_negative():
    from src.packing_core.stability import cg_margin

    # 台が候補中心(0,0)を覆わない位置（x:[-0.4,-0.1]）にあり、候補中心はhull外側になる。
    platform = _placed_item(-0.4, -0.1, -0.4, 0.4, z_bottom=FLOOR_Z, z_top=0.30)
    state = _state_with_placed([platform])
    cand = _candidate(bottom_z=0.30)
    _register_ems(state, cand)
    result = cg_margin(state, cand)
    assert np.isfinite(result)
    assert result < 0.0


def test_cg_margin_k_less_than_3_is_negative_infinity_synthetic():
    from src.packing_core.stability import _cg_margin_from_polygon

    # ゼロ幅Candidateやゼロ面積セルを作らず、合成2点をprivate helperへ直接渡して検証する
    # （ユーザー確定の検証方法）。
    two_points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    assert _cg_margin_from_polygon(two_points, np.array([0.5, 0.0])) == float("-inf")

    one_point = np.array([[0.0, 0.0]], dtype=np.float64)
    assert _cg_margin_from_polygon(one_point, np.array([0.0, 0.0])) == float("-inf")

    zero_points = np.empty((0, 2), dtype=np.float64)
    assert _cg_margin_from_polygon(zero_points, np.array([0.0, 0.0])) == float("-inf")


def test_cg_margin_inside_boundary_outside_signs_synthetic():
    from src.packing_core.stability import _cg_margin_from_polygon

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    assert _cg_margin_from_polygon(square, np.array([0.5, 0.5])) > 0.0
    assert _cg_margin_from_polygon(square, np.array([1.0, 0.5])) == pytest.approx(0.0, abs=1e-9)
    assert _cg_margin_from_polygon(square, np.array([1.5, 0.5])) < 0.0


def test_cg_margin_uses_vertex_distance_outside_corner_synthetic():
    from src.packing_core.stability import _cg_margin_from_polygon

    # 頂点の外側（対角方向）では、辺の延長線ではなく頂点への距離が使われることを検証する。
    # 射影係数[0,1]クランプが無いと辺(1,0)-(1,1)の無限直線距離1.0を誤って返してしまう。
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    result = _cg_margin_from_polygon(square, np.array([2.0, 2.0]))
    assert result == pytest.approx(-float(np.sqrt(2.0)), abs=1e-9)


def test_cg_margin_return_type_is_float():
    from src.packing_core.stability import cg_margin

    state = _state_with_placed([])
    cand = _candidate(bottom_z=FLOOR_Z)
    _register_ems(state, cand)
    result = cg_margin(state, cand)
    assert isinstance(result, float)
