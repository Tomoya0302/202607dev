"""T-006/T-007: packing_core.container_space のcut・棚ケース検証
（詳細仕様書 §2・§3・§4.2・§6 T-006/T-007、interface_notes.md §K）。

container_space.py は T-007時点でもcut/棚対応が未実装のため、収集(--collect-only)を成功させるべく
各テスト関数の内部で対象モジュールを import する（モジュール直下 import は禁止、test_geometry.py と同方針）。
T-007の新規テストは、本体未実装により失敗することが想定されている（DoD検収は別セッション）。

cutなしフィクスチャについて:
    公式 `write_open_cut_corner_cup_obj`（ground_handling/utils.py）は `cut_x/cut_y<=0` を
    ValueError にするため、cut=0 の実データは公式パイプラインから生成できない。
    そのため本ファイルでは「軸整列6面（±X/±Y/±Z）の points/n_vecs」を直接組み立てた合成 cdict を用いる。
    interface_notes.md §I-7 の解決方針に従い、buffer キーは cdict に含めず、
    inner_min_rel/inner_max_rel は points（世界座標の代表点）と n_vecs（外向き法線）から
    半空間 `normal_rel・x <= d` の交差として復元される前提でフィクスチャを作る
    （6面すべて軸整列のため、この交差はそのまま AABB になり、cut_planes は空になるはず）。

    interface_notes.md §K.5 の方針により、既定の `cut_x`/`cut_y` は 0.0（真のcutなし直方体）に
    修正済み（旧: 0.3。小棚の無条件生成則と矛盾するため）。`cut_x`/`cut_y`/`shelf` は
    T-007の追加テスト（Fixture C：小棚・大棚の生成条件を独立に検証する）向けに引数化してある。

T-007 golden fixture について:
    Fixture A/B（cutあり実ジオメトリ）は `tests/fixtures/container_space_golden.py` が公式
    `write_open_cut_corner_cup_obj`/`aff` を直接呼び出して構築する。期待値（floor_z/ceil_z/
    shelf AABB/格子体積）は同モジュールが `container_space.py` を使わず独立計算する
    （interface_notes.md §K.9）。
"""
import numpy as np
import pytest

from fixtures import container_space_golden as golden
from src.packing_core import constants

# セル境界が内壁寸法をちょうど割り切る値を選び、格子丸め誤差の影響を排除する。
INNER_MIN_REL = np.array([-0.48, -0.73, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.48, 0.73, 1.58], dtype=np.float64)
CELL = constants.GridParams().cell  # 0.02
NX = 48  # (0.48 - (-0.48)) / 0.02
NY = 73  # (0.73 - (-0.73)) / 0.02
INNER_VOLUME = float(np.prod(INNER_MAX_REL - INNER_MIN_REL))  # 2.186496
EPS_GEOM = constants.EPS_GEOM

# Fixture C（軸整列直方体）を再構成するための寸法（_box_cdict と自己無矛盾）。
FIXTURE_C_THICKNESS = 0.02
FIXTURE_C_BUFFER = 0.02
FIXTURE_C_LENGTH = float(INNER_MAX_REL[0] - INNER_MIN_REL[0]) + 2 * FIXTURE_C_THICKNESS
FIXTURE_C_WIDTH = float(INNER_MAX_REL[1] - INNER_MIN_REL[1]) + 2 * FIXTURE_C_THICKNESS
FIXTURE_C_HEIGHT = float(INNER_MAX_REL[2]) + FIXTURE_C_BUFFER


def _box_cdict(
    index: int = 0,
    spacing: float = 2.0,
    inner_min_rel: np.ndarray = INNER_MIN_REL,
    inner_max_rel: np.ndarray = INNER_MAX_REL,
    cut_x: float = 0.0,
    cut_y: float = 0.0,
    shelf: bool = False,
    offset_x: float | None = None,
) -> dict:
    """軸整列直方体コンテナの合成 cdict（container_list要素）を組み立てる。

    6面（±X/±Y/±Z）それぞれの代表点（世界座標）と外向き法線から points/n_vecs を構成する
    （`cut_x`/`cut_y`/`shelf` の値に関わらず、ジオメトリは常に純粋な軸整列直方体であり
    `cut_planes` は常に空になる）。buffer キーは含めない（interface_notes.md §I-7）。
    center.z は公式の `pos=(0,0,height/2+buffer)` 相当になるよう、height/thickness/buffer を
    自己無矛盾に設定する。

    既定値 `cut_x=0.0, cut_y=0.0` は「真のcutなし直方体」を表す（interface_notes.md §K.5）。
    `cut_x>0` を指定した場合でも本関数のジオメトリは直方体のままであり、これは
    「小棚は `cut_planes`/`shelf` に依存せず `cut_x` の値だけで計算される」という
    §K.2 の生成条件を独立に検証するための Fixture C として意図的に用いる。

    `offset_x` を明示指定すると、`index * spacing` の代わりにその値を `center`/`points` の
    原点世界Xとして使う（`index * spacing` では表現できない非等間隔配置の回帰テスト用、
    interface_notes.md §I-8）。省略時（`None`）は従来通り `index * spacing` を使用する。
    """
    if offset_x is None:
        offset_x = index * spacing
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
    points = [(px + offset_x, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]

    thickness = 0.02
    buffer = 0.02  # 既定値0.01ではなく、config上書きされ得る値であることを示すため別値にする
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer  # ceil_z(=inner_max.z) = height - buffer と自己無矛盾

    assert abs(imin[2] - thickness) < 1e-12  # floor_z(=inner_min.z) = thickness と自己無矛盾

    center = (offset_x, 0.0, height / 2.0 + buffer)

    return {
        "index": index,
        "length": length,
        "width": width,
        "height": height,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "thickness": thickness,
        "center": center,
        "n_vecs": n_vecs,
        "points": points,
        "volume": INNER_VOLUME if np.array_equal(imin, INNER_MIN_REL) and np.array_equal(imax, INNER_MAX_REL) else float(np.prod(imax - imin)),
        "shelf": shelf,
        "is_prioritized": False,
        "packed_items": [],
    }


# --- ContainerSpace の基本フィールド -------------------------------------------------

def test_container_space_dataclass_fields():
    import dataclasses

    from src.packing_core.container_space import ContainerSpace

    field_names = {f.name for f in dataclasses.fields(ContainerSpace)}
    assert field_names == {
        "index", "offset_x", "inner_min_rel", "inner_max_rel",
        "cut_planes", "shelf_boxes", "cell", "floor_z", "ceil_z", "height",
    }


# --- build_container_space（cutなし） --------------------------------------------------

def test_build_container_space_basic_index_and_offset():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert space.index == 0
    assert space.offset_x == pytest.approx(0.0)
    assert space.cell == pytest.approx(CELL)


def test_build_container_space_inner_bounds_recovered_from_points_and_n_vecs():
    # interface_notes.md §I-7: buffer固定値ではなく points/n_vecs の半空間交差から復元される。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    np.testing.assert_allclose(space.inner_min_rel, INNER_MIN_REL, atol=1e-9)
    np.testing.assert_allclose(space.inner_max_rel, INNER_MAX_REL, atol=1e-9)


def test_build_container_space_no_cut_planes_and_no_shelf():
    # cutなし・棚なし: 6面すべて軸整列のため cut_planes は空、shelf_boxes も空になる。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert list(space.cut_planes) == []
    assert list(space.shelf_boxes) == []


# --- floor_z / ceil_z / height の初期化 --------------------------------------------------

def test_floor_z_initialized_at_inner_wall_bottom():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert space.floor_z.shape == (NX, NY)
    assert space.floor_z.dtype == np.float64
    np.testing.assert_allclose(space.floor_z, np.full((NX, NY), INNER_MIN_REL[2]), atol=1e-9)


def test_ceil_z_initialized_at_inner_wall_top():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert space.ceil_z.shape == (NX, NY)
    assert space.ceil_z.dtype == np.float64
    np.testing.assert_allclose(space.ceil_z, np.full((NX, NY), INNER_MAX_REL[2]), atol=1e-9)


def test_height_initialized_at_initial_floor_height():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert space.height.shape == (NX, NY)
    assert space.height.dtype == np.float64
    np.testing.assert_allclose(space.height, space.floor_z, atol=1e-9)

    # height は floor_z のコピーであり、同一配列を共有してはならない
    # （bake_placed が height だけを更新できることを保証するため）。
    space.height[0, 0] += 1.0
    assert space.floor_z[0, 0] == pytest.approx(INNER_MIN_REL[2])


# --- contains_oriented_box（cutなし） -----------------------------------------------------

def test_contains_oriented_box_inside_is_true():
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    center_rel = np.array([0.0, 0.0, (INNER_MIN_REL[2] + INNER_MAX_REL[2]) / 2.0], dtype=np.float64)
    osize = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is True


def test_contains_oriented_box_poking_through_side_wall_is_false():
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    # +X 側の内壁(0.48)を0.1mはみ出す位置に配置
    center_rel = np.array([INNER_MAX_REL[0] - 0.1, 0.0, 0.8], dtype=np.float64)
    osize = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is False


def test_contains_oriented_box_poking_through_ceiling_is_false():
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    # 天井(1.58)を0.1mはみ出す位置に配置
    center_rel = np.array([0.0, 0.0, INNER_MAX_REL[2] - 0.1], dtype=np.float64)
    osize = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is False


# --- effective_volume -----------------------------------------------------------------------

def test_effective_volume_matches_inner_wall_volume_within_1_percent():
    # ContainerSpace は cdict をそのまま保持しない（volume フィールドを持たない）ため、
    # effective_volume(space) は cdict["volume"] へアクセスしようがなく、
    # 「そのまま返す」実装は構造上不可能。したがって関係式のみを検証する。
    from src.packing_core.container_space import build_container_space, effective_volume

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, cell=CELL)

    result = effective_volume(space)
    relative_error = abs(result - cdict["volume"]) / cdict["volume"]
    assert relative_error < 0.01


# --- offset_x = cdict["center"][0]（並進不変性込み） --------------------------------------
# interface_notes.md §I-8: offset_x は cdict["center"][0] を直接使用する。以下の fixture では
# center[0] がたまたま index*spacing の値になるよう構成しているが、アサーションの根拠は
# あくまで cdict["center"][0] とする（index*spacing の再計算ではない）。

def test_offset_x_uses_reported_center():
    from src.packing_core.container_space import build_container_space

    index = 3
    spacing = 1.8
    cdict = _box_cdict(index=index, spacing=spacing)
    space = build_container_space(cdict, index=index, cell=CELL)

    assert space.offset_x == pytest.approx(cdict["center"][0])

    # inner_min_rel/inner_max_rel はコンテナ相対座標であり、offset_x（並進）に依存しないはず。
    np.testing.assert_allclose(space.inner_min_rel, INNER_MIN_REL, atol=1e-9)
    np.testing.assert_allclose(space.inner_max_rel, INNER_MAX_REL, atol=1e-9)


def test_offset_x_uses_reported_center_for_nonuniform_layout():
    """center.x が単一の index*spacing では表現できない非等間隔配置でも、
    build_container_space は cdict["center"][0] をそのまま offset_x として採用する
    （interface_notes.md §I-8 の回帰テスト）。"""
    from src.packing_core.container_space import build_container_space

    cdict_a = _box_cdict(index=2, offset_x=0.37)
    cdict_b = _box_cdict(index=5, offset_x=5.93)

    space_a = build_container_space(cdict_a, index=2, cell=CELL)
    space_b = build_container_space(cdict_b, index=5, cell=CELL)

    assert space_a.offset_x == pytest.approx(0.37)
    assert space_a.offset_x == pytest.approx(cdict_a["center"][0])

    assert space_b.offset_x == pytest.approx(5.93)
    assert space_b.offset_x == pytest.approx(cdict_b["center"][0])


# ============================================================================
# T-007: cut・棚の変換規則（interface_notes.md §K、実装詳細仕様書.md §4.2）
#
# 以下は container_space.py の cut/棚対応（T-007）が未実装のため失敗することが想定されている。
# cut_planes 自体はT-006の汎用ロジックで既に構築されるが、floor_z/ceil_z の可変化、
# shelf_boxes 構築、contains_oriented_box の拡張、effective_volume の格子積分は未実装のため。
# ============================================================================


def _fixture_ab_expected(shelf: bool):
    """Fixture A（shelf=False）／Fixture B（shelf=True）の cdict と golden 中間値を返す。

    golden 側の計算は `container_space.py` を一切使わない独立実装
    （`tests/fixtures/container_space_golden.py`、interface_notes.md §K.9）。
    """
    cdict = golden.build_fixture_ab_cdict(index=0, spacing=2.0, shelf=shelf)
    inner_min, inner_max, cut_planes = golden.classify_planes(cdict["points"], cdict["n_vecs"])
    shelf_boxes = golden.expected_shelf_boxes(
        golden.FIXTURE_AB_LENGTH,
        golden.FIXTURE_AB_WIDTH,
        golden.FIXTURE_AB_HEIGHT,
        golden.FIXTURE_AB_THICKNESS,
        golden.FIXTURE_AB_CUT_X,
        golden.FIXTURE_AB_BUFFER,
        shelf,
        inner_min,
        inner_max,
    )
    floor_z, ceil_z = golden.expected_floor_ceil(inner_min, inner_max, cut_planes, shelf_boxes, golden.CELL)
    return cdict, inner_min, inner_max, cut_planes, shelf_boxes, floor_z, ceil_z


# --- Fixture C: 小棚・大棚の生成条件（cut_planes/shelfフラグからの独立性、§K.2/K.3） ------------


def test_small_shelf_computed_regardless_of_cut_planes_and_shelf_flag():
    # cut_x=0.3 だが幾何は純粋な直方体（cut_planes=[]）。shelf=False でも小棚は計算されるはず
    # （interface_notes.md §K.2: 小棚は cdict["shelf"]・cut_planes の有無に関わらず常時計算）。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(cut_x=0.3, cut_y=0.3, shelf=False)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert list(space.cut_planes) == []  # 幾何上は非軸整列面なし
    assert len(space.shelf_boxes) == 1  # 小棚のみ（大棚は shelf=False のため無し）


def test_small_shelf_not_added_when_clipped_volume_is_zero():
    # cut_x=0.0 では小棚の半径が0となり、クリップ後AABBがゼロ体積になるため追加されないはず
    # （§K.2/K.4 の正体積フィルタ）。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(cut_x=0.0, cut_y=0.0, shelf=False)
    space = build_container_space(cdict, index=0, cell=CELL)

    assert list(space.shelf_boxes) == []


def test_main_shelf_absent_when_shelf_flag_false():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(cut_x=0.3, cut_y=0.3, shelf=False)
    space = build_container_space(cdict, index=0, cell=CELL)

    # 小棚のみ（大棚に相当する2個目のAABBは含まれない）。
    assert len(space.shelf_boxes) == 1


def test_main_shelf_present_when_shelf_flag_true():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(cut_x=0.3, cut_y=0.3, shelf=True)
    space = build_container_space(cdict, index=0, cell=CELL)

    # 小棚 + 大棚 の2個。
    assert len(space.shelf_boxes) == 2


def test_shelf_boxes_aabb_matches_golden_for_fixture_c():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(cut_x=0.3, cut_y=0.3, shelf=True)
    space = build_container_space(cdict, index=0, cell=CELL)

    expected = golden.expected_shelf_boxes(
        FIXTURE_C_LENGTH,
        FIXTURE_C_WIDTH,
        FIXTURE_C_HEIGHT,
        FIXTURE_C_THICKNESS,
        cut_x=0.3,
        buffer=FIXTURE_C_BUFFER,
        shelf=True,
        inner_min=INNER_MIN_REL,
        inner_max=INNER_MAX_REL,
    )
    assert len(space.shelf_boxes) == len(expected) == 2

    # 順序非依存で比較する（小棚・大棚の追加順序は実装依存とみなす）。
    actual_sorted = sorted(space.shelf_boxes, key=lambda box: tuple(np.asarray(box[0])))
    expected_sorted = sorted(expected, key=lambda box: tuple(np.asarray(box[0])))
    for (a_min, a_max), (e_min, e_max) in zip(actual_sorted, expected_sorted):
        np.testing.assert_allclose(np.asarray(a_min), e_min, rtol=0.0, atol=EPS_GEOM)
        np.testing.assert_allclose(np.asarray(a_max), e_max, rtol=0.0, atol=EPS_GEOM)


# --- Fixture A/B: cutあり実ジオメトリ（floor_z/ceil_z/contains_oriented_box、§K.6/K.7/K.9） -----


def test_floor_z_matches_golden_for_cut_fixture():
    # Fixture A（shelf=False）。floor_z は cut_planes のみに依存し、棚の有無とは無関係。
    from src.packing_core.container_space import build_container_space

    cdict, inner_min, inner_max, cut_planes, shelf_boxes, floor_z_golden, ceil_z_golden = (
        _fixture_ab_expected(shelf=False)
    )
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    np.testing.assert_allclose(space.floor_z, floor_z_golden, rtol=0.0, atol=EPS_GEOM)


def test_ceil_z_matches_golden_for_shelf_fixture():
    # Fixture B（shelf=True）。ceil_z は内壁天井と棚下面（小棚+大棚）両方のcapを反映する。
    from src.packing_core.container_space import build_container_space

    cdict, inner_min, inner_max, cut_planes, shelf_boxes, floor_z_golden, ceil_z_golden = (
        _fixture_ab_expected(shelf=True)
    )
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    np.testing.assert_allclose(space.ceil_z, ceil_z_golden, rtol=0.0, atol=EPS_GEOM)


def test_contains_oriented_box_false_when_poking_into_cut_wedge():
    # Fixture A。cut平面で削られた領域（x<=-0.2917付近、床がz≈0.19まで持ち上がる側）に
    # 沈み込む箱はFalseになるはず。
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict, *_ = _fixture_ab_expected(shelf=False)
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    center_rel = np.array([-0.44, 0.0, 0.10], dtype=np.float64)
    osize = np.array([0.04, 0.20, 0.10], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is False


def test_contains_oriented_box_true_on_raised_floor():
    # 同じX帯だが、cutで持ち上がった床（x=-0.44でz≈0.188）より上に置けばTrueになるはず。
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict, *_ = _fixture_ab_expected(shelf=False)
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    center_rel = np.array([-0.44, 0.0, 0.30], dtype=np.float64)
    osize = np.array([0.04, 0.20, 0.10], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is True


def test_contains_oriented_box_false_when_overlapping_main_shelf():
    # Fixture B（shelf=True）。大棚クリップ後AABB（Y∈[0.02,0.48], Z∈[0.52,0.54]付近）と
    # 重なる箱はFalseになるはず。
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict, *_ = _fixture_ab_expected(shelf=True)
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    center_rel = np.array([0.0, 0.25, 0.53], dtype=np.float64)
    osize = np.array([0.10, 0.10, 0.02], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is False


# --- effective_volume: DoD①（直方体基準）／DoD②（cut・棚fixture、golden格子積分と一致） --------
# （実装詳細仕様書.md §4.2 T-007節、interface_notes.md §K.10）


def test_effective_volume_rectangular_baseline_matches_analytic_volume():
    # DoD①: cut_x=0, cut_y=0, shelf=False, 小棚クリップ後ゼロ体積の直方体基準ケースでは、
    # 解析的な内壁体積との相対誤差<1%を維持する。
    from src.packing_core.container_space import build_container_space, effective_volume

    cdict = _box_cdict(cut_x=0.0, cut_y=0.0, shelf=False)
    space = build_container_space(cdict, index=0, cell=CELL)

    expected = golden.expected_analytic_box_volume(INNER_MIN_REL, INNER_MAX_REL)
    result = effective_volume(space)
    relative_error = abs(result - expected) / expected
    assert relative_error < 0.01


def test_effective_volume_matches_independent_grid_integral_for_cut_fixture():
    # DoD②: cdict["volume"]（公式体積式）との一致は求めない。golden floor_z/ceil_zからの
    # 独立格子積分とのみ一致することを検証する。
    from src.packing_core.container_space import build_container_space, effective_volume

    cdict, inner_min, inner_max, cut_planes, shelf_boxes, floor_z_golden, ceil_z_golden = (
        _fixture_ab_expected(shelf=False)
    )
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    expected_grid_volume = golden.expected_grid_volume(floor_z_golden, ceil_z_golden, golden.CELL)
    result = effective_volume(space)
    np.testing.assert_allclose(result, expected_grid_volume, rtol=0.0, atol=EPS_GEOM)

    # DoD③: 公式体積式との差は診断値としてのみ記録し、合否条件にはしない
    # （interface_notes.md §K.10。数値は fixture 入力から都度計算し、ハードコードしない）。
    # 変数として保持するのみで assert しない（合否条件にしない、interface_notes.md §K.10）。
    diagnostic_relative_error_vs_official_volume = abs(result - cdict["volume"]) / cdict["volume"]
    assert isinstance(diagnostic_relative_error_vs_official_volume, float)  # 計算のみ記録、閾値判定はしない


def test_effective_volume_matches_independent_grid_integral_for_cut_and_shelf_fixture():
    from src.packing_core.container_space import build_container_space, effective_volume

    cdict, inner_min, inner_max, cut_planes, shelf_boxes, floor_z_golden, ceil_z_golden = (
        _fixture_ab_expected(shelf=True)
    )
    space = build_container_space(cdict, index=0, cell=golden.CELL)

    expected_grid_volume = golden.expected_grid_volume(floor_z_golden, ceil_z_golden, golden.CELL)
    result = effective_volume(space)
    np.testing.assert_allclose(result, expected_grid_volume, rtol=0.0, atol=EPS_GEOM)

    # DoD③: 診断値のみ（合否条件にしない）。
    # 変数として保持するのみで assert しない（合否条件にしない、interface_notes.md §K.10）。
    diagnostic_relative_error_vs_official_volume = abs(result - cdict["volume"]) / cdict["volume"]
    assert isinstance(diagnostic_relative_error_vs_official_volume, float)  # 計算のみ記録、閾値判定はしない
