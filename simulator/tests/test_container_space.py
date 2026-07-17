"""T-006: packing_core.container_space のcutなし直方体ケース検証
（詳細仕様書 §2・§3・§4.2・§6 T-006）。

container_space.py は本チケット時点では未実装のため、収集(--collect-only)を成功させるべく
各テスト関数の内部で対象モジュールを import する（モジュール直下 import は禁止、test_geometry.py と同方針）。

cutなしフィクスチャについて:
    公式 `write_open_cut_corner_cup_obj`（ground_handling/utils.py）は `cut_x/cut_y<=0` を
    ValueError にするため、cut=0 の実データは公式パイプラインから生成できない。
    そのため本ファイルでは「軸整列6面（±X/±Y/±Z）の points/n_vecs」を直接組み立てた合成 cdict を用いる。
    interface_notes.md §I-7 の解決方針に従い、buffer キーは cdict に含めず、
    inner_min_rel/inner_max_rel は points（世界座標の代表点）と n_vecs（外向き法線）から
    半空間 `normal_rel・x <= d` の交差として復元される前提でフィクスチャを作る
    （6面すべて軸整列のため、この交差はそのまま AABB になり、cut_planes は空になるはず）。
"""
import numpy as np
import pytest

from src.packing_core import constants

# セル境界が内壁寸法をちょうど割り切る値を選び、格子丸め誤差の影響を排除する。
INNER_MIN_REL = np.array([-0.48, -0.73, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.48, 0.73, 1.58], dtype=np.float64)
CELL = constants.GridParams().cell  # 0.02
NX = 48  # (0.48 - (-0.48)) / 0.02
NY = 73  # (0.73 - (-0.73)) / 0.02
INNER_VOLUME = float(np.prod(INNER_MAX_REL - INNER_MIN_REL))  # 2.186496


def _box_cdict(
    index: int = 0,
    spacing: float = 2.0,
    inner_min_rel: np.ndarray = INNER_MIN_REL,
    inner_max_rel: np.ndarray = INNER_MAX_REL,
) -> dict:
    """cutなし直方体コンテナの合成 cdict（container_list要素）を組み立てる。

    6面（±X/±Y/±Z）それぞれの代表点（世界座標）と外向き法線から points/n_vecs を構成する。
    buffer キーは含めない（interface_notes.md §I-7）。center.z は公式の
    `pos=(0,0,height/2+buffer)` 相当になるよう、height/thickness/buffer を自己無矛盾に設定する。
    """
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
        "cut_x": 0.3,
        "cut_y": 0.3,
        "thickness": thickness,
        "center": center,
        "n_vecs": n_vecs,
        "points": points,
        "volume": INNER_VOLUME if np.array_equal(imin, INNER_MIN_REL) and np.array_equal(imax, INNER_MAX_REL) else float(np.prod(imax - imin)),
        "shelf": False,
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
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    assert space.index == 0
    assert space.offset_x == pytest.approx(0.0)
    assert space.cell == pytest.approx(CELL)


def test_build_container_space_inner_bounds_recovered_from_points_and_n_vecs():
    # interface_notes.md §I-7: buffer固定値ではなく points/n_vecs の半空間交差から復元される。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    np.testing.assert_allclose(space.inner_min_rel, INNER_MIN_REL, atol=1e-9)
    np.testing.assert_allclose(space.inner_max_rel, INNER_MAX_REL, atol=1e-9)


def test_build_container_space_no_cut_planes_and_no_shelf():
    # cutなし・棚なし: 6面すべて軸整列のため cut_planes は空、shelf_boxes も空になる。
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    assert list(space.cut_planes) == []
    assert list(space.shelf_boxes) == []


# --- floor_z / ceil_z / height の初期化 --------------------------------------------------

def test_floor_z_initialized_at_inner_wall_bottom():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    assert space.floor_z.shape == (NX, NY)
    assert space.floor_z.dtype == np.float64
    np.testing.assert_allclose(space.floor_z, np.full((NX, NY), INNER_MIN_REL[2]), atol=1e-9)


def test_ceil_z_initialized_at_inner_wall_top():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    assert space.ceil_z.shape == (NX, NY)
    assert space.ceil_z.dtype == np.float64
    np.testing.assert_allclose(space.ceil_z, np.full((NX, NY), INNER_MAX_REL[2]), atol=1e-9)


def test_height_initialized_at_initial_floor_height():
    from src.packing_core.container_space import build_container_space

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

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
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    center_rel = np.array([0.0, 0.0, (INNER_MIN_REL[2] + INNER_MAX_REL[2]) / 2.0], dtype=np.float64)
    osize = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is True


def test_contains_oriented_box_poking_through_side_wall_is_false():
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    # +X 側の内壁(0.48)を0.1mはみ出す位置に配置
    center_rel = np.array([INNER_MAX_REL[0] - 0.1, 0.0, 0.8], dtype=np.float64)
    osize = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    assert contains_oriented_box(space, center_rel, osize, margin=0.0) is False


def test_contains_oriented_box_poking_through_ceiling_is_false():
    from src.packing_core.container_space import build_container_space, contains_oriented_box

    cdict = _box_cdict(index=0, spacing=2.0)
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

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
    space = build_container_space(cdict, index=0, spacing=2.0, cell=CELL)

    result = effective_volume(space)
    relative_error = abs(result - cdict["volume"]) / cdict["volume"]
    assert relative_error < 0.01


# --- offset_x = index * spacing（並進不変性込み） -----------------------------------------

def test_offset_x_equals_index_times_spacing():
    from src.packing_core.container_space import build_container_space

    index = 3
    spacing = 1.8
    cdict = _box_cdict(index=index, spacing=spacing)
    space = build_container_space(cdict, index=index, spacing=spacing, cell=CELL)

    assert space.offset_x == pytest.approx(index * spacing)

    # inner_min_rel/inner_max_rel はコンテナ相対座標であり、offset_x（並進）に依存しないはず。
    np.testing.assert_allclose(space.inner_min_rel, INNER_MIN_REL, atol=1e-9)
    np.testing.assert_allclose(space.inner_max_rel, INNER_MAX_REL, atol=1e-9)
