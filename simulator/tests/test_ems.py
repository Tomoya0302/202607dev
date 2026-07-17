"""T-009/T-010: packing_core.ems の全再構築・差分更新検証
（詳細仕様書 §2・§3・§4.3・§6 T-009/T-010。依存: T-005 geometry AABB系、T-007 cut/棚対応）。

ems.py は T-010時点で `update_ems`／`remove_contained` が未実装のため、
収集(--collect-only)を成功させるべく各テスト関数の内部で対象モジュールを
import する（モジュール直下 import は禁止、test_geometry.py・test_container_space.py
と同方針）。

本ファイルは T-009（generate_ems の全再構築、既知3EMSケース）と T-010（update_ems の
6面切断・remove_contained 公開API・全再構築との一致性質テスト）を対象とする。
T-011（select_topn/normalize_descriptors/打切り率/prune_min_dim）のテストは含めない
（update_ems は `remove_contained` までを行い、`prune_min_dim` は呼ばない）。
"""
import numpy as np
import pytest

from src.packing_core import constants, geometry
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


def _assert_no_redundant_containment(boxes) -> None:
    """boxes 内のいずれの要素も他の要素へ完全内包（境界一致含む）されていないことを検証する。

    完全一致する重複が残っていれば相互内包として検出される（remove_contained 後は
    重複も1個に絞られているべきという契約を性質テストとして裏付ける）。
    """
    for i, box in enumerate(boxes):
        for j, other in enumerate(boxes):
            if i == j:
                continue
            assert not geometry.aabb_contains(
                other.min_rel, other.max_rel, box.min_rel, box.max_rel, margin=0.0
            ), f"box {i} ({box.min_rel}, {box.max_rel}) is contained in box {j} ({other.min_rel}, {other.max_rel})"


def _random_non_overlapping_obstacles(
    rng: np.random.Generator,
    inner_min: np.ndarray,
    inner_max: np.ndarray,
    n: int,
    max_attempts: int = 50,
) -> list:
    """内壁AABB内に完全に収まる正体積のランダムAABBを n 個生成する。

    各配置は既存の配置と交差しないよう rejection sampling で試みるが、
    `max_attempts` 回で非交差の候補が見つからない場合は最後の候補をそのまま
    採用する（テストが乱数の巡り合わせで不安定に失敗しないようにするため。
    サイズは内壁寸法比 0.08〜0.3 に抑えており、通常は早期に非交差候補が
    見つかる）。
    """
    span = inner_max - inner_min
    obstacles: list = []
    for _ in range(n):
        candidate_min = candidate_max = None
        for _attempt in range(max_attempts):
            frac_size = rng.uniform(0.08, 0.3, size=3)
            size = frac_size * span
            start_frac = rng.uniform(0.0, 1.0, size=3) * (1.0 - frac_size)
            candidate_min = inner_min + start_frac * span
            candidate_max = candidate_min + size
            if not any(
                geometry.aabb_intersects(candidate_min, candidate_max, o_min, o_max, tol=0.0)
                for o_min, o_max in obstacles
            ):
                break
        obstacles.append((candidate_min, candidate_max))
    return obstacles


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


# =====================================================================================
# T-010: update_ems（差分更新）・remove_contained（公開API）
# =====================================================================================

# --- update_ems: 非交差のEMSは保持される --------------------------------------------

def test_update_ems_keeps_ems_untouched_when_no_intersection():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    initial = EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy())

    # 内壁の外側にある障害物: initial とは交差しない。
    obstacle_min = INNER_MAX_REL + np.array([1.0, 1.0, 1.0], dtype=np.float64)
    obstacle_max = obstacle_min + np.array([0.3, 0.3, 0.3], dtype=np.float64)

    result = ems.update_ems([initial], (obstacle_min, obstacle_max), space)

    assert len(result) == 1
    np.testing.assert_allclose(result[0].min_rel, INNER_MIN_REL, atol=1e-6)
    np.testing.assert_allclose(result[0].max_rel, INNER_MAX_REL, atol=1e-6)


# --- update_ems: 交差するEMSが障害物の6面で切断される --------------------------------

def test_update_ems_splits_ems_on_all_six_faces_for_interior_obstacle():
    """障害物がEMSの内部（境界に接しない）にある場合、6面切断の6個すべてが
    正体積となり、ちょうど6個の部分EMSが得られる（詳細仕様書 §4.3 の6面切断定義）。
    """
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    initial = EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy())
    obstacle_min = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    obstacle_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    result = ems.update_ems([initial], (obstacle_min, obstacle_max), space)

    expected = {
        _round_bounds(np.array([0.0, 0.0, 0.0]), np.array([0.5, 2.0, 1.6])),  # x < obstacle
        _round_bounds(np.array([1.0, 0.0, 0.0]), np.array([1.5, 2.0, 1.6])),  # x > obstacle
        _round_bounds(np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.5, 1.6])),  # y < obstacle
        _round_bounds(np.array([0.0, 1.0, 0.0]), np.array([1.5, 2.0, 1.6])),  # y > obstacle
        _round_bounds(np.array([0.0, 0.0, 0.0]), np.array([1.5, 2.0, 0.5])),  # z < obstacle
        _round_bounds(np.array([0.0, 0.0, 1.0]), np.array([1.5, 2.0, 1.6])),  # z > obstacle
    }
    assert len(result) == 6
    assert _rounded_box_set(result) == expected


# --- update_ems: 体積が0以下の部分箱は残らない ----------------------------------------

def test_update_ems_drops_zero_or_negative_volume_sub_boxes():
    """障害物がEMSの角（3面）に接する場合、6面切断のうち3面は退化（体積0）となり
    除去され、ちょうど3個の部分EMSが残る（T-009 の3EMSケースを update_ems 直接
    呼び出しで再現）。
    """
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    initial = EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy())
    box_min = INNER_MIN_REL.copy()
    box_max = INNER_MIN_REL + np.array([0.5, 0.5, 0.5], dtype=np.float64)

    result = ems.update_ems([initial], (box_min, box_max), space)

    assert len(result) == 3
    for box in result:
        assert np.all(box.max_rel - box.min_rel > 0.0)

    expected = {
        _round_bounds(
            np.array([0.5, INNER_MIN_REL[1], INNER_MIN_REL[2]]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
        _round_bounds(
            np.array([INNER_MIN_REL[0], 0.5, INNER_MIN_REL[2]]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
        _round_bounds(
            np.array([INNER_MIN_REL[0], INNER_MIN_REL[1], 0.5]),
            np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], INNER_MAX_REL[2]]),
        ),
    }
    assert _rounded_box_set(result) == expected


# --- update_ems: 更新後に他EMSへ完全内包されるEMSは残らない ---------------------------

def test_update_ems_removes_ems_contained_after_split():
    """分割で生じた部分EMSが既存の別EMSへ完全内包される場合、その部分EMSは残らない。

    box_a=[0,0,0]-[1.5,1.2,1.6], box_b=[0,0.8,0]-[1.5,2.0,1.6] を入力とし、
    障害物 [0,0.3,0]-[1.5,0.8,1.6] は box_a とのみ交差する（box_b とは y=0.8 で
    面接触のみ、非交差）。box_a は y 軸方向に [0,0.3]・[0.8,1.2] の2個へ分割され、
    後者 [0,0.8,0]-[1.5,1.2,1.6] は box_b の部分集合として完全内包される。
    """
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    box_a = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 1.2, 1.6], dtype=np.float64),
    )
    box_b = EMSBox(
        min_rel=np.array([0.0, 0.8, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )
    obstacle_min = np.array([0.0, 0.3, 0.0], dtype=np.float64)
    obstacle_max = np.array([1.5, 0.8, 1.6], dtype=np.float64)

    result = ems.update_ems([box_a, box_b], (obstacle_min, obstacle_max), space)

    expected = {
        _round_bounds(np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.3, 1.6])),  # box_a の残存部分
        _round_bounds(np.array([0.0, 0.8, 0.0]), np.array([1.5, 2.0, 1.6])),  # box_b（無交差のため保持）
    }
    assert _rounded_box_set(result) == expected
    _assert_no_redundant_containment(result)


# --- remove_contained: 公開APIの直接テスト --------------------------------------------

def test_remove_contained_removes_fully_contained_ems():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    outer = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )
    inner = EMSBox(
        min_rel=np.array([0.2, 0.2, 0.2], dtype=np.float64),
        max_rel=np.array([0.8, 0.8, 0.8], dtype=np.float64),
    )

    result = ems.remove_contained([outer, inner])

    assert _rounded_box_set(result) == {_round_bounds(outer.min_rel, outer.max_rel)}


def test_remove_contained_keeps_single_copy_of_exact_duplicate():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    box1 = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )
    box2 = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )

    result = ems.remove_contained([box1, box2])

    assert len(result) == 1
    assert _rounded_box_set(result) == {_round_bounds(box1.min_rel, box1.max_rel)}


def test_remove_contained_keeps_ems_that_are_not_contained():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    box_a = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.8, 2.0, 1.6], dtype=np.float64),
    )
    box_b = EMSBox(
        min_rel=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )

    result = ems.remove_contained([box_a, box_b])

    assert _rounded_box_set(result) == {
        _round_bounds(box_a.min_rel, box_a.max_rel),
        _round_bounds(box_b.min_rel, box_b.max_rel),
    }


def test_remove_contained_does_not_mutate_input_list():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    outer = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.5, 2.0, 1.6], dtype=np.float64),
    )
    inner = EMSBox(
        min_rel=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        max_rel=np.array([0.2, 0.2, 0.2], dtype=np.float64),
    )
    ems_list = [outer, inner]
    before = _rounded_box_set(ems_list)

    result = ems.remove_contained(ems_list)

    assert len(ems_list) == 2
    assert _rounded_box_set(ems_list) == before
    assert _rounded_box_set(result) == {_round_bounds(outer.min_rel, outer.max_rel)}


# --- 性質テスト: generate_ems（全再構築）と update_ems（逐次適用）の一致 ----------------

def test_update_ems_sequential_matches_generate_ems_full_rebuild():
    """ランダム10配置×20シードで、generate_ems（全再構築）と初期EMSへ update_ems を
    逐次適用した結果が一致することを確認する（詳細仕様書 §4.3 T-010節の性質テスト）。

    あわせて最終EMS集合の不変条件（正体積・内壁内包・障害物と非交差・相互内包ゼロ）も
    検証する。
    """
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    for seed in range(20):
        rng = np.random.default_rng(seed)
        obstacles = _random_non_overlapping_obstacles(rng, INNER_MIN_REL, INNER_MAX_REL, n=10)

        space_rebuild = _empty_space()
        rebuilt = ems.generate_ems(space_rebuild, placed_aabbs=obstacles)

        space_incremental = _empty_space()
        incremental = [EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy())]
        for obstacle_min, obstacle_max in obstacles:
            incremental = ems.update_ems(incremental, (obstacle_min, obstacle_max), space_incremental)

        assert _rounded_box_set(rebuilt) == _rounded_box_set(incremental), f"seed={seed}"

        _assert_no_redundant_containment(incremental)
        for box in incremental:
            assert np.all(box.max_rel - box.min_rel > constants.EPS_GEOM), f"seed={seed}"
            assert geometry.aabb_contains(
                INNER_MIN_REL, INNER_MAX_REL, box.min_rel, box.max_rel, margin=0.0
            ), f"seed={seed}"
            for obstacle_min, obstacle_max in obstacles:
                assert not geometry.aabb_intersects(
                    box.min_rel, box.max_rel, obstacle_min, obstacle_max, tol=0.0
                ), f"seed={seed}"
