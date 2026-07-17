"""T-009/T-010/T-011: packing_core.ems の全再構築・差分更新・上位選択/正規化/打切り率検証
（詳細仕様書 §2・§3・§4.3・§6 T-009〜T-011。依存: T-005 geometry AABB系、T-007 cut/棚対応）。

ems.py は T-011時点で `prune_min_dim`／`select_topn`／`normalize_descriptors` が未実装のため、
収集(--collect-only)を成功させるべく各テスト関数の内部で対象モジュールを
import する（モジュール直下 import は禁止、test_geometry.py・test_container_space.py
と同方針）。T-011の3関数を呼び出すテストは、実装が入るまで `AttributeError` で失敗する
（意図した「未実装起因の失敗」であり、テスト自体の不備ではない）。

本ファイルは T-009（generate_ems の全再構築、既知3EMSケース）・T-010（update_ems の
6面切断・remove_contained 公開API・全再構築との一致性質テスト）・T-011（prune_min_dim の
3軸判定、select_topn の並び順（z1昇順→体積降順→座標タイブレーク）・打切り率・空入力／
n_per_container<=0、normalize_descriptors の軸別正規化・空入力・ゼロ幅軸）を対象とする。
"""
import numpy as np
import pytest

from src.packing_core import constants, geometry
from src.packing_core.container_space import ContainerSpace, build_container_space

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


# =====================================================================================
# T-011: prune_min_dim / select_topn / normalize_descriptors
#
# 決定事項（人間承認済み、2026-07-17）:
#   * select_topn の同順位タイブレーク: (min_rel[2], -volume(), min_rel[0], min_rel[1],
#     max_rel[0], max_rel[1], max_rel[2]) の辞書式昇順。
#   * 打切り率 = discarded_count / valid_ems_count。valid_ems_count==0（空入力）のときは
#     0.0（切り捨て対象が存在しないため）。
#   * n_per_container < 0 は ValueError。n_per_container == 0 は空リスト入力に対応する
#     打切り率（非空入力→1.0、空入力→0.0）。
#   * normalize_descriptors はコンテナ内壁のいずれかの軸幅（inner_max_rel-inner_min_rel）
#     が EPS_GEOM 以下なら ValueError（0除算を np.clip 等で隠さない）。
# =====================================================================================

def _container_space_with_inner_bounds(
    inner_min_rel: np.ndarray, inner_max_rel: np.ndarray
) -> ContainerSpace:
    """normalize_descriptors 専用: inner_min_rel/inner_max_rel だけを指定した最小 ContainerSpace。

    normalize_descriptors は inner_min_rel/inner_max_rel のみを参照する契約のため、
    他フィールド（cut_planes/shelf_boxes/floor_z 等）はダミー値で埋める
    （本ファイル専用、他所非依存）。
    """
    dummy_grid = np.zeros((1, 1), dtype=np.float64)
    return ContainerSpace(
        index=0,
        offset_x=0.0,
        inner_min_rel=np.asarray(inner_min_rel, dtype=np.float64),
        inner_max_rel=np.asarray(inner_max_rel, dtype=np.float64),
        cut_planes=[],
        shelf_boxes=[],
        cell=CELL,
        floor_z=dummy_grid,
        ceil_z=dummy_grid,
        height=dummy_grid,
    )


# --- prune_min_dim: 各軸が min_dim 以上のEMSだけが残る -------------------------------

def test_prune_min_dim_keeps_only_boxes_meeting_min_dim_on_all_axes():
    """各軸の寸法が min_dim 以上（境界=min_dimと等しい場合を含む）のEMSだけが残る。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    min_dim = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    keep_exact = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.3, 0.5, 0.4], dtype=np.float64),
    )  # x寸法がちょうど min_dim（境界は残す側）
    keep_large = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.6, 0.6, 0.6], dtype=np.float64),
    )
    fail_x = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.2, 0.5, 0.4], dtype=np.float64),
    )
    fail_y = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.5, 0.2, 0.4], dtype=np.float64),
    )
    fail_z = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.5, 0.5, 0.2], dtype=np.float64),
    )

    result = ems.prune_min_dim([keep_exact, keep_large, fail_x, fail_y, fail_z], min_dim)

    assert _rounded_box_set(result) == {
        _round_bounds(keep_exact.min_rel, keep_exact.max_rel),
        _round_bounds(keep_large.min_rel, keep_large.max_rel),
    }


def test_prune_min_dim_excludes_box_failing_on_any_single_axis():
    """1軸でも min_dim を下回れば、他の軸が十分でもEMSは除外される。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    min_dim = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    fail_x = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.29, 1.0, 1.0])
    )
    fail_y = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([1.0, 0.29, 1.0])
    )
    fail_z = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([1.0, 1.0, 0.29])
    )

    result = ems.prune_min_dim([fail_x, fail_y, fail_z], min_dim)

    assert result == []


def test_prune_min_dim_is_order_independent():
    """出力（残る集合）は入力の並び順に依存しない。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    min_dim = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    keep = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.6, 0.6, 0.6])
    )
    fail = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.1, 0.6, 0.6])
    )
    boxes = [keep, fail]

    forward = ems.prune_min_dim(boxes, min_dim)
    backward = ems.prune_min_dim(list(reversed(boxes)), min_dim)

    expected = {_round_bounds(keep.min_rel, keep.max_rel)}
    assert _rounded_box_set(forward) == expected
    assert _rounded_box_set(backward) == expected


def test_prune_min_dim_does_not_mutate_input_list():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    min_dim = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    keep = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.6, 0.6, 0.6])
    )
    fail = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.1, 0.6, 0.6])
    )
    boxes = [keep, fail]
    before = _rounded_box_set(boxes)

    ems.prune_min_dim(boxes, min_dim)

    assert len(boxes) == 2
    assert _rounded_box_set(boxes) == before


# --- select_topn: 並び順（z1昇順→体積降順→座標タイブレーク） ------------------------

def test_select_topn_orders_by_z1_ascending():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    low_z = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.1], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 0.5], dtype=np.float64),
    )
    mid_z = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.3], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 0.6], dtype=np.float64),
    )
    high_z = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.5], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 0.9], dtype=np.float64),
    )

    result, rate = ems.select_topn([high_z, low_z, mid_z], n_per_container=3)

    assert [round(float(box.min_rel[2]), 6) for box in result] == [0.1, 0.3, 0.5]
    assert rate == pytest.approx(0.0)


def test_select_topn_orders_by_volume_descending_when_z1_ties():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    small = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.5, 0.5, 0.5])
    )  # volume 0.125
    large = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([1.0, 1.0, 1.0])
    )  # volume 1.0
    medium = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64), max_rel=np.array([0.8, 0.8, 0.8])
    )  # volume 0.512

    result, rate = ems.select_topn([small, large, medium], n_per_container=3)

    assert [round(box.volume(), 6) for box in result] == [1.0, 0.512, 0.125]
    assert rate == pytest.approx(0.0)


def test_select_topn_coordinate_tiebreak_uses_min_x_when_z1_and_volume_tie():
    """z1・体積が同じ場合、次点キー min_rel[0] の昇順で決定的に並べる。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    box_a = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    box_b = EMSBox(
        min_rel=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([2.0, 1.0, 1.0], dtype=np.float64),
    )
    box_c = EMSBox(
        min_rel=np.array([2.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([3.0, 1.0, 1.0], dtype=np.float64),
    )

    result, rate = ems.select_topn([box_c, box_a, box_b], n_per_container=3)

    assert [float(box.min_rel[0]) for box in result] == [0.0, 1.0, 2.0]
    assert rate == pytest.approx(0.0)


def test_select_topn_coordinate_tiebreak_uses_min_y_when_min_x_also_ties():
    """z1・体積・min_x が同じ場合、次点キー min_rel[1] の昇順で決定的に並べる。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    box_a = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    box_b = EMSBox(
        min_rel=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.0, 2.0, 1.0], dtype=np.float64),
    )
    box_c = EMSBox(
        min_rel=np.array([0.0, 2.0, 0.0], dtype=np.float64),
        max_rel=np.array([1.0, 3.0, 1.0], dtype=np.float64),
    )

    result, rate = ems.select_topn([box_c, box_a, box_b], n_per_container=3)

    assert [float(box.min_rel[1]) for box in result] == [0.0, 1.0, 2.0]
    assert rate == pytest.approx(0.0)


# --- select_topn: 打切り（N未満のときだけ切り捨て、打切り率=discarded/valid） ------------

def test_select_topn_truncates_only_when_more_than_n_available():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    boxes = [
        EMSBox(
            min_rel=np.array([0.0, 0.0, float(i)], dtype=np.float64),
            max_rel=np.array([1.0, 1.0, float(i) + 0.5], dtype=np.float64),
        )
        for i in range(5)
    ]  # z1 = 0,1,2,3,4 と相異なるため体積タイブレークは発生しない

    result, rate = ems.select_topn(boxes, n_per_container=2)

    assert len(result) == 2
    assert [float(box.min_rel[2]) for box in result] == [0.0, 1.0]
    assert rate == pytest.approx((5 - 2) / 5)


def test_select_topn_zero_truncation_rate_when_n_covers_all():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    boxes = [
        EMSBox(
            min_rel=np.array([0.0, 0.0, float(i)], dtype=np.float64),
            max_rel=np.array([1.0, 1.0, float(i) + 0.5], dtype=np.float64),
        )
        for i in range(3)
    ]

    result, rate = ems.select_topn(boxes, n_per_container=10)  # N が全件数以上

    assert len(result) == 3
    assert rate == pytest.approx(0.0)


# --- select_topn: 空入力・n_per_container<=0 -----------------------------------------

def test_select_topn_empty_input_returns_empty_with_zero_rate():
    from src.packing_core import ems

    result, rate = ems.select_topn([], n_per_container=5)

    assert result == []
    assert rate == pytest.approx(0.0)


def test_select_topn_zero_n_with_nonempty_input_returns_empty_and_full_truncation_rate():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    boxes = [
        EMSBox(
            min_rel=np.zeros(3, dtype=np.float64),
            max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        )
    ]

    result, rate = ems.select_topn(boxes, n_per_container=0)

    assert result == []
    assert rate == pytest.approx(1.0)


def test_select_topn_zero_n_with_empty_input_returns_empty_and_zero_rate():
    from src.packing_core import ems

    result, rate = ems.select_topn([], n_per_container=0)

    assert result == []
    assert rate == pytest.approx(0.0)


def test_select_topn_negative_n_raises_value_error():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    boxes = [
        EMSBox(
            min_rel=np.zeros(3, dtype=np.float64),
            max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        )
    ]

    with pytest.raises(ValueError):
        ems.select_topn(boxes, n_per_container=-1)


def test_select_topn_does_not_mutate_input_list():
    """select_topn は並び替えた新規リストを返し、入力リスト自体の順序は変えない。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    box_a = EMSBox(
        min_rel=np.zeros(3, dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    box_b = EMSBox(
        min_rel=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 2.0], dtype=np.float64),
    )
    boxes = [box_b, box_a]  # 入力順は z1 降順（sort後の期待順とは逆）
    before_ids = [id(b) for b in boxes]

    ems.select_topn(boxes, n_per_container=1)

    assert [id(b) for b in boxes] == before_ids
    assert boxes[0] is box_b and boxes[1] is box_a


# --- normalize_descriptors: shape・軸別正規化・空入力・dtype・非破壊 -------------------

def test_normalize_descriptors_shape_and_dtype():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    boxes = [
        EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy()),
        EMSBox(
            min_rel=np.array([0.5, 0.5, 0.5], dtype=np.float64),
            max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        ),
    ]

    result = ems.normalize_descriptors(boxes, space)

    assert result.shape == (2, 6)
    assert result.dtype == np.float64


def test_normalize_descriptors_inner_min_maps_to_zero():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    box = EMSBox(
        min_rel=INNER_MIN_REL.copy(),
        max_rel=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    result = ems.normalize_descriptors([box], space)

    np.testing.assert_allclose(result[0, 0:3], [0.0, 0.0, 0.0], atol=1e-9)


def test_normalize_descriptors_inner_max_maps_to_one():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    box = EMSBox(
        min_rel=np.array([1.0, 1.5, 1.0], dtype=np.float64),
        max_rel=INNER_MAX_REL.copy(),
    )

    result = ems.normalize_descriptors([box], space)

    np.testing.assert_allclose(result[0, 3:6], [1.0, 1.0, 1.0], atol=1e-9)


def test_normalize_descriptors_axes_are_normalized_independently():
    """内壁が非立方体(1.5x2.0x1.6)であることを利用し、軸ごとに異なる係数（内壁スパン）で
    正規化されることを確認する（コンテナ外寸や単一係数の流用を検出する）。
    """
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    box = EMSBox(
        min_rel=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )

    result = ems.normalize_descriptors([box], space)

    expected_min = [0.5 / 1.5, 0.5 / 2.0, 0.5 / 1.6]
    expected_max = [1.0 / 1.5, 1.0 / 2.0, 1.0 / 1.6]
    np.testing.assert_allclose(result[0, 0:3], expected_min, atol=1e-9)
    np.testing.assert_allclose(result[0, 3:6], expected_max, atol=1e-9)


def test_normalize_descriptors_empty_list_shape():
    from src.packing_core import ems

    space = _empty_space()

    result = ems.normalize_descriptors([], space)

    assert result.shape == (0, 6)
    assert result.dtype == np.float64


def test_normalize_descriptors_does_not_mutate_inputs():
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    space = _empty_space()
    box = EMSBox(
        min_rel=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        max_rel=np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    boxes = [box]
    min_before = box.min_rel.copy()
    max_before = box.max_rel.copy()
    inner_min_before = space.inner_min_rel.copy()
    inner_max_before = space.inner_max_rel.copy()

    ems.normalize_descriptors(boxes, space)

    np.testing.assert_array_equal(box.min_rel, min_before)
    np.testing.assert_array_equal(box.max_rel, max_before)
    np.testing.assert_array_equal(space.inner_min_rel, inner_min_before)
    np.testing.assert_array_equal(space.inner_max_rel, inner_max_before)
    assert len(boxes) == 1 and boxes[0] is box


def test_normalize_descriptors_raises_on_zero_width_container_axis():
    """内壁のいずれかの軸幅が0（縮退）の場合、0除算を隠さず ValueError を送出する。"""
    from src.packing_core import ems
    from src.packing_core.types import EMSBox

    degenerate_space = _container_space_with_inner_bounds(
        inner_min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        inner_max_rel=np.array([0.0, 2.0, 1.6], dtype=np.float64),  # x幅=0
    )
    box = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.0, 1.0, 1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError):
        ems.normalize_descriptors([box], degenerate_space)
