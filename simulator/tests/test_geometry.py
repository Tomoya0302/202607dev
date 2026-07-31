"""T-003: packing_core.geometry の幾何プリミティブ検証（詳細仕様書 §3.2・§4.1・§6 T-003）。

geometry.py は本チケット時点では未実装のため、収集(--collect-only)を成功させるべく
各テスト関数の内部で対象モジュールを import する（モジュール直下 import は禁止）。
"""
import numpy as np
import pytest

from agents.heuristic.packing_core import constants

# 仕様 §4.1: 回転を含む比較の許容誤差は 1e-6 を使用する。
ROT_TOL = 1e-6


# --- oriented_size -------------------------------------------------------------

@pytest.mark.parametrize(
    "orientation, expected",
    [
        (0, (0.5, 0.4, 0.3)),  # そのまま (L,W,H)
        (1, (0.5, 0.3, 0.4)),  # X軸90° (L,H,W)
        (2, (0.3, 0.4, 0.5)),  # Y軸90° (H,W,L)
        (3, (0.4, 0.5, 0.3)),  # Z軸90° (W,L,H)
        (4, (0.4, 0.3, 0.5)),  # Y90°→Z90° (W,H,L)
        (5, (0.3, 0.5, 0.4)),  # X90°→Z90° (H,L,W)
    ],
)
def test_oriented_size_matches_spec_table(orientation, expected):
    from agents.heuristic.packing_core import geometry

    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    result = geometry.oriented_size(size, orientation)
    np.testing.assert_allclose(result, np.array(expected, dtype=np.float64), atol=constants.EPS_GEOM)


def test_oriented_size_invalid_orientation_raises():
    from agents.heuristic.packing_core import geometry

    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.oriented_size(size, 6)
    with pytest.raises(ValueError):
        geometry.oriented_size(size, -1)


def test_oriented_size_zero_dimension_raises():
    from agents.heuristic.packing_core import geometry

    size = np.array([0.0, 0.4, 0.3], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.oriented_size(size, 0)


def test_oriented_size_negative_dimension_raises():
    from agents.heuristic.packing_core import geometry

    size = np.array([-0.1, 0.4, 0.3], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.oriented_size(size, 0)


# --- aabb_from_center ------------------------------------------------------------

def test_aabb_from_center_basic():
    from agents.heuristic.packing_core import geometry

    center = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    osize = np.array([0.4, 0.6, 0.2], dtype=np.float64)
    amin, amax = geometry.aabb_from_center(center, osize)
    np.testing.assert_allclose(amin, np.array([0.8, 1.7, 0.4]), atol=constants.EPS_GEOM)
    np.testing.assert_allclose(amax, np.array([1.2, 2.3, 0.6]), atol=constants.EPS_GEOM)


def test_aabb_from_center_zero_dimension_raises():
    from agents.heuristic.packing_core import geometry

    center = np.zeros(3, dtype=np.float64)
    osize = np.array([0.0, 0.6, 0.2], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.aabb_from_center(center, osize)


def test_aabb_from_center_negative_dimension_raises():
    from agents.heuristic.packing_core import geometry

    center = np.zeros(3, dtype=np.float64)
    osize = np.array([-0.1, 0.6, 0.2], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.aabb_from_center(center, osize)


# --- aabb_intersects --------------------------------------------------------------

def test_aabb_intersects_clear_overlap_true():
    from agents.heuristic.packing_core import geometry

    amin, amax = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    bmin, bmax = np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])
    assert geometry.aabb_intersects(amin, amax, bmin, bmax)


def test_aabb_intersects_clear_separation_false():
    from agents.heuristic.packing_core import geometry

    amin, amax = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    bmin, bmax = np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0])
    assert not geometry.aabb_intersects(amin, amax, bmin, bmax)


def test_aabb_intersects_face_touching_false():
    # 仕様 §4.1 のテスト例: [0,1]^3 と [1,2]x[0,1]^2 は面接触のみで交差ではない（tol=0）。
    from agents.heuristic.packing_core import geometry

    amin, amax = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    bmin, bmax = np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])
    assert not geometry.aabb_intersects(amin, amax, bmin, bmax)


def test_aabb_intersects_negative_tol_treats_small_gap_as_overlap():
    # §4.5 check_overlap の用法: tol=-internal_extra のとき、internal_extra(v38=1mm) 未満の
    # 隙間は交差(重なり)とみなされる（厳格側）。ここでは 0.5mm の隙間 < internal_extra=1mm。
    from agents.heuristic.packing_core import geometry

    amin, amax = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    bmin, bmax = np.array([1.0005, 0.0, 0.0]), np.array([2.0005, 1.0, 1.0])
    assert not geometry.aabb_intersects(amin, amax, bmin, bmax, tol=0.0)
    internal_extra = constants.PlacementParams().internal_extra
    assert geometry.aabb_intersects(amin, amax, bmin, bmax, tol=-internal_extra)


def test_aabb_intersects_positive_tol_ignores_small_overlap():
    # 判定式 (amin < bmax - tol) から導かれる挙動: 正の tol は tol 未満のわずかな重なりを
    # 非交差とみなす（tol=0 では重なりとして検出される 3mm の重なりが tol=5mm で消える）。
    from agents.heuristic.packing_core import geometry

    amin, amax = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    bmin, bmax = np.array([0.997, 0.0, 0.0]), np.array([1.997, 1.0, 1.0])
    assert geometry.aabb_intersects(amin, amax, bmin, bmax, tol=0.0)
    assert not geometry.aabb_intersects(amin, amax, bmin, bmax, tol=0.005)


# --- aabb_contains --------------------------------------------------------------

def test_aabb_contains_normal_case_true():
    from agents.heuristic.packing_core import geometry

    outer_min, outer_max = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    inner_min, inner_max = np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])
    assert geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max)


def test_aabb_contains_outside_false():
    from agents.heuristic.packing_core import geometry

    outer_min, outer_max = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    inner_min, inner_max = np.array([0.5, 0.5, 0.5]), np.array([2.5, 1.5, 1.5])
    assert not geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max)


def test_aabb_contains_positive_margin_stricter():
    # margin>0 は outer を margin 分だけ縮める（厳格化）。
    from agents.heuristic.packing_core import geometry

    outer_min, outer_max = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    inner_min, inner_max = np.array([0.05, 0.05, 0.05]), np.array([1.95, 1.95, 1.95])
    assert geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max, margin=0.0)
    assert not geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max, margin=0.1)


def test_aabb_contains_negative_margin_relaxed():
    # margin<0 は outer を |margin| 分だけ拡大する（緩和）。
    from agents.heuristic.packing_core import geometry

    outer_min, outer_max = np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0])
    inner_min, inner_max = np.array([-0.1, -0.1, -0.1]), np.array([2.1, 2.1, 2.1])
    assert not geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max, margin=0.0)
    assert geometry.aabb_contains(outer_min, outer_max, inner_min, inner_max, margin=-0.2)


# --- quat_to_matrix ---------------------------------------------------------------

def test_quat_to_matrix_identity():
    from agents.heuristic.packing_core import geometry

    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    result = geometry.quat_to_matrix(q)
    np.testing.assert_allclose(result, np.eye(3), atol=ROT_TOL)


def test_quat_to_matrix_z90():
    # 公式 utils.ORNS[3]=[0,0,pi/2] を getQuaternionFromEuler した値
    # (0,0,0.7071068,0.7071068) に対応（interface_notes.md 読解・手計算で突合済み）。
    from agents.heuristic.packing_core import geometry

    q = np.array([0.0, 0.0, 0.7071068, 0.7071068], dtype=np.float64)
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = geometry.quat_to_matrix(q)
    np.testing.assert_allclose(result, expected, atol=ROT_TOL)


def test_quat_to_matrix_non_unit_normalizes():
    from agents.heuristic.packing_core import geometry

    q_unit = np.array([0.0, 0.0, 0.7071068, 0.7071068], dtype=np.float64)
    q_scaled = q_unit * 2.0
    result_unit = geometry.quat_to_matrix(q_unit)
    result_scaled = geometry.quat_to_matrix(q_scaled)
    np.testing.assert_allclose(result_scaled, result_unit, atol=ROT_TOL)


# --- rotated_aabb ------------------------------------------------------------------

def test_rotated_aabb_identity_matches_aabb_from_center():
    from agents.heuristic.packing_core import geometry

    center = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    q_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    expected_min, expected_max = geometry.aabb_from_center(center, size)
    result_min, result_max = geometry.rotated_aabb(center, size, q_identity)
    np.testing.assert_allclose(result_min, expected_min, atol=ROT_TOL)
    np.testing.assert_allclose(result_max, expected_max, atol=ROT_TOL)


def test_rotated_aabb_z90_dims():
    from agents.heuristic.packing_core import geometry

    center = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    q_z90 = np.array([0.0, 0.0, 0.7071068, 0.7071068], dtype=np.float64)

    result_min, result_max = geometry.rotated_aabb(center, size, q_z90)
    dims = result_max - result_min
    np.testing.assert_allclose(dims, np.array([0.4, 0.5, 0.3]), atol=ROT_TOL)
    # 回転は中心周りに行われるため、AABB の中心は元の center を保つ。
    np.testing.assert_allclose((result_min + result_max) / 2.0, center, atol=ROT_TOL)


def test_rotated_aabb_zero_dimension_raises():
    from agents.heuristic.packing_core import geometry

    center = np.zeros(3, dtype=np.float64)
    size = np.array([0.0, 0.4, 0.3], dtype=np.float64)
    q_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.rotated_aabb(center, size, q_identity)


def test_rotated_aabb_negative_dimension_raises():
    from agents.heuristic.packing_core import geometry

    center = np.zeros(3, dtype=np.float64)
    size = np.array([-0.1, 0.4, 0.3], dtype=np.float64)
    q_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        geometry.rotated_aabb(center, size, q_identity)


# --- inflate -----------------------------------------------------------------------

def test_inflate_positive_delta_expands():
    from agents.heuristic.packing_core import geometry

    bmin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    bmax = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    result_min, result_max = geometry.inflate(bmin, bmax, 0.1)
    np.testing.assert_allclose(result_min, np.array([-0.1, -0.1, -0.1]), atol=constants.EPS_GEOM)
    np.testing.assert_allclose(result_max, np.array([1.1, 1.1, 1.1]), atol=constants.EPS_GEOM)
