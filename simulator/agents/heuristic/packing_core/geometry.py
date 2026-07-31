"""幾何プリミティブ（回転・AABB・交差・スイープ用）。状態を持たない純関数群。

詳細仕様書 §4.1（T-004: 回転系／T-005: AABB系）。
"""
import numpy as np

from .types import Vec3

# oriented_size: orientation(0..5) -> (L,W,H) 3軸の並べ替えindex。詳細仕様書 §3.2 の表に対応。
_ORIENTATION_PERM: dict[int, tuple[int, int, int]] = {
    0: (0, 1, 2),
    1: (0, 2, 1),
    2: (2, 1, 0),
    3: (1, 0, 2),
    4: (1, 2, 0),
    5: (2, 0, 1),
}

# rotated_aabb: 単位立方体の8頂点（各軸 ±0.5）。size を乗じてローカル座標の頂点にする。
_UNIT_CORNERS: np.ndarray = np.array(
    [
        [sx, sy, sz]
        for sx in (-0.5, 0.5)
        for sy in (-0.5, 0.5)
        for sz in (-0.5, 0.5)
    ],
    dtype=np.float64,
)


def oriented_size(size: Vec3, orientation: int) -> Vec3:
    """orientation コードに応じて回転後の外形寸法を返す（詳細仕様書 §3.2）。

    Args:
        size: 回転前 (L, W, H) [m]。shape (3,), float64。
        orientation: 0..5 の orientation コード。

    Returns:
        回転後の外形寸法 (X, Y, Z) [m]。shape (3,), float64。

    Raises:
        ValueError: size にゼロ以下の要素がある場合、または orientation が 0..5 の範囲外の場合。
    """
    size = np.asarray(size, dtype=np.float64)
    if np.any(size <= 0):
        raise ValueError(f"size の全要素は正である必要があります: {size}")
    if orientation not in _ORIENTATION_PERM:
        raise ValueError(f"orientation は 0..5 の範囲である必要があります: {orientation}")
    perm = _ORIENTATION_PERM[orientation]
    return size[list(perm)]


def aabb_from_center(center: Vec3, osize: Vec3) -> tuple[Vec3, Vec3]:
    """中心座標と寸法から AABB を返す。

    Args:
        center: 中心座標 [m]。shape (3,), float64。
        osize: 回転後寸法 [m]。shape (3,), float64。

    Returns:
        `(amin, amax)`。それぞれ shape (3,), float64。

    Raises:
        ValueError: osize にゼロ以下の要素がある場合。
    """
    center = np.asarray(center, dtype=np.float64)
    osize = np.asarray(osize, dtype=np.float64)
    if np.any(osize <= 0):
        raise ValueError(f"osize の全要素は正である必要があります: {osize}")
    half = osize / 2.0
    return center - half, center + half


def aabb_intersects(amin: Vec3, amax: Vec3, bmin: Vec3, bmax: Vec3, tol: float = 0.0) -> bool:
    """2つの AABB が交差するかを判定する。

    全軸で `amin < bmax - tol` かつ `bmin < amax - tol` のとき交差とみなす（厳密比較）。
    `tol=0` のとき面がぴったり接するだけの場合は交差とみなさない。

    Args:
        amin: 箱Aの最小点。shape (3,), float64。
        amax: 箱Aの最大点。shape (3,), float64。
        bmin: 箱Bの最小点。shape (3,), float64。
        bmax: 箱Bの最大点。shape (3,), float64。
        tol: 判定の許容量 [m]。正で判定を緩め、負で判定を厳しくする。

    Returns:
        交差していれば True。
    """
    amin = np.asarray(amin, dtype=np.float64)
    amax = np.asarray(amax, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    cond_a = amin < (bmax - tol)
    cond_b = bmin < (amax - tol)
    return bool(np.all(cond_a) and np.all(cond_b))


def aabb_contains(
    outer_min: Vec3, outer_max: Vec3, inner_min: Vec3, inner_max: Vec3, margin: float = 0.0
) -> bool:
    """inner が outer を margin 分だけ縮めた箱に収まるかを判定する。

    Args:
        outer_min: 外箱の最小点。shape (3,), float64。
        outer_max: 外箱の最大点。shape (3,), float64。
        inner_min: 内箱の最小点。shape (3,), float64。
        inner_max: 内箱の最大点。shape (3,), float64。
        margin: 外箱を縮める量 [m]。正で厳格化、負で拡大（緩和）。

    Returns:
        収まっていれば True（境界一致は収まっている扱い）。
    """
    outer_min = np.asarray(outer_min, dtype=np.float64)
    outer_max = np.asarray(outer_max, dtype=np.float64)
    inner_min = np.asarray(inner_min, dtype=np.float64)
    inner_max = np.asarray(inner_max, dtype=np.float64)
    shrunk_min = outer_min + margin
    shrunk_max = outer_max - margin
    return bool(np.all(inner_min >= shrunk_min) and np.all(inner_max <= shrunk_max))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """クォータニオンを3x3回転行列に変換する。

    Args:
        q: クォータニオン。PyBullet順 `(x, y, z, w)`（仮定A9）。shape (4,)。
           正規化されていない場合は事前に正規化する。

    Returns:
        3x3 回転行列。shape (3, 3), float64。
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )


def rotated_aabb(center: Vec3, size: Vec3, quat: np.ndarray) -> tuple[Vec3, Vec3]:
    """回転後の8頂点から AABB を計算する。

    Args:
        center: 中心座標 [m]。shape (3,), float64。
        size: 回転前寸法 (L, W, H) [m]。shape (3,), float64。
        quat: クォータニオン。PyBullet順 `(x, y, z, w)`。shape (4,)。

    Returns:
        `(amin, amax)`。それぞれ shape (3,), float64。

    Raises:
        ValueError: size にゼロ以下の要素がある場合。
    """
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    if np.any(size <= 0):
        raise ValueError(f"size の全要素は正である必要があります: {size}")
    corners_local = _UNIT_CORNERS * size  # shape (8, 3)
    rot = quat_to_matrix(quat)
    corners_world = corners_local @ rot.T + center  # shape (8, 3)
    return corners_world.min(axis=0), corners_world.max(axis=0)


def inflate(bmin: Vec3, bmax: Vec3, delta: float) -> tuple[Vec3, Vec3]:
    """AABB を各軸方向へ delta だけ拡張する。

    Args:
        bmin: 最小点。shape (3,), float64。
        bmax: 最大点。shape (3,), float64。
        delta: 拡張量 [m]。負で縮小。

    Returns:
        `(bmin - delta, bmax + delta)`。それぞれ shape (3,), float64。
    """
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    return bmin - delta, bmax + delta
