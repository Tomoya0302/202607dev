"""空き空間候補（EMS: Empty Maximal Space）。詳細仕様書 §4.3（T-009: 全再構築）。

本チケット（T-009）では `generate_ems`（有効空間AABBからの全再構築）と、その内部でのみ
使用する分割・重複/内包除去のprivate helperのみを実装する。`update_ems`（T-010の差分更新、
公開API）・`prune_min_dim`／`select_topn`／`normalize_descriptors`（T-011の上位選択・
正規化・打切り率）はスコープ外であり、本ファイルには含まない。
"""
import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM
from src.packing_core.container_space import ContainerSpace
from src.packing_core.types import EMSBox, Vec3


def _split_ems_by_obstacle(ems: EMSBox, obstacle_min: Vec3, obstacle_max: Vec3) -> list[EMSBox]:
    """1つの障害物でEMSを最大6個の部分EMSへ分割する（詳細仕様書 §4.3 の6面切断）。

    障害物と交差しない場合は元のEMSをそのまま1個返す。交差する場合、各軸について
    障害物の手前側（`max[axis]` を `obstacle_min[axis]` に置き換え）と奥側
    （`min[axis]` を `obstacle_max[axis]` に置き換え）の2個の部分箱を作る。各部分箱は
    分割軸以外の範囲を元のEMSのまま維持する。正体積（全軸の寸法が `EPS_GEOM` を超える）
    の部分箱だけを返す。

    Args:
        ems: 分割対象のEMS。
        obstacle_min: 障害物AABBの最小点（コンテナ相対）。shape (3,), float64。
        obstacle_max: 障害物AABBの最大点（コンテナ相対）。shape (3,), float64。

    Returns:
        分割後の正体積 `EMSBox` のリスト（交差なしなら元のEMSのみを含む長さ1のリスト）。
    """
    if not geometry.aabb_intersects(ems.min_rel, ems.max_rel, obstacle_min, obstacle_max, tol=0.0):
        return [ems]

    subs: list[EMSBox] = []
    for axis in range(3):
        near_max = ems.max_rel.copy()
        near_max[axis] = obstacle_min[axis]
        subs.append(EMSBox(min_rel=ems.min_rel.copy(), max_rel=near_max))

        far_min = ems.min_rel.copy()
        far_min[axis] = obstacle_max[axis]
        subs.append(EMSBox(min_rel=far_min, max_rel=ems.max_rel.copy()))

    return [box for box in subs if np.all(box.max_rel - box.min_rel > EPS_GEOM)]


def _remove_duplicate_or_contained(ems_list: list[EMSBox]) -> list[EMSBox]:
    """重複・他のEMSに完全内包されるEMSを除去する（詳細仕様書 §4.3 T-009節）。

    `geometry.aabb_contains`（境界一致を含む判定、margin=0.0）で他のEMSに完全に収まる
    EMSを除去する。境界が完全に一致する組（相互内包）は `ems_list` 内で最も早く現れた
    ものだけを残す（同順位の重複を両方除去してしまわないための tie-break）。

    Args:
        ems_list: 重複・内包チェック対象のEMSリスト。

    Returns:
        重複・内包を除いたEMSのリスト（元の順序を維持）。
    """
    result: list[EMSBox] = []
    for i, box in enumerate(ems_list):
        redundant = False
        for j, other in enumerate(ems_list):
            if i == j:
                continue
            if not geometry.aabb_contains(
                other.min_rel, other.max_rel, box.min_rel, box.max_rel, margin=0.0
            ):
                continue
            same_bounds = bool(
                np.allclose(box.min_rel, other.min_rel, rtol=0.0, atol=EPS_GEOM)
                and np.allclose(box.max_rel, other.max_rel, rtol=0.0, atol=EPS_GEOM)
            )
            if same_bounds:
                if j < i:
                    redundant = True
                    break
                continue
            redundant = True
            break
        if not redundant:
            result.append(box)
    return result


def generate_ems(
    space: ContainerSpace,
    placed_aabbs: list[tuple[Vec3, Vec3]],
) -> list[EMSBox]:
    """有効空間AABBから EMS (Empty Maximal Space) を全再構築する（詳細仕様書 §4.3 T-009節）。

    `space.inner_min_rel`/`inner_max_rel` を初期EMSとし、`placed_aabbs` と
    `space.shelf_boxes` を区別せず障害物として順に適用する。障害物ごとに現在のEMS集合の
    各要素を `_split_ems_by_obstacle` で分割し、`_remove_duplicate_or_contained` で
    重複・完全内包されたEMSを除去する（T-010の差分更新用 `update_ems` 公開APIと
    T-011の `prune_min_dim`／`select_topn`／`normalize_descriptors` はスコープ外）。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        placed_aabbs: 既配置荷物のAABB（コンテナ相対）のリスト。各要素は `(min, max)`。

    Returns:
        正体積かつ相互に非内包な `EMSBox` のリスト。`(min_rel, max_rel)` による
        決定的な順序で返す。
    """
    ems_list: list[EMSBox] = [
        EMSBox(min_rel=space.inner_min_rel.copy(), max_rel=space.inner_max_rel.copy())
    ]

    obstacles = list(placed_aabbs) + list(space.shelf_boxes)
    for obstacle_min, obstacle_max in obstacles:
        obstacle_min = np.asarray(obstacle_min, dtype=np.float64)
        obstacle_max = np.asarray(obstacle_max, dtype=np.float64)

        next_ems_list: list[EMSBox] = []
        for ems in ems_list:
            next_ems_list.extend(_split_ems_by_obstacle(ems, obstacle_min, obstacle_max))
        ems_list = _remove_duplicate_or_contained(next_ems_list)

    ems_list.sort(key=lambda box: tuple(box.min_rel) + tuple(box.max_rel))
    return ems_list
