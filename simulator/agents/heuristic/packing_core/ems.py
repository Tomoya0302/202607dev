"""空き空間候補（EMS: Empty Maximal Space）。詳細仕様書 §4.3
（T-009: 全再構築、T-010: 差分更新、T-011: 上位選択・正規化・打切り率）。

本ファイルは `generate_ems`（全再構築）・`update_ems`（差分更新）・`remove_contained`
（重複/完全内包除去）・`prune_min_dim`（min_dim未満のEMS除去）・`select_topn`
（上位N選択・打切り率算出）・`normalize_descriptors`（[0,1]正規化記述子）の6公開APIと、
その内部でのみ使用する6面切断のprivate helperを実装する。
"""
import numpy as np

from . import geometry
from .constants import EPS_GEOM
from .container_space import ContainerSpace
from .types import EMSBox, Vec3


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


def remove_contained(ems_list: list[EMSBox]) -> list[EMSBox]:
    """重複・他のEMSに完全内包されるEMSを除去する（詳細仕様書 §4.3、公開API）。

    `geometry.aabb_contains`（境界一致を含む判定、margin=0.0）で他のEMSに完全に収まる
    EMSを除去する。境界が完全に一致する組（相互内包）は `ems_list` 内で最も早く現れた
    ものだけを残す（同順位の重複を両方除去してしまわないための tie-break）。非内包の
    EMSはすべて保持する。入力リスト・入力EMS自体はいずれも変更しない（`EMSBox` は
    `frozen=True` のため要素側は元々不変。新規リストのみを構築して返す）。

    Args:
        ems_list: 重複・内包チェック対象のEMSリスト。

    Returns:
        重複・内包を除いたEMSのリスト。出力順は入力順を基準に決定的（除去されなかった
        要素を入力に現れた順のまま並べる）。
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


def update_ems(
    ems_list: list[EMSBox],
    obstacle: tuple[Vec3, Vec3],
    space: ContainerSpace,
) -> list[EMSBox]:
    """1個の障害物を適用してEMS集合を差分更新する（詳細仕様書 §4.3 T-010節）。

    `ems_list` の各要素を `_split_ems_by_obstacle` に通す（障害物と交差しない要素は
    そのまま1個保持され、交差する要素は6面切断のうち正体積の部分箱だけに置き換わる）。
    続けて `space.inner_min_rel`/`inner_max_rel` の内側に完全に収まらない部分箱を除去し
    （`geometry.aabb_contains`、margin=0.0、境界一致は内側扱い）、最後に `remove_contained`
    で重複・完全内包されたEMSを除去する。`prune_min_dim` 以降（T-011）は行わない。

    `ems_list` 自体は読み取りのみで変更しない（新規リストを構築して返す）。

    Args:
        ems_list: 更新対象のEMSリスト。
        obstacle: 障害物AABB（コンテナ相対）の `(min, max)`。
        space: 対象コンテナの `ContainerSpace`（内壁包含チェックに使用）。

    Returns:
        差分更新後の正体積・内壁内・相互非内包な `EMSBox` のリスト。
    """
    obstacle_min = np.asarray(obstacle[0], dtype=np.float64)
    obstacle_max = np.asarray(obstacle[1], dtype=np.float64)

    next_ems_list: list[EMSBox] = []
    for ems in ems_list:
        next_ems_list.extend(_split_ems_by_obstacle(ems, obstacle_min, obstacle_max))

    inner_min = space.inner_min_rel
    inner_max = space.inner_max_rel
    next_ems_list = [
        box
        for box in next_ems_list
        if geometry.aabb_contains(inner_min, inner_max, box.min_rel, box.max_rel, margin=0.0)
    ]

    return remove_contained(next_ems_list)


def generate_ems(
    space: ContainerSpace,
    placed_aabbs: list[tuple[Vec3, Vec3]],
) -> list[EMSBox]:
    """有効空間AABBから EMS (Empty Maximal Space) を全再構築する（詳細仕様書 §4.3 T-009節）。

    `space.inner_min_rel`/`inner_max_rel` を初期EMSとし、`placed_aabbs` と
    `space.shelf_boxes` を区別せず障害物として順に `update_ems` を適用する
    （T-011の `prune_min_dim`／`select_topn`／`normalize_descriptors` はスコープ外）。

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
        ems_list = update_ems(ems_list, (obstacle_min, obstacle_max), space)

    ems_list.sort(key=lambda box: tuple(box.min_rel) + tuple(box.max_rel))
    return ems_list


def prune_min_dim(ems_list: list[EMSBox], min_dim: Vec3) -> list[EMSBox]:
    """各軸寸法が `min_dim` 未満のEMSを除去する（詳細仕様書 §4.3 T-011節）。

    3軸すべての寸法（`max_rel - min_rel`）が対応する `min_dim` 以上の場合だけ残す
    （1軸でも不足すれば除外、境界値と等しい場合は残す）。独自の許容誤差は追加しない。
    入力リスト・EMS内部の配列はいずれも変更せず、新規リストのみを構築して返す。
    出力順は入力順を維持する。

    Args:
        ems_list: 判定対象のEMSリスト。
        min_dim: 各軸の最小寸法。shape (3,), float64。

    Returns:
        3軸すべてが `min_dim` 以上のEMSのみを入力順で含むリスト。
    """
    min_dim = np.asarray(min_dim, dtype=np.float64)
    return [box for box in ems_list if np.all(box.max_rel - box.min_rel >= min_dim)]


def select_topn(
    ems_list: list[EMSBox], n_per_container: int
) -> tuple[list[EMSBox], float]:
    """決定的ソートキーでEMSを並べ替え、上位 `n_per_container` 件を選ぶ（詳細仕様書 §4.3 T-011節）。

    ソートキーは `(min_z, -volume, min_x, min_y, max_x, max_y, max_z)` の辞書式昇順
    （`min_z` 昇順→体積降順→座標タイブレーク）。入力リストは並べ替えず、ソート済みの
    新規リストを返す。

    打切り率は `discarded_count / valid_ems_count`（`valid_ems_count` は `ems_list` の件数）。
    空入力は分母が0のため `0.0` とする。

    Args:
        ems_list: 選択対象のEMSリスト（`prune_min_dim` 済みを想定）。
        n_per_container: コンテナ当たりの上位選択数。

    Returns:
        `(選択されたEMSのソート済みリスト, 打切り率)`。

    Raises:
        ValueError: `n_per_container` が負の場合。
    """
    if n_per_container < 0:
        raise ValueError(f"n_per_container は0以上である必要があります: {n_per_container}")

    valid_count = len(ems_list)
    if valid_count == 0:
        return [], 0.0
    if n_per_container == 0:
        return [], 1.0

    sorted_ems = sorted(
        ems_list,
        key=lambda box: (
            float(box.min_rel[2]),
            -float(box.volume()),
            float(box.min_rel[0]),
            float(box.min_rel[1]),
            float(box.max_rel[0]),
            float(box.max_rel[1]),
            float(box.max_rel[2]),
        ),
    )
    selected = sorted_ems[:n_per_container]
    discarded_count = valid_count - len(selected)
    return selected, discarded_count / valid_count


def normalize_descriptors(ems_list: list[EMSBox], space: ContainerSpace) -> np.ndarray:
    """EMSを内壁基準で `[0, 1]` に正規化した記述子配列へ変換する（詳細仕様書 §4.3 T-011節）。

    各EMSを `(min_x, min_y, min_z, max_x, max_y, max_z)` の順で6次元化し、各軸を
    `(v - inner_min_rel[axis]) / (inner_max_rel[axis] - inner_min_rel[axis])` で独立に
    正規化する（`np.clip` による丸めはしない）。入力EMS・`space` はいずれも変更しない。

    Args:
        ems_list: 正規化対象のEMSリスト。
        space: 正規化基準となる内壁AABBを持つ `ContainerSpace`。

    Returns:
        shape `(N, 6)`、dtype `float64` の正規化記述子配列。空入力は shape `(0, 6)`。

    Raises:
        ValueError: `space` の内壁幅（`inner_max_rel - inner_min_rel`）のいずれかの軸が
            `EPS_GEOM` 以下の場合。
    """
    inner_min = space.inner_min_rel
    inner_max = space.inner_max_rel
    width = inner_max - inner_min
    if np.any(width <= EPS_GEOM):
        raise ValueError(f"内壁幅がEPS_GEOM以下です: width={width}")

    if not ems_list:
        return np.zeros((0, 6), dtype=np.float64)

    raw = np.array(
        [np.concatenate([box.min_rel, box.max_rel]) for box in ems_list], dtype=np.float64
    )
    denom = np.concatenate([width, width])
    origin = np.concatenate([inner_min, inner_min])
    return (raw - origin) / denom
