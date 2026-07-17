"""積付コアで共有するデータ構造（§3.3 の写像）。"""
from dataclasses import dataclass, field

import numpy as np

Vec3 = np.ndarray  # shape (3,), float64


@dataclass(frozen=True)
class ItemSpec:
    """プール内の荷物仕様。

    Attributes:
        idx: プール内 index（仮定 A11: policy の item_idx はこの値）。
        size: 回転前 (L, W, H) [m]。shape (3,), float64。
        weight: 重量 [kg]。
        kind: item_params.xlsx 由来の種別キー。公式 observation に kind 相当キーは
            存在しないため（interface_notes.md §H）、`state.build_state` が observation
            から構築する場合は常に None。提出時のランタイム方策は kind に依存せず、
            種別相当の特徴量は is_soft/is_priority を使う（実装詳細仕様書 §3.3/§4.4）。
        is_soft: 種別→bool 変換は `constants.KIND_IS_SOFT` を参照。
        is_priority: 優先荷物かどうか。
    """

    idx: int
    size: Vec3
    weight: float
    kind: str | None
    is_soft: bool
    is_priority: bool


@dataclass(frozen=True)
class PlacedItem:
    """既に配置済みの荷物。

    Attributes:
        pos_world: 幾何中心・世界座標（仮定 A10）。shape (3,), float64。
        orn_quat: 姿勢クォータニオン。shape (4,), PyBullet順 (x,y,z,w)（仮定 A9）。
        size: 回転前寸法 (L, W, H) [m]。shape (3,), float64。
        weight: 重量 [kg]。
        is_soft: ソフト荷物かどうか。
        is_priority: 優先荷物かどうか。
        aabb_min_rel: 回転後AABBの最小点（コンテナ相対）。state.py が計算して充填。
        aabb_max_rel: 回転後AABBの最大点（コンテナ相対）。state.py が計算して充填。
    """

    pos_world: Vec3
    orn_quat: np.ndarray
    size: Vec3
    weight: float
    is_soft: bool
    is_priority: bool
    aabb_min_rel: Vec3
    aabb_max_rel: Vec3


@dataclass(frozen=True)
class EMSBox:
    """EMS（Empty Maximal Space）を表すAABB（コンテナ相対）。

    Attributes:
        min_rel: 最小点（コンテナ相対）。shape (3,), float64。
        max_rel: 最大点（コンテナ相対）。shape (3,), float64。
    """

    min_rel: Vec3
    max_rel: Vec3

    def size(self) -> Vec3:
        """各軸の寸法を返す。

        Returns:
            `max_rel - min_rel`。shape (3,), float64。
        """
        return self.max_rel - self.min_rel

    def volume(self) -> float:
        """体積を返す。

        Returns:
            寸法3成分の積 [m^3]。
        """
        return float(np.prod(self.size()))


@dataclass
class Candidate:
    """配置候補。段階フィルタとスコアリングの結果を保持する可変データ。

    Attributes:
        item_idx: プール内 index（仮定 A11）。
        container_idx: 対象コンテナ index。
        ems_id: 対象 EMS の index。
        orientation: §3.2 の orientation コード（0..5）。
        pos_rel: 配置中心（コンテナ相対）。shape (3,), float64。
        osize: 回転後寸法。shape (3,), float64。
        feasible: 全段階フィルタ通過で True。
        reject_reason: 不合格理由（"dims"|"inclusion"|"overlap"|"ceiling"|"path"|""）。
        features: §4.8 の特徴量辞書。
        score: heuristic_score の値。
        p_success: 1 - p_ng（配置成功確率の推定）。
    """

    item_idx: int
    container_idx: int
    ems_id: int
    orientation: int
    pos_rel: Vec3
    osize: Vec3
    feasible: bool = False
    reject_reason: str = ""
    features: dict = field(default_factory=dict)
    score: float = 0.0
    p_success: float = 1.0
