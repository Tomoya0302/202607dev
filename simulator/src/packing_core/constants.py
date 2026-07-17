"""単一情報源。TODO(P0) は Phase 0 の interface_notes.md から転記して確定する。
転記完了まで assert_confirmed() が失敗する設計とし、未確定のまま提出させない。
"""
from dataclasses import dataclass

EPS_GEOM = 1e-9        # 同一判定
TOL_CONTACT = 5e-3     # 接触判定 5mm


@dataclass(frozen=True)
class PlacementParams:
    """配置判定に用いるマージン・高さ関連パラメータ。

    Attributes:
        inclusion_margin: 内壁包含判定の許容量 [m]（符号解釈=A14）。
        safety_margin: L字経路等の安全マージン [m]。
        start_z: 配置開始高さ [m]。
        ceiling_margin: 天井とのクリアランス [m]。
        internal_extra: 内部判定の厳格化幅 [m]。
    """

    inclusion_margin: float = -0.005   # 符号解釈=A14
    safety_margin: float = 0.015
    start_z: float = 0.08
    ceiling_margin: float = 0.018
    internal_extra: float = 0.005      # 内部判定の厳格化幅


@dataclass(frozen=True)
class TimeParams:
    """ステップ内の時間予算パラメータ。

    Attributes:
        policy_soft: policy() のソフト締切 [s]。
        policy_hard: policy() のハード締切 [s]。
        optimize_stop: optimize() の打ち切り時刻 [s]。
        budget_poll_every: 候補ループ中の締切ポーリング周期 [件]。
    """

    policy_soft: float = 6.5
    policy_hard: float = 7.0
    optimize_stop: float = 170.0
    budget_poll_every: int = 64


@dataclass(frozen=True)
class GridParams:
    """絞り込み専用格子のパラメータ。

    Attributes:
        cell: 格子セル一辺の長さ [m]（絞り込み専用。最終判定は連続幾何）。
    """

    cell: float = 0.02                 # 絞り込み専用


@dataclass(frozen=True)
class StageParams:
    """段階フィルタのパラメータ。

    Attributes:
        l_path_top_m: L字判定に回す上位候補数。
    """

    l_path_top_m: int = 64


@dataclass(frozen=True)
class ScoreParams:
    """heuristic_score の線形結合重み。

    Attributes:
        w_z: 低さを優先する重み。
        w_y: 奥（+Y）方向を優先する重み。
        w_x: X若い側を優先する重み。
        w_support: 支持率を優先する重み。
        w_cg_h: 重量物を低く置く重み。
        w_soft: ソフト上ハードのペナルティ重み。
        w_prio: 優先荷物の優先配置ボーナス重み。
    """

    w_z: float = 1.0
    w_y: float = 0.3
    w_x: float = 0.1
    w_support: float = 0.5
    w_cg_h: float = 0.3
    w_soft: float = 0.8
    w_prio: float = 0.4


@dataclass(frozen=True)
class RegimeParams:
    """レジーム判定と期待値計算のパラメータ。

    Attributes:
        regime_threshold_n: この既配置数未満なら conservative（TODO(P0)）。
        gamma_cons: conservative レジームの指数。
        gamma_aggr: aggressive レジームの指数。
    """

    regime_threshold_n: int = -1       # TODO(P0): evaluator.py の「一定数」を転記
    gamma_cons: float = 4.0
    gamma_aggr: float = 1.0


KIND_IS_SOFT: dict = {}                # TODO(P0): item_params.xlsx から転記
POOL_MAX_WEIGHT: float = -1.0          # TODO(P0)
OBS_KEYS: dict = {}                    # TODO(P0): observation/init のキー名転記表

ASSUMPTIONS = {
    # id: {"claim": str, "status": "unconfirmed|confirmed|rejected", "ref": "Q番号/根拠"}
    "A9":  {"claim": "quat は (x,y,z,w)", "status": "unconfirmed", "ref": "interface_notes"},
    "A10": {"claim": "pos は幾何中心", "status": "unconfirmed", "ref": "T-004 fixture"},
    "A11": {"claim": "item_idx はプール内 index", "status": "unconfirmed", "ref": "env.py"},
    "A12": {"claim": "入口レーン x は公式定義に従う", "status": "unconfirmed", "ref": "T-016"},
    "A13": {"claim": "orientation 表は §3.2", "status": "unconfirmed", "ref": "T-004"},
    "A14": {"claim": "inclusion_margin は緩和方向", "status": "unconfirmed", "ref": "読解"},
}


def assert_confirmed(*ids: str) -> None:
    """指定した仮定IDがすべて confirmed であることを検証する。

    Args:
        *ids: `ASSUMPTIONS` のキー（例: "A9", "A10"）。

    Raises:
        RuntimeError: いずれかの仮定が confirmed でない場合。
    """
    bad = [i for i in ids if ASSUMPTIONS.get(i, {}).get("status") != "confirmed"]
    if bad:
        raise RuntimeError(f"未確認の仮定に依存: {bad}. 台帳を更新してから使用すること")
