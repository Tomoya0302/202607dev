"""単一情報源。TODO(P0) は Phase 0 の interface_notes.md から転記して確定する。
転記完了まで assert_confirmed() が失敗する設計とし、未確定のまま提出させない。
"""
from dataclasses import dataclass

EPS_GEOM = 1e-9        # 同一判定
TOL_CONTACT = 5e-3     # 接触判定 5mm

# L字経路プロキシ用の意味付き定数（A15確定、T-016B。出典: validator.py::check_transport_path
# L118-133、docs/実装詳細仕様書.md §4.2「T-016B: L字経路用派生フィールド」）。
RESTING_SNAP_BAND = 0.05       # 直置き面直上とみなす帯幅 [m]（validator.py:120 の 0.05）
CEILING_CLIP_SAFETY = 0.0005   # 天井頭打ち回避クリップの安全余裕 [m]（validator.py:132 の 0.0005）


@dataclass(frozen=True)
class PlacementParams:
    """配置判定に用いるマージン・高さ関連パラメータ。

    Attributes:
        inclusion_margin: 内壁包含判定の許容量 [m]（符号解釈=A14）。
        safety_margin: L字経路等の安全マージン [m]。
        start_z: 配置開始高さ [m]。
        ceiling_margin: 天井とのクリアランス [m]。
        internal_extra: 内部判定の厳格化幅 [m]。
        start_margin: L字経路の入口レーン・頭打ちクリップに使う余裕 [m]（公式
            `BaseValidator.__init__` の `config.get('start_margin', 0.01)` 既定を転記。
            A15確定、T-016B）。
    """

    inclusion_margin: float = -0.005   # 符号解釈=A14
    safety_margin: float = 0.015
    start_z: float = 0.08
    ceiling_margin: float = 0.018
    internal_extra: float = 0.005      # 内部判定の厳格化幅
    start_margin: float = 0.01         # A15確定、T-016B


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
        ems_top_n_per_container: build_state が各コンテナごとに select_topn へ渡すEMS予算
            （T-012確定。コンテナ単位の値であり、全コンテナ合計ではない）。
    """

    l_path_top_m: int = 64
    ems_top_n_per_container: int = 80


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
        regime_threshold_n: この既配置数未満なら conservative（未確定。下記コメント参照）。
        gamma_cons: conservative レジームの指数。
        gamma_aggr: aggressive レジームの指数。
    """

    # 未確定: README の「一定数以上積めないと fill 以外 0」に対応する具体的な閾値だが、
    # 配布 evaluator.py にはこのロジック・数値が実装されていない（読解済み、interface_notes.md §I-2）。
    # 推測で埋めず -1 のまま据え置き、運営照会中（interface_notes.md §J）。回答後に確定値へ更新する。
    regime_threshold_n: int = -1
    gamma_cons: float = 4.0
    gamma_aggr: float = 1.0


KIND_IS_SOFT: dict = {
    # 出典: configs/item_params.xlsx（is_soft 行）。interface_notes.md §H 参照。
    # 注意: ランタイムの item 辞書に kind フィールドは無く is_soft を直接持つ。
    #       本表は学習データ合成・参照用（キー名は xlsx 列の英スラッグ化、暫定）。
    "suitcase_large":  False,   # スーツケース(大)
    "suitcase_medium": False,   # スーツケース(中)
    "suitcase_small":  False,   # スーツケース(小)
    "duffel_boston":   True,    # ダッフル/ボストン
    "cardboard":       True,    # 段ボール
    "backpack_large":  True,    # 大型リュックサック
    "daypack_small":   True,    # 小型デイパック
}

POOL_MAX_WEIGHT: float = 18.0
# 出典: item_params.xlsx 最大 mass（スーツケース(大)=18kg）。interface_notes.md §H 参照。
# 評価基盤の実荷物は xlsx と別分布（README）のため、正規化に使う際は下流で [0,1] にクランプすること。

OBS_KEYS: dict = {
    # 出典: env.py / containers.py / items.py。interface_notes.md §E 参照。
    "init_states": ["optimize", "lookahead_k", "container_list"],
    "observation": ["optimize", "lookahead_k", "depth_map", "container_list", "pool_list"],
    "observation_raw_shm": ["shm_name", "shm_shape", "shm_dtype"],  # runner 復元前の生observation
    "container": [
        "index", "length", "width", "height", "cut_x", "cut_y", "thickness",
        "center", "n_vecs", "points", "volume", "shelf", "is_prioritized", "packed_items",
    ],
    "item": [
        "index", "length", "width", "height", "mass", "is_prioritized", "is_soft",
        "belongs_to", "pos", "orn", "lateralFriction", "rollingFriction",
        "spinningFriction", "restitution", "angularDamping",
    ],
    "item_soft_extra": ["contactStiffness", "contactDamping", "linearDamping"],  # is_soft=True のみ付与
    "action": ["item_idx", "container_idx", "place_pos", "orientation"],
}

ASSUMPTIONS = {
    # id: {"claim": str, "status": "unconfirmed|confirmed|rejected", "ref": "Q番号/根拠"}
    "A9":  {"claim": "quat は (x,y,z,w)", "status": "confirmed",
            "ref": "T-002 interface_notes.md §E; README:280, items.py, validator.py:181"},
    "A10": {"claim": "pos は幾何中心", "status": "unconfirmed", "ref": "T-004 fixture"},
    "A11": {"claim": "item_idx はプール内 index（消費で縮小）", "status": "confirmed",
            "ref": "T-002 interface_notes.md §F; env.py:209, items.py:218, README:297,304"},
    "A12": {"claim": ("公式の入口面・入口レーン計算式：入口面は世界座標で "
                       "rel_start.y = -container.width/2。入口レーン "
                       "lane_x = clamp(rel_target.x, x_min, x_max)（x_min=-length/2+thickness"
                       "+cut_x+half_lwh[0]+start_margin、x_max=length/2-thickness-half_lwh[0]"
                       "-start_margin）"),
            "status": "confirmed",
            "ref": "T-016 調査; validator.py::check_transport_path L85-175"},
    "A13": {"claim": "orientation 表は §3.2", "status": "unconfirmed", "ref": "T-004"},
    "A14": {"claim": ("inclusion_margin は符号付きマージン: 正で緩和（はみ出し許容）／"
                       "負で厳格（内側クリアランス要求）。-0.005 は内側5mm必須の意"),
            "status": "confirmed",
            "ref": "T-002 interface_notes.md §C; validator.py:78, evaluator.py:56, utils.py:207"},
    "A15": {"claim": ("A12 の式（lane_x／入口面 y／start_z／resting・ceiling surfaces）を純NumPy "
                       "ContainerSpace/PackingState へ写像する式は path_entry_y_rel／"
                       "path_lane_x_min_geom_rel／path_lane_x_max_geom_rel／"
                       "path_mid_resting_z_rel／path_mid_ceiling_z_rel／path_obstacle_boxes_rel "
                       "の6フィールド（§4.2「T-016B: L字経路用派生フィールド」）"),
            "status": "confirmed",
            "ref": "T-016B仕様追補 v1.13; T-016Aゴールデン1,000件で危険な誤合格0件・採択率403/403実測"},
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
