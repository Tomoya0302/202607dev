"""単一情報源。TODO(P0) は Phase 0 の interface_notes.md から転記して確定する。
転記完了まで assert_confirmed() が失敗する設計とし、未確定のまま提出させない。
"""
import os
from dataclasses import dataclass

EPS_GEOM = 1e-9        # 同一判定
TOL_CONTACT = 5e-3     # 接触判定 5mm


def _envf(name: str, default: float) -> float:
    """環境変数があれば float で上書き（Track1 重み探索・決定論評価用。既定は不変）。

    本番評価環境では該当環境変数を設定しないため、既定値がそのまま使われる。数値化に失敗した
    場合は既定値へフォールバックする（黙って壊れた探索値を採用しない）。
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    """環境変数があれば int で上書き（探索の breadth 調整用。既定は不変）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

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
        internal_extra: 内部判定の残余安全余裕 [m]。inclusion の主要クリアランスは
            公式 `inclusion_margin`（内側5mm）で決まり、本値はその上に積む追加余裕である
            （HF-002 で 0.005→0.001 に是正。公式 validator の2倍厳しかった過剰保守を解消）。
        start_margin: L字経路の入口レーン・頭打ちクリップに使う余裕 [m]（公式
            `BaseValidator.__init__` の `config.get('start_margin', 0.01)` 既定を転記。
            A15確定、T-016B）。
    """

    inclusion_margin: float = -0.005   # 符号解釈=A14
    safety_margin: float = 0.015
    start_z: float = 0.08
    ceiling_margin: float = 0.018
    internal_extra: float = 0.001      # 残余安全余裕（HF-002: 0.005→0.001、公式整合）
    start_margin: float = 0.01         # A15確定、T-016B
    candidate_generation_slack: float = 1e-6  # v1.16追加、T-024所有（float32往復の数値ガード、§3.5）


@dataclass(frozen=True)
class TimeParams:
    """ステップ内の時間予算パラメータ。

    Attributes:
        policy_soft: policy() のソフト締切 [s]。
        policy_hard: policy() のハード締切 [s]。
        optimize_stop: optimize() の打ち切り時刻 [s]。
        budget_poll_every: 候補ループ中の締切ポーリング周期 [件]。
    """

    # HF-005: プラットフォームの policy_timeout(=8s) に対し十分な余裕を残す。予算は
    # build_state 時間も含めて計測される（StepBudget.t0 は build_state より前に設定）ため、
    # ハード締切を 5.5s に下げると build_state+全段が概ね 5.5s 以内に収まり、8s 制限＋
    # プラットフォームの実行速度差＋IPCオーバヘッドに ~2.5s の安全余裕を持てる。タイムアウトは
    # ワーカー再起動→init喪失→固定emergency退行（HF-005で恒久対策済みだが未然に防ぐ）を招く。
    # GH_POLICY_SOFT/HARD で上書き可（Track1 の決定論評価では大きな値にして時間依存の
    # soft 打ち切り＝run-to-run ノイズを排除する。本番は既定 4.5/5.5 のまま）。
    policy_soft: float = _envf("GH_POLICY_SOFT", 4.5)   # HF-005: 6.5→4.5
    policy_hard: float = _envf("GH_POLICY_HARD", 5.5)   # HF-005: 7.0→5.5
    optimize_stop: float = 170.0
    budget_poll_every: int = 64


@dataclass(frozen=True)
class GridParams:
    """絞り込み専用格子のパラメータ。

    Attributes:
        cell: 格子セル一辺の長さ [m]（絞り込み専用。最終判定は連続幾何）。
    """

    # 絞り込み専用。HF-022 Phase 5 で env-var 化: これまでの方策介入7種はすべて
    # 「同じ格子上でどの候補を選ぶか」を変えるものだった（候補数の拡張・層規律・シャドウ減点・
    # 到達可能性先読み等、すべて No-Go）。**格子自体の解像度は一度も動かしていない**。
    # 2cm 格子で見つからない配置や、densify が +Y/−X の2方向しか詰め寄せない点が
    # 取りこぼしになっている可能性がある。既定 0.02 は不変。
    cell: float = _envf("GH_GRID_CELL", 0.02)


@dataclass(frozen=True)
class StageParams:
    """段階フィルタのパラメータ。

    Attributes:
        l_path_top_m: L字判定に回す上位候補数。
        ems_top_n_per_container: build_state が各コンテナごとに select_topn へ渡すEMS予算
            （T-012確定。コンテナ単位の値であり、全コンテナ合計ではない）。
    """

    # HF-010: 候補数上限を引き上げ（l_path_top_m 64→160, ems_top_n 80→220）。従来値では搬入可能な
    # 候補を打ち切っており、agent は「時間はあるが候補が枯渇」＝候補飢餓 state だった（実測: 多くの
    # step で policy 時間が予算(soft 4.5s)を大きく下回って終了）。上限を上げると 2vCPU で全11 local
    # config の積載数が非回帰で合計 +44%（3.05→4.41）、最大 policy 時間は ≤2.6s（8s制限・予算ゲート
    # 5.5s の内側）。grid3（幾何依存アンカー追加）と違い「既生成の同種候補を捨てず探索する」だけで
    # 過学習が無く、転倒制限の shelf は l_path_top_m の引上げが特に効く（64では baseline へ逆戻り）。
    l_path_top_m: int = 160
    ems_top_n_per_container: int = 220


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
    # HF-008(却下): 奥行き支配(w_y=3.0>w_z)の back-to-front 壁積みを試したが、ローカル2vCPUで
    # 積載 -8%・shelf は 0.024 へ激減（奥壁へ積み上げて転倒）。scoring だけでは壁積みにならず却下し
    # 0.6 へ戻した。w_y の再重み付けは無効(0.6)か悪化(3.0)で、このレバーは枯渇。
    w_y: float = 0.6
    w_x: float = 0.0   # HF-004: 0.1→0.0 低X(=−X)加点を除去（cut/棚で塞がる最悪コーナーへの偏りを解消）
    w_support: float = 0.5
    w_cg_h: float = 0.3
    w_soft: float = 0.8
    w_prio: float = 0.4


@dataclass(frozen=True)
class ProvisionalRiskParams:
    """provisional_p_ng の線形結合重み（§4.7、T-024所有・単一情報源）。

    ScoreParams とは独立したフィールド集合を持つ（重複キーなし、CONST-008）。

    Attributes:
        w_support_gap: 支持不足 (1-support_ratio) に対する重み。
        w_negative_cg: 負のcg_margin (max(0,-cg_margin)) に対する重み。
        cg_scale: cg_margin不足量のスケール係数。
        p_min: p_ng の下限クリップ値。
        p_max: p_ng の上限クリップ値（非有限入力時のフォールバック値でもある）。
    """

    w_support_gap: float = 0.6
    w_negative_cg: float = 0.4
    cg_scale: float = 10.0
    p_min: float = 0.0
    p_max: float = 0.95


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


# HF-011 / §4.14 order.py: 荷物順序最適化（optimize() フェーズ）のパラメータ（単一情報源）。
# 貪欲デコーダは「最初に置けない荷物」で終了するため積載数 ≒ 順序の関数。optimize() の
# 予算(自己締切 TimeParams.optimize_stop=170s)を使い、オフライン貪欲リプレイを評価器に
# 順序を探索する。SIM_* は探索中の高速化ノブ（本番 policy には無影響。order.py のみが参照）。
BEAM_W = 3                     # マルチスタート・アンカー数（上位BEAM_Wシードから山登り）。§4.14が参照
SIM_EMS_TOPN = 48              # 探索リプレイでコンテナあたり使うEMS上限（本番 220 を切詰め＝候補数↓）
SIM_LPATH_TOP_M = 32           # 探索リプレイの L_PATH 評価上位数（本番 160 を切詰め）
ORDER_SEED_CAP = 12            # 評価する初期シード順序の最大数（決定論・予算制御）
ORDER_MAX_ITERS = 4000         # 山登り総反復上限（実際は締切ゲートで先に停止することが多い）
ORDER_STAGNANT_RESTART = 32    # この回数だけ改善が無ければ次アンカーへ再スタート
ORDER_TIME_RESERVE_S = 5.0     # 最終フルfidelity再評価＋確定に残す余裕 [s]
ORDER_RNG_SEED = 12345         # 山登りの固定シード（再現性・決定論）


# HF-012 / Phase H1: heightmap 配置エンジンのパラメータ（src/packing_core/heightmap.py が参照）。
# 値は sample(50点) の実測チューニングを踏襲・調整。マージンは公式 validator の既定
# (inclusion=-0.005 / safety=0.015 / ceiling=0.018 / start_z=0.08) の内側に留める。
HM_WALL_CLEARANCE = 0.02       # 生成候補が壁と保つ余裕 [m]（inclusion 0マージン回避）
HM_CEIL_CLEARANCE = 0.02       # 候補上面が構造物と保つ頭上余裕 [m]
HM_SHELF_STANDOFF = 0.03       # 棚上面への標準オフセット [m]（搬入で棚を擦らない）
HM_RESTING_BAND = 0.05         # 直置き帯（この帯に底面が入ると公式は flush 搬入）[m]
HM_RESTING_BUMP = 0.051        # 棚上面近傍の着地をこの分だけ持ち上げる [m]（自由落下≤8cm利用）

# 接触セル判定の高さ許容 [m]（沈降傾きで AABB上面が実接触より上）。HF-022 で env-var 化:
# `_support_ok` は窓内 max を基準面にするため、1セルだけ突出していると支持率が過小評価される。
# 実測では沈降後の傾きが最大 1.3-3.5°（閾値45°）に収まっており、モデルは物理より保守的。
# 率(GH_HM_SUP_RATIO)を下げるより許容を広げる方が「実際に接している面」に近い。
HM_SUPPORT_TOL = _envf("GH_HM_SUP_TOL", 0.03)
HM_SUPPORT_RATIO_MIN = _envf("GH_HM_SUP_RATIO", 0.80)    # 支持セル率の下限（strict/relaxed）
HM_SUPPORT_CENTER_MIN = _envf("GH_HM_SUP_CENTER", 0.55)   # 中央1/4 の支持率下限（重心の縁越え防止）
HM_SUPPORT_RATIO_DESP = _envf("GH_HM_SUP_RATIO_DESP", 0.60)   # desperate の支持率下限
HM_SUPPORT_CENTER_DESP = _envf("GH_HM_SUP_CENTER_DESP", 0.50)  # desperate の中央支持率下限

# 候補スコア重み（fill 寄り：奥・低・平坦・壁寄せ＋大物ボーナス＋soft/prio違反減点）。
# Track1 の重み探索用に GH_HM_W_* で上書き可能（既定値は本番でそのまま使われる）。
# 既定値は HF-013 の公式env重み探索（座標降下, myopia-aware=stress_big含む）の最良: subset
# mean_fill 21.2→24.9。DEPTH 15→7.5 が支配項の過剰さ＝搬入レーン myopia を解消し全 config で
# fill 向上（stress_big 含む）。LEFT 0.5→0.77, FLAT 2.0→1.4, TALL 8.0→13.6。GH_HM_W_* で上書き可。
HM_W_DEPTH = _envf("GH_HM_W_DEPTH", 4.5)     # 奥（+Y）優先。全11探索で高原4.0-5.0(total292-302)の中央。6.0で崖(272)なのでロバスト中央値を採用。v13。
# 低い着地優先。【v27 で 3.0 を提出 → Public 53 で退行、6.0 へ戻した】
# 本番実測 v27 − v25: fill +0.59 / np +0.98pt（局所予測と符号一致・約0.43倍）だが
# **cog −5.27 / stability −3.97 / placement −2.55**。差引き mean5 −2.31。
# → **cog は幾何（高さ分布）に強く反応する**。v25 で「cog の駆動因は item selection で
#   幾何ではない」と結論したのは誤りだった（局所の重心メトリクスが当たらないことを
#   cog が幾何と無関係と誤って一般化していた）。
# 傾きは 1単位あたり cog −1.76 / fill +0.20。**上げれば cog が上がる**方向。
# 【v28 で 7.0 をベイクしたが提出前に撤回】局所 cog 予測子 `ctr_mean`（中心高の単純平均、
# 本番 cog の偏微分と 8/9 一致）で測ると、W_HEIGHT は **上げても下げても ctr_mean が上がる**:
#   3.0 で +0.0105 / 7.0 で +0.0144 → 予測 Δcog は両方向で負（−5.3 / −7.2）。
# つまり **6.0 は山の頂点**で、v27 の傾きからの外挿（上げれば cog が上がる）は誤りだった。
# 片側1点では傾きと山頂を区別できない（W_SOFT_CAP でも同じ誤りをした）。6.0 を維持する。
HM_W_HEIGHT = _envf("GH_HM_W_HEIGHT", 6.0)
# 左（-X, cut側可達）優先。HF-022: 2.5 を測ったが **採用しない**。W_HEIGHT=3.0 との組み合わせは
# 集計では最良（Δfill +2.34 / 18勝3敗）だが構成別に分けると dr10 +4.21 / holdout +1.71 /
# **overload −1.75**（1タスクが −5.00 の外れ値）で、SUP/FLOOR の過適合（dr10 +5.19 / overload −8.95
# → 本番退行）と同じ形。W_HEIGHT 単独は3構成すべて正なのでそちらを採る。
HM_W_LEFT = _envf("GH_HM_W_LEFT", 0.77)
HM_W_FLAT = _envf("GH_HM_W_FLAT", 1.4)       # 平坦面優先（land-min の差を減点）
# 縦長姿勢を抑制。HF-013 は 10.0 を採用（探索は13.6だが subset 過適合と判断、全11構成で10が最良）。
# HF-022 v21 で 13.0 を本番測定した結果: stability は **上がらなかった**（69.66→69.53）。
# 代わりに fill 41.69→42.31（全提出中の最高値）・np 0.588→0.611 だが、cog −0.52 / soft −0.35 /
# stability −0.13 が相殺して mean5 +0.07（Public 53）。品質4項は「積んだ荷物の平均」なので
# np を上げると薄まる＝fill/np と品質は限界でトレード。単因子測定を保つため 10.0 へ戻す。
HM_W_TALL = _envf("GH_HM_W_TALL", 10.0)
HM_W_BIG = _envf("GH_HM_W_BIG", 25.0)        # 大物を先に置く選択圧（体積比例、候補一律加点）
HM_W_SOFT_VIOL = _envf("GH_HM_W_SOFT_VIOL", 40.0)   # 非softをsoft上面に載せる減点（位置レベル。順序不変→fill保護）
HM_W_PRIO_VIOL = _envf("GH_HM_W_PRIO_VIOL", 30.0)   # 非prioをprio上面に載せる減点
# HF-012 Phase H2: 非fillサブスコア（cog/placement/soft）向けの項。README 定義に沿う。
HM_W_COG = _envf("GH_HM_W_COG", 0.3)  # 重量物を低く（cog_score）：-W_COG*mass*land（GH_HM_W_COG で上書き可）
# soft_item_score 用：`-W_SOFT_CAP * max(headroom,0)`。正値は soft を低頭上ポケットへ誘導する。
# HF-022 v22 で **−0.5 へベイク（符号反転）**。根拠は v15 の本番実測が与える偏微分:
#   v15 は W_SOFT_CAP=+4.0 で np 0.5878→0.5871・fill −0.02 と **np/fill をほぼ完全に保ったまま**
#   品質4項を全部下げた（cog −7.87 / stability −7.78 / placement −7.70 / soft −20.95）。
#   つまり本ノブは「np を動かさず品質だけを動かす」= fill/np↔品質のトレード尾根に**垂直**な方向で、
#   d(soft)/dW ≈ −5.2/単位。よって負値にすれば品質が上がる側へ動くはず。
# 大きさは v15 の 1/8 に抑えた（v15/16 の失敗＝全強度で一気に振ったこと）。23タスクの事前スクリーンで
# Δnp −0.4pt / Δfill −0.57（いずれも測定誤差 ±0.94pt / ±0.47 の内側）＝ np/fill 非退行を確認済み。
HM_W_SOFT_CAP = _envf("GH_HM_W_SOFT_CAP", 4.0)
# HF-022: placement_score 直結だが未測定だったため env-var 化（既定値は不変）。
HM_W_PRIO_CONTAINER_BONUS = _envf("GH_HM_W_PRIO_CONT_BONUS", 20.0)
HM_W_PRIO_CONTAINER_PENALTY = _envf("GH_HM_W_PRIO_CONT_PEN", 50.0)
HM_W_NONPRIO_IN_PRIO_CONTAINER_PENALTY = _envf("GH_HM_W_NONPRIO_IN_PRIO_PEN", 5.0)
# HF-022 v26: 優先品だけを高く置く加点（placement_score 狙い）。既定 0.0 = v25 恒等。
# 状態ダンプで判明: v24/v25 は `GH_PRIO_LATE=1` により置かれる優先品が平均 **1個** で、
# その位置が y_rel 0.064（扉側＝良）・**z_rel 0.036（ほぼ床＝最悪）**。最後に回しているのに
# 上へ載らず、空いている2台目コンテナの床へ落ちている。placement は「優先品の取り出しやすさ」
# なので、置く数を増やさず**その1個の高さだけ**を上げれば cog（=stability、駆動因は item selection）
# を一切犠牲にせずに placement を回復できる。影響が1荷物なので np/fill へのコストもほぼ無い。
# 【v26 で 30.0 を提出 → Public 52 で退行、既定 0.0 へ戻した】
# 本番実測 v26 − v25: fill −0.60 / cog −4.71 / stability −4.54 / placement −1.75 / soft −4.65。
# **狙った placement すら下がった**＝「置かれる少数の優先品を高くする」は placement を上げない。
# 退行の原因は測定側: この項のスクリーンは優先品率を 15% に膨らませ体積も偏っていた旧ベンチで
# 行っており（gen_bench_config.py のフラグ振り直しバグ、後に修正）、Δnp/Δfill の符号すら
# 逆に出ていた（局所 +1.52pt/+0.87 に対し本番 −0.82pt/−0.60）。修正ベンチでは優先品が
# 平均 0.4個しか置かれず、優先品対象のレバーは軒並み no-op。
HM_W_PRIO_HIGH = _envf("GH_HM_W_PRIO_HIGH", 0.0)

# HF-022 Phase 4（配置方策の置換）: 層規律。この床被覆率に達するまで積み上げを禁じる。
# 診断（scripts/diag_layering.py, 修正ベンチ dr10 × v24/v25/v27）で判明した実態:
#   **最初の積み上げが始まる時点の床被覆率が平均 32%**（20/20 コンテナで 80% 未満）、
#   配置の 79% が積み上げ、扉側(y_rel<0.35)に高さ45cm超の荷物が全体の 1/4。
# 公式 check_transport_path は底面が resting surface から 0.05m 以内なら持ち上げ無しの水平搬入、
# それ以外は 8cm 持ち上げるので、**床層は順序自由だが積み上げは扉へ降りる階段を要求する**。
# 床が1/3しか埋まらないうちに積み上げると凸凹＋扉側の壁ができ、後続の回廊が塞がる。
# これは重みでは直せない（線形スコアは層の規律を表現できない）ので制約として入れる。
# 0.0 = 無効（v25 恒等）。0.7 なら「床被覆 70% までは床候補だけを採る」。
HM_FLOOR_FIRST = _envf("GH_HM_FLOOR_FIRST", 0.0)

# HF-022 Phase 4: **z帯を考慮した**搬入路シャドウ減点。既定 0.0 = v25 恒等。
# 搬入路チェックを外すと同じ貪欲が 97.2% を配置し fill 63.30（上限 64.7 の 98%）に達する
# ＝ **壁は幾何ではなく到達可能性**（現状 61.8% / 45.34）。損失は配置率 35.4pt・fill 18pt。
# 先の HM_W_BLOCK は z を無視して奥の自由体積すべてを罰したため低い荷物にも過大な罰を与え、
# W_DEPTH と競合して単調に悪化した。本項は「候補が占める z 帯 + 0.08m（持ち上げ分）」に
# 限って奥の自由体積を数える。値は (影にする自由体積 / 残り自由体積) に掛かる。
HM_W_ZBLOCK = _envf("GH_HM_W_ZBLOCK", 0.0)

# HF-022 Phase 4: 到達可能性の1手先読み。既定 0 = 無効（v25 恒等）。
# 搬入路を外すと同じ貪欲が 97.2% を配置できる（現状 61.8%）＝壁は到達可能性で、損失は
# 配置率 35.4pt / 代理fill 18pt。シャドウ減点（W_BLOCK / W_ZBLOCK）はソフトな誘導にすぎず
# 効かなかった（z帯版でも cog は 0.7-1.3σ のノイズ、fill は ±0.6）。本質は貪欲の近視眼で、
# 「いま A を置くと将来 B,C,... が l_path で死ぬ」を**実際に数えていない**こと。
# そこで上位 K 個の実行可能候補について、仮置きして他候補（別荷物のもの）が何個 l_path を
# 通り続けるかを数え、生存数が最大の手を選ぶ。`check_l_path` は state.placed の AABB しか
# 読まないので仮挿入は軽い（AABB 2つのダミーを push/pop するだけ）。
HM_LOOKAHEAD_K = _envi("GH_HM_LOOKAHEAD_K", 0)     # 評価する実行可能候補の数（0=無効）
HM_LOOKAHEAD_PROBE = _envi("GH_HM_LOOKAHEAD_PROBE", 80)  # 生存を数える他候補の数

# HF-022 Phase 6: スラブ跨ぎのコスト。既定 0.0 = v25 恒等。
# 診断（scripts/diag_slab.py, 修正ベンチ dr10 × v24/v25）で分かった構造の破れ:
#   - スラブ内の奥→手前は **守られている**（整列 ρ −0.55、87% のスラブで明確）＝W_DEPTH が機能
#   - しかし **44% の手で高さ帯が変わる**（スラブ高 0.2m で測ると 59%）
#   - 同スラブ・同Xレーンで既存より奥に置く「逆行」が 28%
# 公式の搬入は単一高さでの水平スライドなので、回廊要件は水平スラブになる。帯を行き来すると
# どの帯の回廊も中途半端に埋まって完成しない。
# `W_HEIGHT` は**絶対高**を罰する（常に積み上げと戦う）が、本項は「**現在の作業面からの
# 飛び出し**」だけを罰する。全体が一緒に上がるのは許し、突出を抑える別の道具。
# 作業面は履歴なしで状態から定義する（実本番の policy は毎ステップ state を再構築するため）:
#   ref = 下層サーフェスの 30 パーセンタイル。penalty = W_SLAB * max(0, land - ref) / SLAB_H
HM_W_SLAB = _envf("GH_HM_W_SLAB", 0.0)

# HF-022 Phase 7: **奥→手前の一方向フロンティア制約**（構成法）。既定 -1 = 無効（v25 恒等）。
# 公式の搬入は単一高さでの水平スライドなので、あるXレーンで奥(j大)から手前(j小)へ一方向に
# 進めば回廊は常に空く（前方が未使用）。積み上げの +0.08m 持ち上げも同様。つまり
# **「一度手前に進んだらそのレーンでは二度と奥に戻らない」制約は搬入可能性を設計で保証する**。
# 搬入チェックに弾かれるのではなく、弾かれない順序しか生成しない。
# 診断（diag_slab.py）ではスラブ内の奥→手前は守られていた（ρ −0.55）が **逆行が 28%** あり、
# それが後続の回廊を狭めていた。scoring では表現できない（docs HF-008:「scoring だけでは
# 壁積みにならず却下」）ので制約として実装する。
# 値は許容する後戻り量（格子セル数）。0 なら厳密に単調、2 なら 2セル(=4cm)まで後戻りを許す。
HM_FRONTIER_SLACK = _envi("GH_HM_FRONTIER_SLACK", -1)

# HF-022 Phase 7: **体積×高さの交差項**。既定 0.0 = 無効（v25 恒等）。
# cog_score は「相対中心高の**単純平均**」（fit_cog_predictor.py で本番の偏微分と 8/9 一致）。
#   ctr_mean = mean(bottom_i + h_i/2)、Σh_i/2 は積んだ集合で決まる
#   ⇒ **cog の最小化は Σbottom_i の最小化と厳密に等価**。しかも**重みなし**。
# 帰結: 小物を高所へ置くコストは大物を高所へ置くコストと**同額**。小物の上段積みは
# cog を同じだけ払って fill をほとんど得ない**純損**である。従来の「大物を下・小物を上」は
# この指標に対して逆向きだった（v25 の HEAVY_SHIFT=0.3＝大物を後回し＝上へ、が効いたのと整合）。
# 実装は交差項ひとつ: land が高いほど**大きい荷物**を優先する。周辺項（W_HEIGHT=6.0 の山頂、
# W_BIG）を動かさないので、既に最適化済みの1次係数を壊さない。
HM_W_VOL_HIGH = _envf("GH_HM_W_VOLHIGH", 0.0)

# HF-022 Phase 9: **優先品のための容量予約**。既定 0.0 = 無効（v29 恒等）。
# placement_score の違反は3種類ある（analysis/platform_score.py::compute_placement）:
#   (a) 非優先品が優先品の真上に載る  (b) 優先容器があるのに非優先容器へ  (c) **未積載**
# `GH_PRIO_LATE` は優先品を末尾へ回して (a)(b) を解いたが、その代償に (c) を作っていた
# （優先品配置率 0.629 → 0.100）。予約はこの緊張を構造的に解く: 優先容器の**上部**を
# 優先品用に空けておけば、優先品が最後に到着しても必ず入り、蓋の位置（上に何も載らない）で、
# 正しい容器に置かれる ⇒ 3種類の違反が同時に消える。
# 値は予約高さの安全係数。残り優先品の体積を優先容器の内寸床面積で割った高さ × 本係数を、
# 天井から下へ予約し、その帯には**非優先品を置かせない**。優先品が積まれるたび予約は縮む。
# 1.0 なら「残り優先品がぴったり詰まる高さ」、1.5 なら 50% の余裕を持たせる。
# リスク: これは placement 軸であり、v31（PRIO_SHIFT=-1.6）が placement を +11.65 動かして
# Public を −7.30 にした軸そのもの。破れの機構は未解明（ground-handling-mean5-model-broken）。
# v32 で 1.0 をベイク。3ベンチで Δctr_mean −0.0169/−0.0197/−0.0044（符号一致）。
# 機構は当初の設計（placement）ではなく **cog**: 優先容器の上部を空けると非優先品が低い位置に
# 留まらざるを得ず、パック全体が下がる。placement の動きは +1.53/−0.55/+1.00 でほぼ中立なので、
# v31 が壊れた placement の大振り領域（+11.65）には入らない。
# v32 は 1.0（本番 np ≈ 0.609）。**v33 で 1.5 へ引き上げ**た。1.5 単独では局所 np 0.616 →
# 本番換算 0.587 で膝 0.588 を割るため v32 では採れなかったが、`GH_SOFT_SHIFT=-0.6`（soft の
# 前倒し）が np を +1.9pt 戻す（0.616 → 0.635、本番換算 0.616）ので併用で安全域に入る。
# **予約の np コストを soft 前倒しの np 利得で払う**構成。2.2 は soft 前倒しでも戻し切れない。
# v35 で **1.0 に戻した**（v32 の値）。1.5 は num_placed を削る: 局所 np は
# 1.0 で 0.642 / 1.5 で 0.635。**np は Public の R²=0.73 を説明する支配変数**だった
# （18提出で 本番np↔Public r=+0.857、局所np↔本番np r=+0.938）ので、np を削る設定は採らない。
# **v38: 0.0（無効）にベイク。これは診断目的の提出であり改善提出ではない。**
# 目的は**課題の混合比の測定**。予約は「優先コンテナの上部を空ける」機構なので
# **優先コンテナが存在しない 1容器40個 タスク族では構造的に無効**（`cont_prio` が成立しない）。
# 実測: 1容器族39タスクで全指標が**厳密に 0.0000**、2容器族99タスクで合成 ΔS −0.708（t=−5.24）。
# したがって本設定の ΔPublic は「2容器族のシェア × 転移比 × (−0.708)」をそのまま与える。
# 混合比は非公開（運営に確認済み）で、局所からの回帰では識別できなかった
# （α_A = 0.19 ± 0.61。希釈では符号が変わらないのに比が 0.00〜0.69 と −4.8〜−13.0 に分裂）。
# v29→v32 で予約を入れたとき局所 ΔS +0.705 → Public +0.20（比 0.28）だったので、
# 対称なら **Public 60.00 → 約 59.8** の見込み。解釈は下記のとおり。
#   ほぼ動かない（±0.1）  → 1容器族が支配的。以後の探索は1容器族に投資する
#   −0.2 前後            → v29→v32 と対称。2容器族シェア × 転移比 ≈ 0.28 を再確認
#   −0.5 以上下落        → 2容器族のシェアが高い。2容器族に投資する
HM_PRIO_RESERVE = _envf("GH_HM_PRIO_RESERVE", 0.0)

# HF-022 Phase 10: **soft 品のための容量予約**。既定 0.0 = 無効（v32 恒等）。
# soft_item_score の違反は2種類（analysis/platform_score.py::compute_soft）:
#   (a) ハード（非soft）が真上に載る  (b) **未積載**。soft-on-soft は OK。
# つまり soft も優先品と同じく「蓋」でなければならない ⇒ v32 で cog に効いた予約機構が
# そのまま移せる。soft は容器を問わないので **全容器**の上部を予約し、予約高は
# 残り soft の体積 / **全容器の合計床面積** から出す。
# soft は 30%（24/80）・総体積 1.23m3・最厚 0.400m と優先品（7個/0.45m3/0.250m）より
# 大きいので、同じ帯幅にするには係数を小さく取る（1.0 で 0.40m ＝ 内寸高の 26%）。
# v32 の実測で「観測域内ならモデルは健全」（残差 −0.35）と確認できており、soft は
# 一度も域外（43.9-53.8）へ出したことがない未開の軸。空きは 46 点で最大。
HM_SOFT_RESERVE = _envf("GH_HM_SOFT_RESERVE", 0.0)

# HF-022 Phase 11: **非soft を soft の上に載せることの禁止**（既定 0 = 無効、v33 恒等）。
# soft_item_score の違反内訳を実測すると（v32, dr10 239個）: 正常 51.9% / **未積載 34.7%** /
# **下敷き 13.4%**。前倒し（GH_SOFT_SHIFT<0）で未積載は 3.8% まで消せるが、下敷きが 23.8% に
# 増える（早く置く＝低い＝後続のハード品が上に載る）。減点 `HM_W_SOFT_VIOL=40` では足りない。
# 1 にすると候補段階で無効化する（順序は不変なので fill/np への影響は位置選択経由のみ）。
HM_SOFT_VETO = _envi("GH_HM_SOFT_VETO", 0)

# HF-025: **重い荷物の高所着地を硬く禁止する**（既定 -1 = 無効、v35 恒等）。
# 公式 cog は `100(1 − (z_com−z_floor)/(z_top−z_floor))`、`z_com = Σm_i z_i / Σm_i` で
# **質量加重**、しかも**置いた荷物だけ**を数える（analysis/platform_score.py::compute_cog、
# 定義は公開されている）。21提出を cog で並べると Public がほぼ単調で、cog+stab は
# `stab = 0.999·cog + 9.24`（r=0.9815）で連動し重み 0.429 を占める ⇒ **勝負は cog 一つ**。
# 帰結: **高所に行く重い荷物は「置かない」方が cog が上がる。** 成功レバー
# （TALL_SHIFT / PRIO_LATE / HEAVY_SHIFT）が効いた真の理由もこれ（PRIO_LATE は優先品
# 配置率を 0.629→0.100 に落として cog +7.40 を得た）。
# これまでの手は全て**減点**か**順序シフト**で、**硬い拒否**は未試行だった。
# 値は容器内高に対する比。0.5 なら「内高の 50% より上には重い荷物を置かせない」。
# v37 で **0.70 をベイク**（質量上位半分は内高の 70% より上に置かせない）。
# dr10 で 局所公式cog 63.70→64.12 / ctr_mean 0.3851→0.3814 / fill 26.37→26.56 / np 不変 ＝
# **無コスト**。0.80 は無効（重量物がそこまで上がらない）、0.50 は cog 66.11 まで伸びるが
# fill −1.51 / np −2.6pt を払う。0.70 が唯一の純増点。
# **v37 で 0.70 を提出して Public 60.00 → 60.00（変化なし）。既定 −1（無効）に戻した。**
# 局所公式cog は +0.42 と上がったのに **本番cog は −0.83 で符号が反転**した。
# 「cog が Public の支配変数」という観察（cog上位4件=Public上位4件）は**相関であって
# 因果ではない**と判定される。公式定義どおりに質量重心を下げても本番 cog は動かない。
HM_HEAVY_CEIL = _envf("GH_HM_HEAVY_CEIL", -1.0)
# 拒否の対象とする質量の下限（全荷物の質量分位。0.5 なら重い方の半分）。
HM_HEAVY_CEIL_Q = _envf("GH_HM_HEAVY_CEIL_Q", 0.5)

# HF-027: **優先品予約の過剰予約バグの修正**（既定 0 = 現状維持）。
# `heightmap.py` の予約帯は「優先品が積まれるたび縮む」設計だが、縮小判定が
# **コンテナごとの配置数を全体の総数と比較**していた（soft 側の同じ処理は全コンテナを合算
# していて正しい）。`GH_PRIO_LATE=1` は優先品を非優先コンテナの床に落とすので
# （constants の v26 注記「空いている2台目コンテナの床へ落ちている」）、それらは帯を
# 縮めない。結果**優先コンテナの上部が最後まで最低 hmax 分塞がれ続け**、非優先品の
# 配置を阻んで fill/np を直接失う。1 で全コンテナ合算に直す。
HM_PRIO_RESERVE_GLOBAL = _envi("GH_HM_PRIO_RESERVE_GLOBAL", 0)

# HF-027: **desperate 段の候補幅の復活**（既定 0 = 現状維持）。
# `HM_CASCADE` の第3段は `HM_STRAT_PER_BIN_DESP=14` / `HM_COARSE_TOPK_DESP=250` を
# 持っているが、**全ての展開箇所が `_pb, _ck` で捨てている**（heightmap.py の3箇所と
# rollout_plan.py）。候補は `_build_models_ranked` で strict 値（4 / 80）のみで一度だけ
# 生成されるので、**desperate 段で幅が広がっていない。** docs（constants の HF-010 注記）は
# 候補幅の拡大に配置数 +44% を帰属させている。
# 候補スコアは幅に依らないので、生成を desperate 幅で行えば全段が広い集合を見る
# （可行な配置が見つかる確率が上がる）。代償は生成時間で、3.5倍/3.1倍になる。
# **提出前に必ず opt 秒を確認する**（60秒超の2提出 v28/v34 はいずれも退行した）。
HM_WIDE_BREADTH = _envi("GH_HM_WIDE_BREADTH", 0)

# HF-022 Phase 11: 床体積ペナルティを**優先品には課さない**（既定 1 = 除外する）。
# `GH_HM_W_FLOOR_VOL=30` は局所 fill +1.46 / np +1.9pt を出すが placement を 43.4 → 27.7 に
# 破壊する。優先品は大きいので「大物を床から追い出す」圧に巻き込まれ、優先容器の予約帯の
# 運用が崩れる。fill の目的（除外される床層の体積を小さくする）は非優先品だけで達成できる。
HM_FLOOR_VOL_SKIP_PRIO = _envi("GH_HM_FLOOR_VOL_SKIP_PRIO", 1)
HM_SLAB_H = _envf("GH_HM_SLAB_H", 0.30)      # スラブ高 [m]（診断と同じ既定）
HM_SLAB_REF_Q = _envf("GH_HM_SLAB_REF_Q", 30.0)   # 作業面のパーセンタイル
HM_SOFT_FLAT_TOL = 0.005      # soft を flat 姿勢のみに制限する許容（hz が最小半寸+この値以下）
# HF-022: 搬入路シャドウ減点。荷物は側扉(y=-W/2)から +Y へスライドして入るため、ある配置は
# 「自分より奥・同じXレーン」の自由空間を到達不能にする。投了時の実測（GH_DIAG）で候補の
# 44-57% が l_path で落ちており、支持/包含の緩和も候補幅の拡張も効かなかった（いずれも 0 通過）。
# 壁は1手の探索ではなく「奥を塞ぐ順に置いてしまうこと」なので、塞ぐ量そのものを減点する。
# 値は「この配置が影にする自由体積 / 残り自由体積」(0-1) に掛かる。既定 0.0 = v14 恒等。
HM_W_BLOCK = _envf("GH_HM_W_BLOCK", 0.0)
# HF-022: 床直置き体積の減点。公式 `Evaluator.calculate_fill_rate` は「全8隅が全境界面から
# 5mm 以上内側」(inclusion_margin=-0.005) の荷物だけを fill に計上する。床面（法線[0,0,-1],
# z=thickness）は荷物の接地面そのものなので、**床に直置きした荷物は沈降後に必ず 5mm 不足で
# 除外される**（実測: 配置の21-24%・fill 10.3-12.0点ぶんが毎回捨てられている）。
# 一方いまの貪欲は W_BIG=25 と -W_HEIGHT*land で「大物を先に・低く」置くため、最も大きい荷物を
# 床に落として除外させている。床レベルに置く候補だけ体積に比例して減点し、床には小物・
# 上段には大物を誘導する。既定 0.0 = v14 恒等。
# v34 で **15.0** をベイク（優先品は HM_FLOOR_VOL_SKIP_PRIO で除外）。
# 実効重みの推定（16提出、v31除く15点、LOO残差RMS 0.425）で **fill の重みは 0.268 で最大**と
# 判明したため、床除外（配置の 21-26%・fill 7点相当が構造的に消える）を直接狙う本レバーを初測定。
# 15 は3ベンチで Δfill +0.35/+1.11/−0.50・Δnp +0.7/+1.1/+0.4pt・Δsoft +4.64/+3.64/+3.46 と
# ほぼ全軸正で ΔPublic +0.23/+0.10/+0.20。30 以上は placement を 43.4 → 26.5 に壊す
# （優先品を除外しても壊れる＝予約帯の運用が崩れる）ので採らない。
# **v34 で 15.0 を試して Public 59.5 → 55.0（−4.5）と大退行。既定 0.0 に戻した。**
# 局所 ctr_mean は改善（0.3672→0.3633）したのに本番 cog −7.81 / stability −7.44。
# 原因: 本レバーは**設計上「大きい荷物を上へ」送る**ので質量-高さ相関を反転させる。
# ctr_mean は中心高の**重みなし**平均なので大物が上がっても小物が床に降りれば相殺されて
# 改善に見えるが、本番 cog は質量分布に感応するため真の重心は上がる。
# ⇒ **ctr_mean 予測子は「大きさと高さの相関を変えない介入」にのみ有効**（順序系はOK）。
HM_W_FLOOR_VOL = _envf("GH_HM_W_FLOOR_VOL", 0.0)
# HF-022 安定性ゲート: 接地面の高低差（roughness = 窓内 max-min）の上限 [m]。
# 4786手の (特徴量→沈降結果) を集めて崩れ15件を分析した結果、崩れの中央 roughness は 0.878m
# （正常 0.220m）・最小でも 0.266m だった。既存の `_support_ok` は「最高点から3cm以内のセルの
# 割合」しか見ないため、深い谷にせり出したカンチレバーが通ってしまう（割合は満たせる）。
# roughness>0.25 で崩れ 15/15 を全捕捉できる（正常の棄却は30.9%）。
# 崩れは `is_placed_safe` 単独失敗＝エピソード即終了（−32pt np）なので、支持率を緩める前に
# こちらで裾を押さえる。`land - raw_min` は候補生成で既に計算済みなので追加コストはほぼゼロ。
# 既定は無効（大きな値）＝v14 恒等。
HM_MAX_ROUGH = _envf("GH_HM_MAX_ROUGH", 1.0e9)

# 空間層化 top-K 保持（偏り防止：Y×X ビンごと上位＋全体上位）。GH_HM_* で breadth を上書き可
# （EMS-skip 高速化で pmax<1s の余裕ができたため、候補を広げて placement 品質＝fill を狙う）。
HM_STRAT_BINS_Y = 6
HM_STRAT_BINS_X = 8
HM_STRAT_PER_BIN = _envi("GH_HM_STRAT_PER_BIN", 4)
HM_STRAT_PER_BIN_DESP = 14
HM_COARSE_TOPK = _envi("GH_HM_COARSE_TOPK", 80)
HM_COARSE_TOPK_DESP = 250

HM_DENSIFY_STEP = _envf("GH_HM_DENSIFY_STEP", 0.004)   # densify 後押しの刻み [m]
HM_DENSIFY_MAX = _envi("GH_HM_DENSIFY_MAX", 6)         # densify 各軸の最大ステップ数
HM_POOL_CAP = _envi("GH_HM_POOL_CAP", 8)              # 1手で候補生成する pool 荷物数の上限（体積降順）

# マージン緩和カスケード（strict→relaxed→desperate）。各段: (inclusion_margin, safety_margin,
# support_ratio_min, support_center_min, strat_per_bin, coarse_topk)。
# 注意: inclusion_margin は `container_space.contains_oriented_box` の符号規約
# （**正で厳格化**＝内側クリアランス増、0で面接触許容、負で緩和）で与える。これは公式 config の
# `inclusion_margin`（負で厳格）とは符号が逆。公式既定 -0.005(config)=+0.005(CO_box) の内側に
# desperate を置く（+0.0052）。safety_margin は check_l_path 用（大きいほど保守的、既定 0.015 以上）。
# safety_margin へ GH_HM_SAFETY_ADD を一律加算可能（check_l_path を保守化し、validator の
# check_transport_path との false-positive 不一致＝早期停止(is_valid=False)を減らす。HF-014）。
_SAFETY_ADD = _envf("GH_HM_SAFETY_ADD", 0.0)
HM_CASCADE = (
    (0.012, 0.022 + _SAFETY_ADD, HM_SUPPORT_RATIO_MIN, HM_SUPPORT_CENTER_MIN, HM_STRAT_PER_BIN, HM_COARSE_TOPK),
    (0.006, 0.0165 + _SAFETY_ADD, HM_SUPPORT_RATIO_MIN, HM_SUPPORT_CENTER_MIN, HM_STRAT_PER_BIN, HM_COARSE_TOPK),
    (0.0052, 0.0155 + _SAFETY_ADD, HM_SUPPORT_RATIO_DESP, HM_SUPPORT_CENTER_DESP, HM_STRAT_PER_BIN_DESP, HM_COARSE_TOPK_DESP),
)


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
