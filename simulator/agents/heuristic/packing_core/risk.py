"""暫定リスク推定（詳細仕様書 §4.7、T-024所有）。

`provisional_p_ng` は学習済み `RiskModel`（T-035/T-036）配備前・配備後どちらでも使う
モデル非依存の純関数。npz読込・学習済み係数はここでは実装しない（§4.12「T-033〜T-036との
責務境界」参照）。
"""
import math

from .constants import ProvisionalRiskParams


def provisional_p_ng(
    support_ratio: float, cg_margin: float, params: ProvisionalRiskParams
) -> float:
    """支持率・重心マージンから配置失敗確率の暫定推定値を返す（§4.7）。

    `support_ratio` は `[0,1]` へクリップしてから使用する。`cg_margin` が非有限
    （NaN/±inf、`-inf` を含む）または `support_ratio` が非有限の場合は、NaN を
    順位付け・p_success へ流さないためのガードとして `params.p_max` を返す。

    Args:
        support_ratio: 支持セル面積比（`stability.support_ratio` の戻り値）。
        cg_margin: 重心の支持凸包境界までの符号付き距離（`stability.cg_margin` の戻り値）。
        params: `ProvisionalRiskParams`（単一情報源、係数を他モジュールへ重複直書きしない）。

    Returns:
        `[params.p_min, params.p_max]` 区間内の有限な組込み float。
    """
    if not math.isfinite(support_ratio) or not math.isfinite(cg_margin):
        return float(params.p_max)

    clipped_support = min(max(support_ratio, 0.0), 1.0)
    raw = (
        params.w_support_gap * (1.0 - clipped_support)
        + params.w_negative_cg * max(0.0, -cg_margin) * params.cg_scale
    )
    return float(min(max(raw, params.p_min), params.p_max))
