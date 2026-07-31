"""統合T-024: risk.provisional_p_ng の契約テスト（詳細仕様書 v1.18 §4.7、契約計画 §3.G）。

`risk.py` は未実装のため対象関数の import は各テスト関数内で行う。全12家族が分類 I。

完全式（§4.7）:
    support_ratio は [0,1] へクリップしてから使用する。
    cg_margin が -inf または非有限（NaN/inf）なら params.p_max を返す
        （support_ratio が非有限の場合も同様に params.p_max を返す）。
    通常時:
        clip(
            w_support_gap * (1.0 - support_ratio)
            + w_negative_cg * max(0.0, -cg_margin) * cg_scale,
            p_min, p_max,
        )
    戻り値は常に組込み float・有限・[p_min, p_max] 区間内。

既定パラメータ（`ProvisionalRiskParams`）: w_support_gap=0.6, w_negative_cg=0.4, cg_scale=10.0,
    p_min=0.0, p_max=0.95。
"""
import math

import pytest

from agents.heuristic.packing_core import constants


def _rp(**overrides):
    return constants.ProvisionalRiskParams(**overrides) if overrides else constants.ProvisionalRiskParams()


def _p_ng(support_ratio, cg_margin, params=None):
    from agents.heuristic.packing_core.risk import provisional_p_ng
    return provisional_p_ng(support_ratio=support_ratio, cg_margin=cg_margin, params=params or _rp())


# --- RISK-001: support=1, cg>=0 → p_min -----------------------------------------------------


def test_risk_001_full_support_nonnegative_cg_returns_p_min():
    rp = _rp()
    result = _p_ng(1.0, 0.5, rp)
    assert result == pytest.approx(rp.p_min)


# --- RISK-002: support_ratioは[0,1]へクリップしてから使用 ------------------------------------


def test_risk_002_support_ratio_clipped_before_use():
    rp = _rp()
    result_over = _p_ng(1.5, 0.0, rp)
    result_at_one = _p_ng(1.0, 0.0, rp)
    assert result_over == pytest.approx(result_at_one)

    result_under = _p_ng(-0.5, 0.0, rp)
    result_at_zero = _p_ng(0.0, 0.0, rp)
    assert result_under == pytest.approx(result_at_zero)


# --- RISK-003: 通常式 --------------------------------------------------------------------------


def test_risk_003_normal_formula_matches_hand_calculation():
    rp = _rp()
    support, cg = 0.5, -0.1
    expected = rp.w_support_gap * (1.0 - support) + rp.w_negative_cg * max(0.0, -cg) * rp.cg_scale
    expected = min(max(expected, rp.p_min), rp.p_max)
    # 手計算: 0.6*0.5 + 0.4*0.1*10 = 0.3+0.4 = 0.7
    assert expected == pytest.approx(0.7)

    result = _p_ng(support, cg, rp)
    assert result == pytest.approx(expected)


# --- RISK-004: 戻り値契約（型・有限性・範囲） --------------------------------------------------


def test_risk_004_return_value_contract():
    rp = _rp()
    result = _p_ng(0.5, -0.1, rp)
    assert type(result) is float
    assert math.isfinite(result)
    assert rp.p_min <= result <= rp.p_max


# --- RISK-005: support単調性（param×2） --------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi",
    [
        pytest.param(0.0, 0.5, id="T024-RISK-005-LOW"),
        pytest.param(0.5, 1.0, id="T024-RISK-005-HIGH"),
    ],
)
def test_risk_005_support_monotonicity(lo, hi):
    """support_ratioが高いほどp_ngは高くない（単調非増加。既定係数では狭義単調減少）。"""
    rp = _rp()
    p_lo = _p_ng(lo, 0.0, rp)
    p_hi = _p_ng(hi, 0.0, rp)
    assert p_lo > p_hi


# --- RISK-006: support非有限→p_max（param×3） --------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="T024-RISK-006-NAN"),
        pytest.param(float("inf"), id="T024-RISK-006-PINF"),
        pytest.param(float("-inf"), id="T024-RISK-006-NINF"),
    ],
)
def test_risk_006_non_finite_support_returns_p_max(value):
    rp = _rp()
    result = _p_ng(value, 0.0, rp)
    assert result == pytest.approx(rp.p_max)


# --- RISK-007: cg_margin非有限/-inf→p_max（param×3） -------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="T024-RISK-007-NAN"),
        pytest.param(float("inf"), id="T024-RISK-007-PINF"),
        pytest.param(float("-inf"), id="T024-RISK-007-NINF"),
    ],
)
def test_risk_007_non_finite_cg_margin_returns_p_max(value):
    rp = _rp()
    result = _p_ng(1.0, value, rp)
    assert result == pytest.approx(rp.p_max)


# --- RISK-008: cg_margin単調性 ------------------------------------------------------------------


def test_risk_008_cg_margin_monotonicity():
    """cg_marginが負に大きいほどp_ngは高い（既定係数では狭義単調増加）。"""
    rp = _rp()
    p_small_neg = _p_ng(1.0, -0.1, rp)
    p_large_neg = _p_ng(1.0, -0.5, rp)
    assert p_large_neg > p_small_neg


# --- RISK-009: p_maxクリップ（上限超過時は上限どまり） -------------------------------------------


def test_risk_009_clips_at_p_max_for_extreme_inputs():
    rp = _rp()
    result = _p_ng(0.0, -1000.0, rp)
    assert result == pytest.approx(rp.p_max)
    assert result <= rp.p_max


# --- RISK-010: 非破壊・決定論的（同一入力を2回呼んでも同じ結果、paramsは不変） -------------------


def test_risk_010_deterministic_and_non_destructive():
    rp = _rp()
    fields_before = (rp.w_support_gap, rp.w_negative_cg, rp.cg_scale, rp.p_min, rp.p_max)

    result_1 = _p_ng(0.5, -0.1, rp)
    result_2 = _p_ng(0.5, -0.1, rp)

    assert result_1 == pytest.approx(result_2)
    fields_after = (rp.w_support_gap, rp.w_negative_cg, rp.cg_scale, rp.p_min, rp.p_max)
    assert fields_before == fields_after


# --- RISK-011: 定数源が単一情報源（他へ重複直書きしていない） -------------------------------------


def test_risk_011_uses_params_as_single_source_not_hardcoded():
    """paramsのフィールドを変えると結果が変わる（agent.py/score.pyへ重複直書きされていない
    ことの間接証拠：関数がparamsを実際に参照している）。"""
    rp_default = _rp()
    rp_custom = _rp(w_support_gap=0.1, w_negative_cg=0.9, cg_scale=2.0, p_min=0.0, p_max=0.5)

    result_default = _p_ng(0.5, -0.1, rp_default)
    result_custom = _p_ng(0.5, -0.1, rp_custom)
    assert result_default != pytest.approx(result_custom)


# --- RISK-012: cg_margin>=0はmax(0,-cg)=0で同値 -------------------------------------------------


def test_risk_012_nonnegative_cg_margin_all_equivalent_to_zero():
    rp = _rp()
    result_zero = _p_ng(0.5, 0.0, rp)
    result_positive = _p_ng(0.5, 5.0, rp)
    assert result_zero == pytest.approx(result_positive)
