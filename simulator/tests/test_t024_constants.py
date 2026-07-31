"""統合T-024: constants.py の契約テスト（詳細仕様書 v1.18 §3.5、契約計画 §3.A）。

T-024 が新規に所有する `PlacementParams.candidate_generation_slack` と
`ProvisionalRiskParams`（いずれも未実装）は、モジュール直下では参照せず各テスト関数内で
参照する（collection error 回避、`tests/test_masks.py`・`tests/test_stability.py` と同方針）。
`constants.py` 自体は T-002〜T-021 で実装済みのため、`constants` モジュールの import・既存
dataclass のインスタンス化はモジュール直下で行ってよい。

契約ファミリー（8）：
    CONST-001/002（I）: PlacementParams.candidate_generation_slack の存在・値=1e-6
    CONST-003（III）: PlacementParams が frozen（既存契約）
    CONST-004×5（III）: 既存マージン定数が不変（inclusion/safety/start_z/ceiling/internal_extra）
    CONST-005（III）: T-024が依存する既存定数（budget_poll_every/l_path_top_m/POOL_MAX_WEIGHT）
    CONST-006（I）: ProvisionalRiskParams の存在・frozen
    CONST-007×5（I）: ProvisionalRiskParams の既定値（w_support_gap/w_negative_cg/cg_scale/p_min/p_max）
    CONST-008（I）: ProvisionalRiskParams が他 dataclass と独立した単一情報源であること
"""
import dataclasses

import pytest

from agents.heuristic.packing_core import constants

PP0 = constants.PlacementParams()  # 既存フィールドのみで構築可能（新規フィールド未実装でもOK）


# --- CONST-001/002: candidate_generation_slack（I） -------------------------------------


def test_const_001_candidate_generation_slack_exists():
    """`PlacementParams.candidate_generation_slack` が存在すること（v1.16新設、T-024所有）。"""
    pp = constants.PlacementParams()
    assert hasattr(pp, "candidate_generation_slack")


def test_const_002_candidate_generation_slack_value_is_1e_minus_6():
    """`candidate_generation_slack` の既定値は 1e-6[m]（§3.5）。"""
    pp = constants.PlacementParams()
    assert pp.candidate_generation_slack == pytest.approx(1e-6)


# --- CONST-003: PlacementParams は frozen（既存契約、III） -------------------------------


def test_const_003_placement_params_is_frozen():
    """既存契約：`PlacementParams` は frozen dataclass であり属性再代入は失敗する。"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        PP0.start_z = 999.0


# --- CONST-004: 既存マージン定数が不変（III、param×5） -----------------------------------


@pytest.mark.parametrize(
    "attr,expected",
    [
        pytest.param("inclusion_margin", -0.005, id="T024-CONST-004-INCLUSION"),
        pytest.param("safety_margin", 0.015, id="T024-CONST-004-SAFETY"),
        pytest.param("start_z", 0.08, id="T024-CONST-004-STARTZ"),
        pytest.param("ceiling_margin", 0.018, id="T024-CONST-004-CEILING"),
        pytest.param("internal_extra", 0.001, id="T024-CONST-004-INTERNAL"),
    ],
)
def test_const_004_existing_margins_unchanged(attr, expected):
    """T-024が候補生成式で参照する既存マージン定数が変更されていないこと。"""
    assert getattr(PP0, attr) == pytest.approx(expected)


# --- CONST-005: T-024が依存する既存定数（III） --------------------------------------------


def test_const_005_existing_dependent_constants_unchanged():
    """T-024の4層・候補生成が参照する既存定数（TimeParams/StageParams/POOL_MAX_WEIGHT）。"""
    tp = constants.TimeParams()
    sp = constants.StageParams()
    assert tp.budget_poll_every == 64
    assert isinstance(tp.budget_poll_every, int)
    assert sp.l_path_top_m == 160
    assert constants.POOL_MAX_WEIGHT == pytest.approx(18.0)


# --- CONST-006: ProvisionalRiskParams の存在・frozen（I） --------------------------------


def test_const_006_provisional_risk_params_exists_and_is_frozen():
    """`constants.ProvisionalRiskParams`（v1.16新設、T-024所有）が存在し frozen であること。"""
    rp = constants.ProvisionalRiskParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rp.p_max = 0.5


# --- CONST-007: ProvisionalRiskParams の既定値（I、param×5） -----------------------------


@pytest.mark.parametrize(
    "attr,expected",
    [
        pytest.param("w_support_gap", 0.6, id="T024-CONST-007-WSUPPORT"),
        pytest.param("w_negative_cg", 0.4, id="T024-CONST-007-WNEGCG"),
        pytest.param("cg_scale", 10.0, id="T024-CONST-007-CGSCALE"),
        pytest.param("p_min", 0.0, id="T024-CONST-007-PMIN"),
        pytest.param("p_max", 0.95, id="T024-CONST-007-PMAX"),
    ],
)
def test_const_007_provisional_risk_params_defaults(attr, expected):
    """`ProvisionalRiskParams` 既定値が §3.5 の単一情報源と一致すること。"""
    rp = constants.ProvisionalRiskParams()
    assert getattr(rp, attr) == pytest.approx(expected)


# --- CONST-008: 独立した単一情報源であること（I） -----------------------------------------


def test_const_008_provisional_risk_params_is_independent_single_source():
    """`ProvisionalRiskParams` が他 dataclass（`ScoreParams` 等）と別の独立した型・
    フィールド集合を持つこと（重複定義・エイリアス化されていないこと）。"""
    rp_field_names = {f.name for f in dataclasses.fields(constants.ProvisionalRiskParams)}
    score_field_names = {f.name for f in dataclasses.fields(constants.ScoreParams)}

    assert constants.ProvisionalRiskParams is not constants.ScoreParams
    assert rp_field_names == {"w_support_gap", "w_negative_cg", "cg_scale", "p_min", "p_max"}
    # ScoreParams と重複するフィールド名を持たない（別の情報源であることの担保）。
    assert rp_field_names.isdisjoint(score_field_names)

    # 独立インスタンス化：一方の変更が他方に波及しない（frozenの新規インスタンス生成で確認）。
    rp_a = constants.ProvisionalRiskParams(p_max=0.5)
    rp_b = constants.ProvisionalRiskParams()
    assert rp_a.p_max == pytest.approx(0.5)
    assert rp_b.p_max == pytest.approx(0.95)
