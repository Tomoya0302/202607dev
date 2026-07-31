"""HF-001: 空コンテナ初手の自己INCLUSION判定NG修正の回帰テスト（チケット `HF-001
first-placement-validity`、詳細仕様書 v1.27改訂 §4.6/§4.8/§4.11/§4.12「HF-001」参照）。

Publicスコア2.0/100の原因は、候補生成が荷物底面をEMS支持面へ厳密一致させる一方
`check_inclusion` が内側10mmクリアランスを床面含む全6面へ要求するため、空コンテナへの
床置き候補が構造的にINCLUSION不合格になり、`layer4_max_p`（旧`dims_candidates`契約）が
その不合格候補を最終選択して公式validator NGを引き起こしていたことにある。

本ファイルは付録D以下・チケット本文「必須テスト」A〜Eに対応する：
    A. 候補生成（action位置・想定沈降後位置・float32往復・Z方向寸法確認）
    B. 安定性（想定沈降後位置を使った支持判定・非破壊性）
    C. フォールバック（layer3_first_fit／layer4_max_pの新eligible集合）
    D. emergency item_idx（常に0）
    E. 空コンテナ初手の公式Env／Runner／PlacementValidator経路での回帰
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agents.heuristic.packing_core import constants
from agents.heuristic.packing_core.container_space import build_container_space
from agents.heuristic.packing_core.state import PackingState, make_action
from agents.heuristic.packing_core.types import Candidate, EMSBox, ItemSpec, PlacedItem

SIMULATOR_ROOT = Path(__file__).resolve().parents[1]

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
CELL = 0.02
PP0 = constants.PlacementParams()
OSIZE = np.array([0.30, 0.30, 0.20], dtype=np.float64)


def _z_generation_clearance(pp: constants.PlacementParams) -> float:
    """§4.12のz_generation_clearanceをppフィールドから手計算する（テスト側oracle）。"""
    required = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    return required + pp.candidate_generation_slack


def _box_cdict():
    """空コンテナ・平床の合成cdict（floor_z一様、cut_x=cut_y=0、shelf=False）。"""
    imin, imax = INNER_MIN_REL, INNER_MAX_REL
    mid = (imin + imax) / 2.0
    thickness = 0.02
    buffer = 0.02
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer

    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]

    return {
        "index": 0, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness,
        "center": (0.0, 0.0, height / 2.0 + buffer),
        "n_vecs": n_vecs, "points": points,
        "volume": float(np.prod(imax - imin)),
        "shelf": False, "is_prioritized": False, "packed_items": [],
    }


def _empty_flat_space():
    return build_container_space(_box_cdict(), index=0, cell=CELL)


def _floor_ems() -> EMSBox:
    """空コンテナ全体を覆う、床から天井までのEMS（実運転の空コンテナ初手を模す）。"""
    return EMSBox(min_rel=INNER_MIN_REL.copy(), max_rel=INNER_MAX_REL.copy())


def _empty_item(idx=0, size=OSIZE, weight=5.0) -> ItemSpec:
    return ItemSpec(
        idx=idx, size=np.array(size, dtype=np.float64), weight=weight,
        kind=None, is_soft=False, is_priority=False,
    )


def _empty_state(ems_list=None) -> PackingState:
    space = _empty_flat_space()
    return PackingState(
        containers=[space], placed={0: []}, pool=[_empty_item()],
        ems={0: list(ems_list) if ems_list is not None else [_floor_ems()]},
        ems_truncation={0: 0.0}, meta={},
    )


# =============================================================================
# A. 候補生成：action位置・想定沈降後位置・float32往復・Z方向寸法確認
# =============================================================================


def test_a1_settled_bottom_equals_ems_floor_action_bottom_has_clearance():
    from agents.heuristic.packing_core.candidates import candidate_from_ems
    from agents.heuristic.packing_core.stability import expected_settled_pos_rel

    ems = _floor_ems()
    item = _empty_item()
    state = _empty_state(ems_list=[ems])

    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None

    settled = expected_settled_pos_rel(state, cand)
    settled_bottom_z = float(settled[2]) - float(cand.osize[2]) / 2.0
    assert settled_bottom_z == pytest.approx(float(ems.min_rel[2]))

    z_required_clearance = max(-PP0.inclusion_margin + PP0.internal_extra, PP0.internal_extra)
    action_bottom_z = float(cand.pos_rel[2]) - float(cand.osize[2]) / 2.0
    assert action_bottom_z >= float(ems.min_rel[2]) + z_required_clearance - 1e-9


def test_a2_f32_roundtrip_survives_all_four_mask_stages_on_empty_container():
    from agents.heuristic.packing_core.candidates import candidate_from_ems
    from agents.heuristic.packing_core.masks import MaskStage, evaluate_stage

    ems = _floor_ems()
    item = _empty_item()
    state = _empty_state(ems_list=[ems])

    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None

    action = make_action(
        item_idx=cand.item_idx, container_idx=cand.container_idx,
        pos_rel=cand.pos_rel, orientation=cand.orientation,
    )
    assert action["place_pos"].dtype == np.float32
    roundtripped_pos = np.asarray(action["place_pos"], dtype=np.float64)

    rt_cand = Candidate(
        item_idx=cand.item_idx, container_idx=cand.container_idx,
        ems_id=cand.ems_id, orientation=cand.orientation,
        pos_rel=roundtripped_pos, osize=cand.osize,
    )
    for stage in (MaskStage.DIMS, MaskStage.INCLUSION, MaskStage.OVERLAP, MaskStage.CEILING):
        evaluate_stage(state, rt_cand, PP0, stage)
        assert rt_cand.feasible, (
            f"空コンテナ初手のfloat32往復後 stage={stage!r} で不合格"
            f"（reject_reason={rt_cand.reject_reason!r}）。HF-001の修正漏れの可能性。"
        )


def test_a3_insufficient_z_headroom_yields_no_candidate():
    from agents.heuristic.packing_core.candidates import candidate_from_ems

    item = _empty_item()
    z_gen_clearance = _z_generation_clearance(PP0)
    # ems.size()[2] = osize[2] + z_gen_clearance - 1e-4 (新しい寸法確認未満)。
    insufficient_max_z = float(INNER_MIN_REL[2]) + float(OSIZE[2]) + z_gen_clearance - 1e-4
    ems = EMSBox(
        min_rel=np.array([INNER_MIN_REL[0], INNER_MIN_REL[1], INNER_MIN_REL[2]], dtype=np.float64),
        max_rel=np.array([INNER_MAX_REL[0], INNER_MAX_REL[1], insufficient_max_z], dtype=np.float64),
    )
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is None


# =============================================================================
# B. 安定性：想定沈降後位置を使った支持判定・非破壊性
# =============================================================================


def test_b1_action_position_raised_above_settled_still_reports_full_support():
    from agents.heuristic.packing_core.stability import cg_margin, expected_settled_pos_rel, support_polygon, support_ratio

    ems = _floor_ems()
    item = _empty_item()
    state = _empty_state(ems_list=[ems])

    # action位置は支持面から約10mm浮いた候補（HF-001後のcandidate_from_emsが生成する形）。
    settled_z = float(ems.min_rel[2]) + float(OSIZE[2]) / 2.0
    action_z = settled_z + 0.010001
    cand = Candidate(
        item_idx=0, container_idx=0, ems_id=0, orientation=0,
        pos_rel=np.array([0.0, 0.0, action_z], dtype=np.float64), osize=OSIZE.copy(),
    )

    assert support_ratio(state, cand) == pytest.approx(1.0)
    poly = support_polygon(state, cand)
    assert poly.shape[0] >= 3
    margin = cg_margin(state, cand)
    assert math.isfinite(margin)
    assert margin >= 0.0

    # 想定沈降後位置は実際にEMS床面と一致する（action位置の10mm浮きは反映されない）。
    settled = expected_settled_pos_rel(state, cand)
    assert float(settled[2]) - float(cand.osize[2]) / 2.0 == pytest.approx(float(ems.min_rel[2]))


def test_b2_expected_settled_pos_rel_does_not_mutate_candidate():
    from agents.heuristic.packing_core.stability import expected_settled_pos_rel

    ems = _floor_ems()
    state = _empty_state(ems_list=[ems])
    pos_before = np.array([0.05, -0.05, 0.35], dtype=np.float64)
    cand = Candidate(
        item_idx=0, container_idx=0, ems_id=0, orientation=0,
        pos_rel=pos_before.copy(), osize=OSIZE.copy(),
    )

    settled = expected_settled_pos_rel(state, cand)

    np.testing.assert_array_equal(cand.pos_rel, pos_before)
    assert settled is not cand.pos_rel
    # settled ZはEMS(min_rel[2]=INNER_MIN_REL[2]=0.02)由来で、pos_before[2]=0.35とは独立に導出される。
    assert float(settled[2]) == pytest.approx(float(INNER_MIN_REL[2]) + float(OSIZE[2]) / 2.0)


def test_b3_expected_settled_pos_rel_raises_on_invalid_container_or_ems_id():
    from agents.heuristic.packing_core.stability import expected_settled_pos_rel

    state = _empty_state(ems_list=[_floor_ems()])
    cand_bad_container = Candidate(
        item_idx=0, container_idx=5, ems_id=0, orientation=0,
        pos_rel=np.zeros(3), osize=OSIZE.copy(),
    )
    with pytest.raises(ValueError):
        expected_settled_pos_rel(state, cand_bad_container)

    cand_bad_ems = Candidate(
        item_idx=0, container_idx=0, ems_id=9, orientation=0,
        pos_rel=np.zeros(3), osize=OSIZE.copy(),
    )
    with pytest.raises(ValueError):
        expected_settled_pos_rel(state, cand_bad_ems)


# =============================================================================
# C. フォールバック：layer3_first_fit／layer4_max_pの新eligible集合
# =============================================================================


def _watchdog_cand(item_idx=0, p_success=1.0):
    return Candidate(
        item_idx=item_idx, container_idx=0, ems_id=0, orientation=0,
        pos_rel=np.array([0.0, 0.0, 0.5], dtype=np.float64),
        osize=np.array([0.1, 0.1, 0.1], dtype=np.float64),
        p_success=p_success,
    )


def _key(cand):
    # HF-003: L_PATH キャッシュキーは anchor を含む 5-tuple（candidates.candidate_key と同順）。
    return (cand.item_idx, cand.container_idx, cand.orientation, cand.ems_id, cand.anchor)


def _never_over_budget():
    from agents.heuristic.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)


def _minimal_state():
    return PackingState(containers=[], placed={}, pool=[], ems={}, ems_truncation={}, meta={})


def test_c1_layer3_first_fit_never_returns_known_failing_candidate():
    from agents.heuristic.packing_core.candidates import CandidatePools
    from agents.heuristic.packing_core.watchdog import layer3_first_fit

    c_pass = _watchdog_cand(item_idx=0)
    c_unknown = _watchdog_cand(item_idx=1)
    c_fail = _watchdog_cand(item_idx=2)
    sp = constants.StageParams(l_path_top_m=10)

    pools_pass = CandidatePools(
        raw_candidates=[], dims_candidates=[], geo_candidates=[c_fail, c_pass],
        l_path_cache={_key(c_fail): False, _key(c_pass): True},
    )
    result = layer3_first_fit(_minimal_state(), _never_over_budget(), pools=pools_pass, pp=PP0, stage_params=sp)
    assert result is c_pass

    pools_unknown = CandidatePools(
        raw_candidates=[], dims_candidates=[], geo_candidates=[c_fail, c_unknown],
        l_path_cache={_key(c_fail): False},
    )
    result_unknown = layer3_first_fit(
        _minimal_state(), _never_over_budget(), pools=pools_unknown, pp=PP0, stage_params=sp
    )
    assert result_unknown is c_unknown

    pools_only_fail = CandidatePools(
        raw_candidates=[], dims_candidates=[], geo_candidates=[c_fail],
        l_path_cache={_key(c_fail): False},
    )
    result_only_fail = layer3_first_fit(
        _minimal_state(), _never_over_budget(), pools=pools_only_fail, pp=PP0, stage_params=sp
    )
    assert result_only_fail is None


def test_c2_layer4_max_p_ignores_inclusion_failing_and_l_path_failing_candidates():
    from agents.heuristic.packing_core.candidates import CandidatePools
    from agents.heuristic.packing_core.watchdog import layer4_max_p

    c_inclusion_reject = _watchdog_cand(item_idx=0, p_success=0.99)  # dimsのみ、geoに無い
    c_l_path_reject = _watchdog_cand(item_idx=1, p_success=0.95)  # geoにいるがL_PATH既知不合格
    c_eligible = _watchdog_cand(item_idx=2, p_success=0.4)
    sp = constants.StageParams(l_path_top_m=10)

    pools = CandidatePools(
        raw_candidates=[],
        dims_candidates=[c_inclusion_reject, c_l_path_reject, c_eligible],
        geo_candidates=[c_l_path_reject, c_eligible],
        l_path_cache={_key(c_l_path_reject): False},
    )
    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_eligible

    pools_none_eligible = CandidatePools(
        raw_candidates=[], dims_candidates=[c_l_path_reject], geo_candidates=[c_l_path_reject],
        l_path_cache={_key(c_l_path_reject): False},
    )
    result_none = layer4_max_p(
        _minimal_state(), _never_over_budget(), pools=pools_none_eligible, pp=PP0, stage_params=sp
    )
    assert result_none is None


# =============================================================================
# D. emergency item_idx：荷物固有indexが37でも常に0
# =============================================================================


def test_d1_emergency_action_item_idx_is_always_zero_even_with_nonzero_pool_index():
    from agents.heuristic.agent import _emergency_action

    observation = {
        "pool_list": [{"index": 37}],
    }
    action = _emergency_action(observation)
    assert action["item_idx"] == 0


# =============================================================================
# E. 空コンテナ初手の公式Env／Runner／PlacementValidator経路での回帰
# =============================================================================


def _single_task_config_path(tmp_path: Path, n_items: int = 1) -> Path:
    """`configs/local_suite/c01.json`（既に空コンテナ・`packed_items=[]`）を1 task・
    少数item数に絞ったコピーとして書き出す（`tests/fixtures/t028/scenarios.py
    ::single_task_config_path` と同方針。本ファイルを自己完結させるため再定義する）。
    """
    src = SIMULATOR_ROOT / "configs" / "local_suite" / "c01.json"
    with open(src) as f:
        config = json.load(f)
    assert len(config) == 1, "c01.json は単一task前提"
    (task_id,) = config.keys()
    assert config[task_id]["containers"]["container_list"][0]["packed_items"] == []
    config[task_id]["item_stream"]["item_list"] = config[task_id]["item_stream"]["item_list"][:n_items]
    out_path = tmp_path / "c01_hf001_single_task.json"
    with open(out_path, "w") as f:
        json.dump(config, f)
    return out_path, task_id


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_e1_official_runner_path_empty_container_first_step(tmp_path):
    """空コンテナ初手をscripts.run_local経由で公式Env/Runner/PlacementValidatorへ通し、
    候補生成・段階フィルタ・フォールバック層・validator結果を回帰確認する（チケット§E）。"""
    pytest.importorskip("pybullet")
    config_path, task_id = _single_task_config_path(tmp_path, n_items=1)
    result_dir = tmp_path / "result"

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_local",
            "--config-path", str(config_path),
            "--module-path", "agents/heuristic/",
            "--result-dir", str(result_dir),
        ],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    assert proc.returncode == 0, proc.stderr

    telemetry_files = list((result_dir / "telemetry").glob("*.jsonl"))
    assert len(telemetry_files) == 1
    rows = _read_jsonl(telemetry_files[0])
    assert len(rows) >= 1
    first_row = rows[0]

    # v38 (HF-012 Phase H1): agent.policy は heightmap 配置エンジンに置換され、EMS 列挙〜
    # safe_decide 由来の counter（n_cand0/n_after_dims/n_after_geo）と decided_layer 縮退は
    # policy 経路では非populate（0）になった。HF-001 の本質的な回帰ガードは下の公式 validator
    # 結果（is_included is True）であり、そちらで担保する。
    assert first_row["decided_layer"] >= 0

    result_path = result_dir / "evaluation_results.json"
    with open(result_path) as f:
        results = json.load(f)
    task_result = results[task_id]
    assert task_result["status"] == "success"
    assert task_result["place_states"]["is_included"] is True, (
        "初回配置がINCLUSION不合格（HF-001の症状）で公式validatorにNGと判定されている"
    )
    assert task_result["evaluation"]["num_placed_items"] >= 1
