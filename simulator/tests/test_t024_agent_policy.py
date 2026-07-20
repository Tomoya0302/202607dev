"""統合T-024: agents/heuristic/agent.py 縦断配線の契約テスト（詳細仕様書 v1.18 §4.12
「Agent配線契約」「telemetryローカル辞書契約」、契約計画 §3.J/K）。

Agent は T-023 骨格（固定placeholder action）のみ実装済みで、手順2〜7の縦断配線は未実装のため、
対象モジュール（candidates.py/score.py/risk.py/watchdog.pyの4層）の import は各テスト関数内で
行う。20家族の分類は III=6（既存骨格が既に満たす）／II=3（T-023固定placeholderと矛盾）／
I=11（新規配線が未実装）。

Q5（AGENT-008）の観測法: `safe_decide` を「実体へ委譲しつつ引数を記録する」ラッパでspy化し、
`layers[0].keywords["pools"]`（4層は functools.partial で pools/pp/stage_params を束縛して
渡す契約、§4.11）から実際の `CandidatePools` を取得して実値検証する。import文の形式に
依存しないよう `watchdog.safe_decide` と `agents.heuristic.agent.safe_decide` の両方へ
（後者は raising=False で）patchする。
"""
import numpy as np
import pytest

INNER_MIN_SMALL = np.array([-0.02, -0.02, 0.02], dtype=np.float64)
INNER_MAX_SMALL = np.array([0.02, 0.02, 0.05], dtype=np.float64)  # container0: どの荷物も入らない
INNER_MIN_BIG = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_BIG = np.array([0.40, 0.40, 1.02], dtype=np.float64)   # container1: 十分な広さ


def _item_dict(index, size, mass, is_soft=False, is_priority=False):
    length, width, height = (float(v) for v in size)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "mass": float(mass), "is_prioritized": bool(is_priority), "is_soft": bool(is_soft),
        "belongs_to": None, "pos": None, "orn": None,
        "lateralFriction": 0.5, "rollingFriction": 0.01, "spinningFriction": 0.01,
        "restitution": 0.1, "angularDamping": 0.05,
    }


def _container_dict(index, offset_x, inner_min_rel, inner_max_rel, packed_items=None,
                     thickness=0.02, buffer=0.02):
    imin = np.asarray(inner_min_rel, dtype=np.float64)
    imax = np.asarray(inner_max_rel, dtype=np.float64)
    mid = (imin + imax) / 2.0
    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px + offset_x, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer
    center = (float(offset_x), 0.0, height / 2.0 + buffer)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness, "center": center,
        "n_vecs": n_vecs, "points": points, "volume": float(np.prod(imax - imin)),
        "shelf": False, "is_prioritized": False, "packed_items": list(packed_items or []),
    }


def _two_container_lists():
    """container0=極小（どの荷物も不適合）、container1=十分な広さ。init/observationで
    別オブジェクトの container_list を渡す（build_stateの非改変契約に沿う）。"""
    specs = [(0, 0.0, INNER_MIN_SMALL, INNER_MAX_SMALL), (1, 2.0, INNER_MIN_BIG, INNER_MAX_BIG)]
    init_list = [_container_dict(idx, off, imin, imax) for idx, off, imin, imax in specs]
    obs_list = [_container_dict(idx, off, imin, imax) for idx, off, imin, imax in specs]
    return init_list, obs_list


def _placeable_fixture():
    """container1にのみ収まる荷物1件を持つ、実配置可能なinit/observationペア。"""
    init_containers, obs_containers = _two_container_lists()
    pool_list = [_item_dict(index=0, size=(0.1, 0.1, 0.1), mass=2.0)]
    init = {"optimize": True, "lookahead_k": 5, "container_list": init_containers}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((2, 64, 64), dtype=np.float32),
        "container_list": obs_containers, "pool_list": pool_list,
    }
    return init, observation


def _make_agent(init):
    from agents.heuristic.agent import Agent
    agent = Agent(module_path="agents/heuristic")
    agent.get_init_states(init)
    return agent


def _deep_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _spy_safe_decide_passthrough(monkeypatch):
    """`safe_decide`を実体へ委譲しつつ引数を記録するラッパでspy化する。

    Returns: `captured` dict。呼出し後に
        captured["layers"]: safe_decideへ渡されたlayers列
        captured["telemetry_before"]: 呼出し直前のtelemetryの浅いコピー（safe_decide呼出し前状態）
        captured["telemetry_live"]: telemetryそのもの（同一オブジェクト、呼出し後の変更も見える）
    を持つ（未呼出しなら空のまま）。
    """
    from src.packing_core import watchdog
    original = watchdog.safe_decide  # T-024実装後に存在。未実装ならAttributeErrorで即RED。

    captured = {}

    def _wrapper(layers, state, budget, telemetry):
        captured["layers"] = layers
        captured["telemetry_before"] = dict(telemetry)
        result = original(layers, state, budget, telemetry)
        captured["telemetry_live"] = telemetry
        return result

    monkeypatch.setattr(watchdog, "safe_decide", _wrapper)
    try:
        from agents.heuristic import agent as agent_module
        monkeypatch.setattr(agent_module, "safe_decide", _wrapper, raising=False)
    except ImportError:
        pass
    return captured


def _observed_pools(captured):
    layers = captured["layers"]
    assert len(layers) == 4
    return layers[0].keywords["pools"]


# --- BUDGET-001/002 ---------------------------------------------------------------------------


def test_budget_001_step_budget_constructed_exactly_once(monkeypatch):
    from src.packing_core import watchdog

    # T-027: __init__ warmupは別契約で検証し、ここでは1回の実policyだけを観測する。
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    calls = {"n": 0}
    original_init = watchdog.StepBudget.__init__

    def _counting_init(self, *args, **kwargs):
        calls["n"] += 1
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(watchdog.StepBudget, "__init__", _counting_init)

    agent.policy(observation)

    assert calls["n"] == 1


def test_budget_002_all_pipeline_stages_share_the_same_budget_object(monkeypatch):
    from src.packing_core import candidates, watchdog

    # T-027: spyの観測窓はAgent生成後の1回の実policyに限定する。
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    budgets_seen = []

    original_enum = candidates.enumerate_candidates

    def _enum_wrapper(state, pp, tp, budget):
        budgets_seen.append(budget)
        return original_enum(state, pp, tp, budget)

    original_filter = candidates.filter_candidates

    def _filter_wrapper(state, raw, pp, tp, budget):
        budgets_seen.append(budget)
        return original_filter(state, raw, pp, tp, budget)

    original_safe = watchdog.safe_decide

    def _safe_wrapper(layers, state, budget, telemetry):
        budgets_seen.append(budget)
        return original_safe(layers, state, budget, telemetry)

    monkeypatch.setattr(candidates, "enumerate_candidates", _enum_wrapper)
    monkeypatch.setattr(candidates, "filter_candidates", _filter_wrapper)
    monkeypatch.setattr(watchdog, "safe_decide", _safe_wrapper)
    try:
        from agents.heuristic import agent as agent_module
        monkeypatch.setattr(agent_module, "enumerate_candidates", _enum_wrapper, raising=False)
        monkeypatch.setattr(agent_module, "filter_candidates", _filter_wrapper, raising=False)
        monkeypatch.setattr(agent_module, "safe_decide", _safe_wrapper, raising=False)
    except ImportError:
        pass

    agent.policy(observation)

    assert len(budgets_seen) == 3
    assert all(b is budgets_seen[0] for b in budgets_seen)


# --- AGENT-001..005/009（III、既存骨格が既に満たす契約） -------------------------------------


def test_agent_001_policy_returns_exactly_four_keys():
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    result = agent.policy(observation)
    assert set(result.keys()) == {"item_idx", "container_idx", "place_pos", "orientation"}


def test_agent_002_four_key_types_and_shape():
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    result = agent.policy(observation)
    assert type(result["item_idx"]) is int
    assert type(result["container_idx"]) is int
    assert type(result["orientation"]) is int
    assert isinstance(result["place_pos"], np.ndarray)
    assert result["place_pos"].dtype == np.float32
    assert result["place_pos"].shape == (3,)


def test_agent_003_does_not_mutate_observation():
    import copy

    init, observation = _placeable_fixture()
    observation_before = copy.deepcopy(observation)
    agent = _make_agent(init)
    agent.policy(observation)
    assert _deep_equal(observation, observation_before)


def test_agent_004_does_not_write_files(monkeypatch):
    import builtins

    write_calls = []
    original_open = builtins.open

    def _spy_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x")):
            write_calls.append((file, mode))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _spy_open)

    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    assert write_calls == []


def test_agent_005_optimize_returns_full_index_permutation_non_destructive():
    from agents.heuristic.agent import Agent

    agent = Agent(module_path="agents/heuristic")
    item_list = [_item_dict(index=10, size=(0.1, 0.1, 0.1), mass=1.0),
                 _item_dict(index=20, size=(0.2, 0.1, 0.1), mass=2.0),
                 _item_dict(index=30, size=(0.1, 0.2, 0.1), mass=3.0)]
    item_list_before = [dict(d) for d in item_list]

    result = agent.optimize(item_list)

    assert sorted(result) == [10, 20, 30]
    assert len(result) == 3
    assert all(type(v) is int for v in result)
    assert item_list == item_list_before  # 非破壊


def test_agent_009_policy_is_deterministic_no_randomness():
    init1, observation1 = _placeable_fixture()
    init2, observation2 = _placeable_fixture()
    agent1 = _make_agent(init1)
    agent2 = _make_agent(init2)

    result1 = agent1.policy(observation1)
    result2 = agent2.policy(observation2)
    assert _deep_equal(result1, result2)


# --- AGENT-006..008（II、T-023固定placeholderと矛盾） -----------------------------------------


def test_agent_006_real_candidate_action_not_fixed_placeholder():
    """container0はどの荷物も入らずcontainer1のみ入る fixture で、実際の候補選択が
    行われていれば container_idx は必ず1になる（T-023固定placeholderのcontainer_idx=0とは
    一致しない）。"""
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    result = agent.policy(observation)

    assert result["container_idx"] == 1


def test_agent_007_decided_layer_is_one_of_four_for_normal_case(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    assert captured["telemetry_live"]["decided_layer"] in (1, 2, 3, 4)


def test_agent_008_q5_observes_real_candidate_pools_via_partial_keywords(monkeypatch):
    from src.packing_core.risk import provisional_p_ng
    from src.packing_core.score import heuristic_score
    from src.packing_core.stability import cg_margin, support_ratio
    from src.packing_core.state import build_state
    from src.packing_core import constants

    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    pools = _observed_pools(captured)
    state = build_state(observation, init)  # 同一入力から独立再構築（オラクル計算用）
    rp = constants.ProvisionalRiskParams()
    sp = constants.ScoreParams()

    assert len(pools.dims_candidates) >= 1
    for cand in pools.dims_candidates:
        sr = support_ratio(state, cand)
        margin = cg_margin(state, cand)
        p_ng = provisional_p_ng(support_ratio=sr, cg_margin=margin, params=rp)
        expected_p_success = float(np.clip(1.0 - p_ng, 0.0, 1.0))

        assert cand.p_success == pytest.approx(expected_p_success)
        assert 0.0 <= cand.p_success <= 1.0
        assert np.isfinite(cand.p_success)

        assert cand.features.get("support_ratio") == pytest.approx(float(sr))
        assert cand.features.get("cg_margin") == pytest.approx(float(margin))
        assert cand.features.get("provisional_p_ng") == pytest.approx(float(p_ng))

        # §4.12「Agent配線契約」: scoreはdims_candidatesの各候補へ設定される
        # （geo_candidatesに限らない）。
        expected_score = heuristic_score(state, cand, sp)
        assert cand.score == pytest.approx(expected_score)

    telemetry = captured["telemetry_live"]
    assert telemetry["n_cand0"] == len(pools.raw_candidates)
    assert telemetry["n_after_dims"] == len(pools.dims_candidates)
    assert telemetry["n_after_geo"] == len(pools.geo_candidates)
    assert telemetry["n_lpath_pass"] == len(pools.path_candidates)


# --- AGENT-010..018（I、新規配線契約） ----------------------------------------------------------


def test_agent_010_telemetry_has_exactly_seven_keys(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    expected_keys = {
        "n_cand0", "n_after_dims", "n_after_geo", "n_lpath_pass",
        "reject_reason_counts", "decided_layer", "layer_error",
    }
    assert set(captured["telemetry_live"].keys()) == expected_keys


def test_agent_011_telemetry_created_before_safe_decide_with_n_lpath_pass_zero(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    assert captured["telemetry_before"]["n_lpath_pass"] == 0


def test_agent_012_n_lpath_pass_updated_after_safe_decide(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    pools = _observed_pools(captured)
    assert captured["telemetry_live"]["n_lpath_pass"] == len(pools.path_candidates)


def test_agent_013_candidate_counts_match_pools(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    pools = _observed_pools(captured)
    tb = captured["telemetry_before"]
    assert tb["n_cand0"] == len(pools.raw_candidates)
    assert tb["n_after_dims"] == len(pools.dims_candidates)
    assert tb["n_after_geo"] == len(pools.geo_candidates)


def test_agent_014_reject_reason_counts_matches_pools_reject_counts(monkeypatch):
    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    pools = _observed_pools(captured)
    assert captured["telemetry_before"]["reject_reason_counts"] == dict(pools.reject_counts)


def test_agent_015_telemetry_not_accumulated_across_calls(monkeypatch):
    captured_1 = _spy_safe_decide_passthrough(monkeypatch)
    init1, observation1 = _placeable_fixture()
    agent1 = _make_agent(init1)
    agent1.policy(observation1)
    telemetry_1 = captured_1["telemetry_live"]
    layer_error_len_1 = len(telemetry_1["layer_error"])

    captured_2 = _spy_safe_decide_passthrough(monkeypatch)
    init2, observation2 = _placeable_fixture()
    agent2 = _make_agent(init2)
    agent2.policy(observation2)
    telemetry_2 = captured_2["telemetry_live"]

    assert telemetry_1 is not telemetry_2  # 呼出しごとに新規辞書
    assert len(telemetry_2["layer_error"]) == layer_error_len_1  # 前回分が積み上がらない


def test_agent_016_make_action_called_with_real_candidate_not_fixed_placeholder(monkeypatch):
    from src.packing_core import state as state_module

    calls = []
    original_make_action = state_module.make_action

    def _spy_make_action(item_idx, container_idx, pos_rel, orientation):
        calls.append({"item_idx": item_idx, "container_idx": container_idx, "orientation": orientation})
        return original_make_action(item_idx, container_idx, pos_rel, orientation)

    monkeypatch.setattr(state_module, "make_action", _spy_make_action)
    try:
        from agents.heuristic import agent as agent_module
        monkeypatch.setattr(agent_module, "make_action", _spy_make_action, raising=False)
    except ImportError:
        pass

    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    assert len(calls) >= 1
    assert calls[-1]["container_idx"] == 1  # container1のみ入るfixtureのため実候補ならこの値


def test_agent_017_calls_provisional_p_ng_not_reimplemented(monkeypatch):
    from src.packing_core import risk as risk_module

    calls = []
    original = risk_module.provisional_p_ng

    def _spy(support_ratio, cg_margin, params):
        calls.append((support_ratio, cg_margin, params))
        return original(support_ratio=support_ratio, cg_margin=cg_margin, params=params)

    monkeypatch.setattr(risk_module, "provisional_p_ng", _spy)
    try:
        from agents.heuristic import agent as agent_module
        monkeypatch.setattr(agent_module, "provisional_p_ng", _spy, raising=False)
    except ImportError:
        pass

    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    assert len(calls) >= 1


def test_agent_018_agent_assigns_score_and_features_not_heuristic_score_itself(monkeypatch):
    """cand.score/cand.featuresへの代入はagent.py側の責務であり、heuristic_score自身は
    Candidateを変更しない（§4.8「Candidate/PackingState/ItemSpecを変更しない」契約）。
    policyが設定した後の値を保持しつつ、heuristic_scoreを異なるScoreParamsで再度直接
    呼んでも cand.score が変化しない（=heuristic_score自体はcand.scoreへ代入しない）
    ことで、代入がagent.py側の明示的な配線であることを確認する。"""
    from src.packing_core import constants
    from src.packing_core.score import heuristic_score
    from src.packing_core.state import build_state

    captured = _spy_safe_decide_passthrough(monkeypatch)
    init, observation = _placeable_fixture()
    agent = _make_agent(init)
    agent.policy(observation)

    pools = _observed_pools(captured)
    assert len(pools.dims_candidates) >= 1
    for cand in pools.dims_candidates:
        assert {"support_ratio", "cg_margin", "provisional_p_ng"} <= set(cand.features.keys())

    # scoreはdims_candidatesの各候補へ設定される契約（§4.12、geo_candidatesに限らない）。
    cand = pools.dims_candidates[0]
    score_after_policy = cand.score

    state = build_state(observation, init)
    different_sp = constants.ScoreParams(w_z=99.0, w_y=99.0, w_x=99.0)
    heuristic_score(state, cand, different_sp)  # 戻り値は使わず、副作用の有無だけを見る

    assert cand.score == pytest.approx(score_after_policy)  # heuristic_score呼出しでは変化しない
