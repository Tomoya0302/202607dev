"""統合T-024: 最外殻emergency actionの契約テスト（詳細仕様書 v1.18 §4.11「最外殻emergency
action」、契約計画 §3.L）。

10家族（8トリガー+2）。分類: II=8（EMG-001..007,010、T-023固定placeholderと矛盾）、
III=2（EMG-008,009、既存骨格でも成立する外形契約）。

emergency発火条件（いずれか）: (a) build_state例外 (b) raw_candidates=0 (c) dims_candidates=0
(d) safe_decideがNoneを返した (e) 全層で例外 (f) Candidate→action変換時の例外。

emergency actionの内容: 正常なobservationからpool_list[0]["index"]を読める場合は
    item_idx=int(observation["pool_list"][0]["index"]), container_idx=0,
    pos_rel=(0.0,0.0,0.5), orientation=0
プールindexすら取得できない異常入力の場合に限り、T-023の固定プレースホルダー item_idx=0。
"""
import numpy as np
import pytest

INNER_MIN_SMALL = np.array([-0.02, -0.02, 0.02], dtype=np.float64)
INNER_MAX_SMALL = np.array([0.02, 0.02, 0.05], dtype=np.float64)
INNER_MIN_BIG = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_BIG = np.array([0.40, 0.40, 1.02], dtype=np.float64)


def _item_dict(index, size, mass=1.0):
    length, width, height = (float(v) for v in size)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "mass": float(mass), "is_prioritized": False, "is_soft": False,
        "belongs_to": None, "pos": None, "orn": None,
        "lateralFriction": 0.5, "rollingFriction": 0.01, "spinningFriction": 0.01,
        "restitution": 0.1, "angularDamping": 0.05,
    }


def _container_dict(index, offset_x, inner_min_rel, inner_max_rel, thickness=0.02, buffer=0.02):
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
        "shelf": False, "is_prioritized": False, "packed_items": [],
    }


def _normal_fixture(pool_index=7):
    """健全な1コンテナ+1アイテム（container0のみ、通常サイズ）。pool_list[0]["index"]は
    引数で指定可能（EMG-007のpool読み取り検証用）。"""
    container = _container_dict(0, 0.0, INNER_MIN_BIG, INNER_MAX_BIG)
    pool_list = [_item_dict(index=pool_index, size=(0.1, 0.1, 0.1), mass=2.0)]
    init = {"optimize": True, "lookahead_k": 5, "container_list": [container]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((1, 64, 64), dtype=np.float32),
        "container_list": [container], "pool_list": pool_list,
    }
    return init, observation


def _oversized_item_fixture(pool_index=7):
    """コンテナに対し明確に大きすぎるアイテム1件（raw_candidates=0を誘発、pool自体は読める）。"""
    container = _container_dict(0, 0.0, INNER_MIN_BIG, INNER_MAX_BIG)
    pool_list = [_item_dict(index=pool_index, size=(10.0, 10.0, 10.0), mass=2.0)]
    init = {"optimize": True, "lookahead_k": 5, "container_list": [container]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((1, 64, 64), dtype=np.float32),
        "container_list": [container], "pool_list": pool_list,
    }
    return init, observation


def _empty_pool_fixture():
    """pool_listが空（プールindex読取不可＝item_idx=0フォールバックを誘発）。"""
    container = _container_dict(0, 0.0, INNER_MIN_BIG, INNER_MAX_BIG)
    init = {"optimize": True, "lookahead_k": 5, "container_list": [container]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((1, 64, 64), dtype=np.float32),
        "container_list": [container], "pool_list": [],
    }
    return init, observation


def _two_container_fixture():
    """container0=極小（不適合）、container1=十分。健全入力での「emergency未使用」検証用
    （EMG-009。実決定ならcontainer_idx=1、emergencyの固定値0とは異なる）。"""
    c0 = _container_dict(0, 0.0, INNER_MIN_SMALL, INNER_MAX_SMALL)
    c1 = _container_dict(1, 2.0, INNER_MIN_BIG, INNER_MAX_BIG)
    pool_list = [_item_dict(index=0, size=(0.1, 0.1, 0.1), mass=2.0)]
    init = {"optimize": True, "lookahead_k": 5, "container_list": [c0, c1]}
    observation = {
        "optimize": True, "lookahead_k": 5,
        "depth_map": np.zeros((2, 64, 64), dtype=np.float32),
        "container_list": [c0, c1], "pool_list": pool_list,
    }
    return init, observation


def _make_agent(init):
    from agents.heuristic.agent import Agent
    agent = Agent(module_path="agents/heuristic")
    agent.get_init_states(init)
    return agent


def _assert_is_emergency_action_with_pool_index(action, expected_item_idx):
    assert action["item_idx"] == expected_item_idx
    assert action["container_idx"] == 0
    assert action["orientation"] == 0
    np.testing.assert_allclose(np.asarray(action["place_pos"], dtype=np.float64), [0.0, 0.0, 0.5])


def _patch_dual(monkeypatch, source_module, agent_module_name, attr, fn):
    """source_module／agent_module双方の同名属性へpatchする（import文の形式に非依存）。

    対象シンボルがまだ存在しない場合（未実装の新規関数・メソッド）でも
    collection error にならないよう、両方とも raising=False で行う。
    """
    monkeypatch.setattr(source_module, attr, fn, raising=False)
    try:
        import importlib
        agent_module = importlib.import_module(agent_module_name)
        monkeypatch.setattr(agent_module, attr, fn, raising=False)
    except ImportError:
        pass


# --- EMG-001: build_state例外 -----------------------------------------------------------------


def test_emg_001_build_state_exception_triggers_emergency(monkeypatch):
    from src.packing_core import state as state_module

    def _raising_build_state(observation, init):
        raise RuntimeError("injected build_state failure")

    _patch_dual(monkeypatch, state_module, "agents.heuristic.agent", "build_state", _raising_build_state)

    init, observation = _normal_fixture(pool_index=11)
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=11)


# --- EMG-002: raw_candidates=0 -----------------------------------------------------------------


def test_emg_002_zero_raw_candidates_triggers_emergency():
    init, observation = _oversized_item_fixture(pool_index=13)
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=13)


# --- EMG-003: dims_candidates=0 -----------------------------------------------------------------


def test_emg_003_zero_dims_candidates_triggers_emergency(monkeypatch):
    from src.packing_core import candidates as candidates_module

    def _empty_dims_filter_candidates(state, raw_candidates, pp, tp, budget):
        from src.packing_core.candidates import CandidatePools
        return CandidatePools(raw_candidates=raw_candidates, dims_candidates=[], geo_candidates=[])

    _patch_dual(
        monkeypatch, candidates_module, "agents.heuristic.agent",
        "filter_candidates", _empty_dims_filter_candidates,
    )

    init, observation = _normal_fixture(pool_index=17)
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=17)


# --- EMG-004: safe_decideがNoneを返した -----------------------------------------------------------


def test_emg_004_safe_decide_returns_none_triggers_emergency(monkeypatch):
    from src.packing_core import watchdog as watchdog_module

    def _always_none_safe_decide(layers, state, budget, telemetry):
        telemetry.setdefault("decided_layer", 0)
        telemetry.setdefault("layer_error", [])
        return None

    _patch_dual(
        monkeypatch, watchdog_module, "agents.heuristic.agent",
        "safe_decide", _always_none_safe_decide,
    )

    init, observation = _normal_fixture(pool_index=19)
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=19)


# --- EMG-005: 全層で例外 -----------------------------------------------------------------------


def test_emg_005_all_layers_raise_triggers_emergency(monkeypatch):
    """4層すべてが例外を送出しても（real safe_decideが内部捕捉してNoneを返す前提で）
    emergencyへフォールバックする。"""
    from src.packing_core import watchdog as watchdog_module

    def _raiser(*args, **kwargs):
        raise RuntimeError("injected layer failure")

    for name in ("layer1_main", "layer2_dblf_strict", "layer3_first_fit", "layer4_max_p"):
        _patch_dual(monkeypatch, watchdog_module, "agents.heuristic.agent", name, _raiser)

    init, observation = _normal_fixture(pool_index=23)
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=23)


# --- EMG-006: Candidate→action変換時の例外（one-shot: 1回目のみ失敗） -----------------------------


def test_emg_006_make_action_conversion_exception_triggers_emergency_then_succeeds(monkeypatch):
    from src.packing_core import state as state_module

    original_make_action = state_module.make_action
    call_count = {"n": 0}

    def _one_shot_raising_make_action(item_idx, container_idx, pos_rel, orientation):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("injected make_action failure (decided candidate conversion)")
        return original_make_action(item_idx, container_idx, pos_rel, orientation)

    _patch_dual(
        monkeypatch, state_module, "agents.heuristic.agent",
        "make_action", _one_shot_raising_make_action,
    )

    init, observation = _normal_fixture(pool_index=29)
    agent = _make_agent(init)
    result = agent.policy(observation)  # 1回目のmake_action呼出しで例外→emergencyへ

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=29)
    assert call_count["n"] >= 2  # 1回目失敗・2回目（emergency構築）で成功


# --- EMG-007: pool可読時のemergency action内容（公式indexフィールドを読む） -----------------------


def test_emg_007_emergency_action_reads_official_pool_index_field():
    """pool_list[0]["index"]が非ゼロ・非連番の値でも、その公式indexフィールドをそのまま
    使う（python位置0やT-023固定0を使わない）。"""
    init, observation = _oversized_item_fixture(pool_index=999)  # raw_candidates=0で発火
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=999)


# --- EMG-010: pool index不可読時のみT-023固定item_idx=0 ------------------------------------------


def test_emg_010_unreadable_pool_index_falls_back_to_fixed_zero():
    init, observation = _empty_pool_fixture()
    agent = _make_agent(init)
    result = agent.policy(observation)

    _assert_is_emergency_action_with_pool_index(result, expected_item_idx=0)


# --- EMG-008: 各トリガーでpolicy例外がプロセス外へ漏れない（III） --------------------------------


def test_emg_008_exceptions_never_leak_out_of_policy(monkeypatch):
    from src.packing_core import state as state_module
    from src.packing_core import watchdog as watchdog_module

    def _raising_build_state(observation, init):
        raise RuntimeError("injected")

    _patch_dual(monkeypatch, state_module, "agents.heuristic.agent", "build_state", _raising_build_state)
    init, observation = _normal_fixture(pool_index=3)
    agent = _make_agent(init)
    try:
        agent.policy(observation)
    except Exception as exc:  # noqa: BLE001（本テストの主張そのもの）
        pytest.fail(f"policy leaked an exception (build_state trigger): {exc!r}")

    monkeypatch.undo()

    def _raiser(*args, **kwargs):
        raise RuntimeError("injected layer failure")

    for name in ("layer1_main", "layer2_dblf_strict", "layer3_first_fit", "layer4_max_p"):
        _patch_dual(monkeypatch, watchdog_module, "agents.heuristic.agent", name, _raiser)
    init2, observation2 = _normal_fixture(pool_index=5)
    agent2 = _make_agent(init2)
    try:
        agent2.policy(observation2)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"policy leaked an exception (all-layers-raise trigger): {exc!r}")


# --- EMG-009: 健全入力ではemergencyを使わない（III） ---------------------------------------------


def test_emg_009_healthy_input_does_not_use_emergency_action():
    """container0=不適合・container1=適合のfixtureで、実決定ならcontainer_idx=1になる
    （emergencyの固定container_idx=0とは異なる値）。"""
    init, observation = _two_container_fixture()
    agent = _make_agent(init)
    result = agent.policy(observation)

    assert result["container_idx"] == 1
