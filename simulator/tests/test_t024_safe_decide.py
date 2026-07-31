"""統合T-024: watchdog.safe_decide の契約テスト（詳細仕様書 v1.18 §4.11「safe_decide」
「safe_decide telemetry契約」、契約計画 §3.I）。

`safe_decide` は未実装のため対象関数の import は各テスト関数内で行う。全16家族が分類 I。

シグネチャ（§4.11）:
    def safe_decide(layers: Sequence[Layer], state, budget, telemetry: dict) -> Candidate | None
    Layer = Callable[[state, budget], Candidate | None]

telemetry契約: 呼出し開始時に telemetry.setdefault("decided_layer", 0)／
    telemetry.setdefault("layer_error", [])。各層で例外を捕捉した場合
    telemetry["layer_error"].append({"layer": layer_index, "error_type": type(exc).__name__})
    （layer_indexは1〜4。メッセージ・トレースバック全文は格納しない）。Candidateが返れば即
    telemetry["decided_layer"]=layer_index として確定。全層None・全層例外ならdecided_layerは
    初期値のまま。

**訂正（SAFE-014）**: 旧「safe_decide開始前hard超過ならNone」はv1.18に明記契約が無いため削除。
    telemetry事前値のsetdefault非上書き契約へ置換。hard期限の挙動はlayer1/layer2側
    （test_t024_layers.py LAYER-006..014／LAYER-015..019）で検証し、本ファイルでは扱わない。
"""
import functools

import numpy as np
import pytest

from agents.heuristic.packing_core.types import Candidate


def _cand(tag=0):
    c = Candidate(
        item_idx=tag, container_idx=0, ems_id=0, orientation=0,
        pos_rel=np.array([0.0, 0.0, 0.5], dtype=np.float64),
        osize=np.array([0.1, 0.1, 0.1], dtype=np.float64),
    )
    return c


def _layer_returns(value, calls=None, name=None):
    def _fn(state, budget):
        if calls is not None:
            calls.append(name)
        return value
    return _fn


def _layer_raises(exc, calls=None, name=None):
    def _fn(state, budget):
        if calls is not None:
            calls.append(name)
        raise exc
    return _fn


def _layer_records_args(recorded, calls=None, name=None, ret=None):
    def _fn(state, budget):
        recorded.append((state, budget))
        if calls is not None:
            calls.append(name)
        return ret
    return _fn


def _dummy_state():
    return object()


def _dummy_budget():
    from agents.heuristic.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)


# --- SAFE-001: 層順1→2→3→4 -------------------------------------------------------------------


def test_safe_001_layers_called_in_order_1_to_4():
    from agents.heuristic.packing_core.watchdog import safe_decide

    calls = []
    layers = [
        _layer_returns(None, calls, 1),
        _layer_returns(None, calls, 2),
        _layer_returns(None, calls, 3),
        _layer_returns(None, calls, 4),
    ]
    telemetry = {}
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    assert calls == [1, 2, 3, 4]
    assert result is None


# --- SAFE-002: k層目が最初にCandidateを返すと後続層を呼ばない（param×4） ---------------------


@pytest.mark.parametrize(
    "k",
    [
        pytest.param(1, id="T024-SAFE-002-L1"),
        pytest.param(2, id="T024-SAFE-002-L2"),
        pytest.param(3, id="T024-SAFE-002-L3"),
        pytest.param(4, id="T024-SAFE-002-L4"),
    ],
)
def test_safe_002_early_confirmation_skips_remaining_layers(k):
    from agents.heuristic.packing_core.watchdog import safe_decide

    calls = []
    winner = _cand(tag=99)
    layers = []
    for i in range(1, 5):
        if i < k:
            layers.append(_layer_returns(None, calls, i))
        elif i == k:
            layers.append(_layer_returns(winner, calls, i))
        else:
            layers.append(_layer_returns(_cand(tag=-1), calls, i))  # 呼ばれてはいけない

    telemetry = {}
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    assert result is winner
    assert calls == list(range(1, k + 1))  # k層目までしか呼ばれない


# --- SAFE-003: 全層Noneなら None -----------------------------------------------------------


def test_safe_003_all_none_returns_none():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [_layer_returns(None) for _ in range(4)]
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})
    assert result is None


# --- SAFE-004: 1層がraiseしても次層へ進み次層の解を返す ---------------------------------------


def test_safe_004_exception_in_one_layer_does_not_stop_next_layer():
    from agents.heuristic.packing_core.watchdog import safe_decide

    winner = _cand(tag=7)
    layers = [
        _layer_raises(ValueError("boom")),
        _layer_returns(winner),
        _layer_returns(_cand(tag=-1)),  # 呼ばれてはいけない
        _layer_returns(_cand(tag=-1)),
    ]
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})
    assert result is winner


# --- SAFE-005: 全層raiseならNone -----------------------------------------------------------


def test_safe_005_all_layers_raise_returns_none():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [_layer_raises(RuntimeError(f"boom{i}")) for i in range(4)]
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})
    assert result is None


# --- SAFE-006: 例外がsafe_decide外へ漏れない -------------------------------------------------


def test_safe_006_exceptions_do_not_propagate_out_of_safe_decide():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [
        _layer_raises(ValueError("a")),
        _layer_raises(TypeError("b")),
        _layer_raises(KeyError("c")),
        _layer_returns(None),
    ]
    try:
        result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})
    except Exception as exc:  # noqa: BLE001（本テストの主張そのもの: 例外が漏れないこと）
        pytest.fail(f"safe_decide leaked an exception: {exc!r}")
    assert result is None


# --- SAFE-007: decided_layer（param×5） -------------------------------------------------------


@pytest.mark.parametrize(
    "k,expected",
    [
        pytest.param(1, 1, id="T024-SAFE-007-L1"),
        pytest.param(2, 2, id="T024-SAFE-007-L2"),
        pytest.param(3, 3, id="T024-SAFE-007-L3"),
        pytest.param(4, 4, id="T024-SAFE-007-L4"),
        pytest.param(None, 0, id="T024-SAFE-007-NONE"),
    ],
)
def test_safe_007_decided_layer_telemetry(k, expected):
    from agents.heuristic.packing_core.watchdog import safe_decide

    if k is None:
        layers = [_layer_returns(None) for _ in range(4)]
    else:
        layers = [
            _layer_returns(_cand(tag=i) if i == k else None) for i in range(1, 5)
        ]

    telemetry = {}
    safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)
    assert telemetry["decided_layer"] == expected


# --- SAFE-008: 開始時のsetdefault初期化 -------------------------------------------------------


def test_safe_008_initializes_telemetry_keys_via_setdefault():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [_layer_returns(None) for _ in range(4)]
    telemetry = {}
    safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    assert telemetry.get("decided_layer") == 0
    assert telemetry.get("layer_error") == []


# --- SAFE-009: 例外捕捉時のlayer_error形状（message/tracebackを保存しない） -------------------


def test_safe_009_layer_error_shape_excludes_message_and_traceback():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [
        _layer_returns(None),
        _layer_raises(ValueError("some sensitive detail")),
        _layer_returns(None),
        _layer_returns(None),
    ]
    telemetry = {}
    safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    assert len(telemetry["layer_error"]) == 1
    entry = telemetry["layer_error"][0]
    assert entry == {"layer": 2, "error_type": "ValueError"}
    assert set(entry.keys()) == {"layer", "error_type"}  # message/tracebackを含まない


# --- SAFE-010: 全層へ同一stateを渡す ----------------------------------------------------------


def test_safe_010_all_layers_receive_the_same_state_object():
    from agents.heuristic.packing_core.watchdog import safe_decide

    recorded = []
    layers = [_layer_records_args(recorded, ret=None) for _ in range(4)]
    state = _dummy_state()
    safe_decide(layers, state, _dummy_budget(), {})

    assert len(recorded) == 4
    assert all(s is state for s, _ in recorded)


# --- SAFE-011: 全層へ同一budgetを渡す ---------------------------------------------------------


def test_safe_011_all_layers_receive_the_same_budget_object():
    from agents.heuristic.packing_core.watchdog import safe_decide

    recorded = []
    layers = [_layer_records_args(recorded, ret=None) for _ in range(4)]
    budget = _dummy_budget()
    safe_decide(layers, _dummy_state(), budget, {})

    assert len(recorded) == 4
    assert all(b is budget for _, b in recorded)


# --- SAFE-012: make_actionを呼ばない ----------------------------------------------------------


def test_safe_012_does_not_call_make_action(monkeypatch):
    from agents.heuristic.packing_core import state as state_module
    from agents.heuristic.packing_core import watchdog
    from agents.heuristic.packing_core.watchdog import safe_decide

    calls = []
    monkeypatch.setattr(state_module, "make_action", lambda *a, **k: calls.append(1), raising=False)
    monkeypatch.setattr(watchdog, "make_action", lambda *a, **k: calls.append(1), raising=False)

    layers = [_layer_returns(_cand(tag=1) if i == 0 else None) for i in range(4)]
    safe_decide(layers, _dummy_state(), _dummy_budget(), {})

    assert calls == []


# --- SAFE-013: 返すCandidateを変更しない -------------------------------------------------------


def test_safe_013_does_not_mutate_returned_candidate():
    from agents.heuristic.packing_core.watchdog import safe_decide

    winner = _cand(tag=42)
    winner.score = 0.75
    pos_before = winner.pos_rel.copy()
    osize_before = winner.osize.copy()

    layers = [_layer_returns(winner if i == 1 else None) for i in range(4)]
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})

    assert result is winner
    assert result.score == 0.75
    np.testing.assert_array_equal(result.pos_rel, pos_before)
    np.testing.assert_array_equal(result.osize, osize_before)


# --- SAFE-014（訂正）: telemetry事前値をsetdefaultが上書きしない --------------------------------


def test_safe_014_preexisting_telemetry_values_not_overwritten():
    from agents.heuristic.packing_core.watchdog import safe_decide

    layers = [_layer_returns(None) for _ in range(4)]  # 全層None（確定しない）
    telemetry = {"decided_layer": 5, "layer_error": ["preexisting"]}
    safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    # setdefaultのみを使うため、既存キーの値はそのまま残る（0/[]へ上書きされない）。
    assert telemetry["decided_layer"] == 5
    assert telemetry["layer_error"] == ["preexisting"]


# --- SAFE-015: layers=functools.partial列、束縛順非依存で受理 -----------------------------------


def test_safe_015_accepts_functools_partial_layers_regardless_of_binding_order():
    from agents.heuristic.packing_core.watchdog import safe_decide

    def _base(state, budget, *, tag, extra=None):
        return _cand(tag=tag)

    layer_a = functools.partial(_base, tag=1, extra="x")   # kwargsの順序A
    layer_b = functools.partial(_base, extra="y", tag=2)   # kwargsの順序B（束縛順が違う）
    layers = [
        functools.partial(lambda state, budget: None),
        layer_b,
        layer_a,
        functools.partial(lambda state, budget: None),
    ]
    # layer_bが2番目（layer2）で確定するはず。
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), {})
    assert result is not None
    assert result.item_idx == 2


# --- SAFE-016: 早期確定後は後続層分をlayer_errorへ積まない --------------------------------------


def test_safe_016_no_layer_error_entries_for_layers_after_early_confirmation():
    from agents.heuristic.packing_core.watchdog import safe_decide

    winner = _cand(tag=1)
    layers = [
        _layer_returns(winner),  # layer1で即確定
        _layer_raises(ValueError("would raise if called")),
        _layer_raises(TypeError("would raise if called")),
        _layer_raises(KeyError("would raise if called")),
    ]
    telemetry = {}
    result = safe_decide(layers, _dummy_state(), _dummy_budget(), telemetry)

    assert result is winner
    assert telemetry["layer_error"] == []  # 後続層は呼ばれないため例外記録も無い
