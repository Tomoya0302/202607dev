"""T-027: Phase 3暫定optimize契約テスト（詳細仕様書 v1.23 §4.12）。"""
from __future__ import annotations

import copy

import numpy as np
import pytest


def _item(index, size=(1.0, 1.0, 1.0), mass=1.0):
    return {
        "index": index,
        "length": size[0],
        "width": size[1],
        "height": size[2],
        "mass": mass,
    }


@pytest.fixture
def agent():
    from agents.heuristic.agent import Agent

    return Agent(module_path="agents/heuristic")


def test_opt_001_empty_and_single_item(agent):
    assert agent.optimize([]) == []
    assert agent.optimize([_item(42)]) == [42]


def test_opt_002_003_normal_sort_returns_non_contiguous_official_indices(agent):
    items = [
        _item(10, size=(1.0, 1.0, 1.0), mass=9.0),
        _item(20, size=(2.0, 1.0, 1.0), mass=2.0),
        _item(30, size=(1.0, 2.0, 1.0), mass=3.0),
    ]

    assert agent.optimize(items) == [30, 20, 10]


def test_opt_004_equal_volume_and_mass_uses_input_position_tiebreak(agent):
    items = [_item(30), _item(10), _item(20)]

    assert agent.optimize(items) == [30, 10, 20]


def test_opt_005_different_dimensions_with_equal_volume_are_tied(agent):
    items = [
        _item(10, size=(1.0, 2.0, 3.0), mass=4.0),
        _item(20, size=(1.0, 1.0, 6.0), mass=4.0),
    ]

    assert agent.optimize(items) == [10, 20]


def test_opt_006_accepts_builtin_and_numpy_numeric_scalars(agent):
    items = [
        _item(np.int64(10), size=(np.int64(1), np.float32(1.0), 1.0), mass=np.float64(1.0)),
        _item(np.int32(20), size=(2, 1.0, np.float64(1.0)), mass=np.int64(2)),
    ]

    result = agent.optimize(items)

    assert result == [20, 10]
    assert all(type(value) is int for value in result)


def test_opt_007_zero_mass_is_valid(agent):
    items = [_item(10, mass=0), _item(20, mass=1)]

    assert agent.optimize(items) == [20, 10]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("length", None),
        ("width", "1.0"),
        ("height", True),
        ("length", float("nan")),
        ("width", float("inf")),
        ("height", float("-inf")),
        ("length", 0.0),
        ("width", -1.0),
        ("height", 0),
        ("mass", -1.0),
    ],
)
def test_opt_008_012_invalid_sort_material_falls_back_to_official_input_order(
    agent, field, value,
):
    items = [_item(10, size=(1.0, 1.0, 1.0)), _item(20, size=(2.0, 2.0, 2.0))]
    items[1][field] = value

    assert agent.optimize(items) == [10, 20]


def test_opt_011_missing_sort_material_falls_back_to_official_input_order(agent):
    items = [_item(10), _item(20, size=(2.0, 2.0, 2.0))]
    del items[1]["mass"]

    assert agent.optimize(items) == [10, 20]


def test_opt_011_nonfinite_volume_falls_back_to_official_input_order(agent):
    items = [_item(10), _item(20, size=(1e308, 1e308, 1e308))]

    assert agent.optimize(items) == [10, 20]


@pytest.mark.parametrize("bad_index", [None, 1.0, True, "10"])
def test_opt_013_invalid_index_type_raises_value_error(agent, bad_index):
    items = [_item(10), _item(bad_index)]

    with pytest.raises(ValueError, match="index"):
        agent.optimize(items)


def test_opt_013_missing_index_raises_value_error(agent):
    items = [_item(10), _item(20)]
    del items[1]["index"]

    with pytest.raises(ValueError, match="index"):
        agent.optimize(items)


def test_opt_014_duplicate_index_raises_value_error(agent):
    with pytest.raises(ValueError, match="index"):
        agent.optimize([_item(10), _item(np.int64(10))])


def test_opt_016_does_not_mutate_input(agent):
    items = [
        _item(np.int64(10), size=(np.float32(1.0), 1, 1.0), mass=np.float64(2.0)),
        _item(np.int64(20), size=(2.0, 1.0, 1.0), mass=np.float32(1.0)),
    ]
    before = copy.deepcopy(items)

    agent.optimize(items)

    assert items == before


def test_opt_017_does_not_call_policy_pipeline(agent, monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("optimize must not call the policy pipeline")

    monkeypatch.setattr(agent, "_run_pipeline", _unexpected)

    assert agent.optimize([_item(10), _item(20, size=(2.0, 1.0, 1.0))]) == [20, 10]


def test_opt_018_large_input_returns_complete_official_index_permutation(agent):
    items = [_item(1000 + i * 3, size=(1.0 + (i % 7), 1.0, 1.0), mass=i % 5) for i in range(500)]

    result = agent.optimize(items)

    assert len(result) == len(items)
    assert set(result) == {int(item["index"]) for item in items}
    assert all(type(value) is int for value in result)
