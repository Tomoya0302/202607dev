"""T-001: packing_core.constants / packing_core.types の仕様値検証（§3.3・付録C）。"""
import dataclasses

import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.types import Candidate, EMSBox, ItemSpec, PlacedItem


# --- constants.py -----------------------------------------------------------

def test_eps_and_tol_values():
    assert constants.EPS_GEOM == 1e-9
    assert constants.TOL_CONTACT == 5e-3


def test_placement_params_defaults():
    pp = constants.PlacementParams()
    assert pp.inclusion_margin == -0.005
    assert pp.safety_margin == 0.015
    assert pp.start_z == 0.08
    assert pp.ceiling_margin == 0.018
    assert pp.internal_extra == 0.005


def test_time_params_defaults():
    tp = constants.TimeParams()
    assert tp.policy_soft == 6.5
    assert tp.policy_hard == 7.0
    assert tp.optimize_stop == 170.0
    assert tp.budget_poll_every == 64


def test_grid_params_defaults():
    assert constants.GridParams().cell == 0.02


def test_stage_params_defaults():
    assert constants.StageParams().l_path_top_m == 64


def test_score_params_defaults():
    sp = constants.ScoreParams()
    assert sp.w_z == 1.0
    assert sp.w_y == 0.3
    assert sp.w_x == 0.1
    assert sp.w_support == 0.5
    assert sp.w_cg_h == 0.3
    assert sp.w_soft == 0.8
    assert sp.w_prio == 0.4


def test_regime_params_defaults():
    rp = constants.RegimeParams()
    assert rp.regime_threshold_n == -1
    assert rp.gamma_cons == 4.0
    assert rp.gamma_aggr == 1.0


def test_todo_placeholders():
    assert constants.KIND_IS_SOFT == {}
    assert constants.POOL_MAX_WEIGHT == -1.0
    assert constants.OBS_KEYS == {}


def test_placement_params_is_frozen():
    pp = constants.PlacementParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        pp.safety_margin = 0.02


def test_assumptions_key_set_and_initial_status():
    # §3.6 / 付録C: 仮定台帳のキーはちょうど A9..A14 で、初期状態はすべて unconfirmed。
    assert set(constants.ASSUMPTIONS.keys()) == {"A9", "A10", "A11", "A12", "A13", "A14"}
    for aid, entry in constants.ASSUMPTIONS.items():
        assert set(entry.keys()) == {"claim", "status", "ref"}
        assert entry["status"] == "unconfirmed", f"{aid} は初期状態で unconfirmed であるべき"


def test_assert_confirmed_no_args_does_not_raise():
    constants.assert_confirmed()


def test_assert_confirmed_raises_for_unconfirmed_assumption():
    with pytest.raises(RuntimeError):
        constants.assert_confirmed("A9")


def test_assert_confirmed_raises_for_unknown_id():
    with pytest.raises(RuntimeError):
        constants.assert_confirmed("A999")


# --- types.py ----------------------------------------------------------------

def test_item_spec_fields_and_frozen():
    item = ItemSpec(
        idx=3,
        size=np.array([0.5, 0.4, 0.3], dtype=np.float64),
        weight=12.5,
        kind="fragile",
        is_soft=False,
        is_priority=True,
    )
    assert item.idx == 3
    assert np.array_equal(item.size, np.array([0.5, 0.4, 0.3]))
    assert item.weight == 12.5
    assert item.kind == "fragile"
    assert item.is_soft is False
    assert item.is_priority is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.weight = 99.0


def test_placed_item_fields_and_frozen():
    placed = PlacedItem(
        pos_world=np.array([1.0, 2.0, 0.5], dtype=np.float64),
        orn_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        size=np.array([0.5, 0.4, 0.3], dtype=np.float64),
        weight=10.0,
        is_soft=True,
        is_priority=False,
        aabb_min_rel=np.array([0.75, 1.8, 0.35], dtype=np.float64),
        aabb_max_rel=np.array([1.25, 2.2, 0.65], dtype=np.float64),
    )
    assert np.array_equal(placed.pos_world, np.array([1.0, 2.0, 0.5]))
    assert np.array_equal(placed.orn_quat, np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(placed.aabb_min_rel, np.array([0.75, 1.8, 0.35]))
    assert np.array_equal(placed.aabb_max_rel, np.array([1.25, 2.2, 0.65]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        placed.weight = 0.0


def test_ems_box_size_and_volume():
    box = EMSBox(
        min_rel=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        max_rel=np.array([0.5, 0.4, 0.3], dtype=np.float64),
    )
    assert np.allclose(box.size(), np.array([0.5, 0.4, 0.3]))
    assert box.volume() == pytest.approx(0.06)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.max_rel = np.array([1.0, 1.0, 1.0])


def test_candidate_defaults_and_mutability():
    cand = Candidate(
        item_idx=0,
        container_idx=1,
        ems_id=2,
        orientation=0,
        pos_rel=np.array([0.25, 0.2, 0.15], dtype=np.float64),
        osize=np.array([0.5, 0.4, 0.3], dtype=np.float64),
    )
    assert cand.feasible is False
    assert cand.reject_reason == ""
    assert cand.features == {}
    assert cand.score == 0.0
    assert cand.p_success == 1.0

    # Candidate は通常 dataclass（可変）。段階フィルタが in-place で更新できる。
    cand.feasible = True
    cand.reject_reason = "dims"
    assert cand.feasible is True
    assert cand.reject_reason == "dims"


def test_candidate_features_default_factory_is_independent_per_instance():
    common_kwargs = dict(
        item_idx=0,
        container_idx=0,
        ems_id=0,
        orientation=0,
        pos_rel=np.zeros(3, dtype=np.float64),
        osize=np.ones(3, dtype=np.float64),
    )
    cand_a = Candidate(**common_kwargs)
    cand_b = Candidate(**common_kwargs)
    cand_a.features["support_ratio"] = 0.9
    assert cand_b.features == {}
