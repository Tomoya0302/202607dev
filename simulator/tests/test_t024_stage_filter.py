"""統合T-024: candidates.filter_candidates の契約テスト（詳細仕様書 v1.18 §4.12
「候補列挙・候補生成契約」「budget確認位置の固定契約」、契約計画 §3.D）。

`candidates.py` は未実装のため対象関数の import は各テスト関数内で行う。全19家族が分類 I
（`FILT-002` は削除済み——直接呼出し検査ではなく公開動作で短絡・reject結果を検証する）。

短絡評価（DIMS→INCLUSION→OVERLAP→CEILING）は `masks.evaluate_stage`（実装済み）を通じて
`masks.py` の leaf 関数（`prefilter_dims`/`check_inclusion`/`check_overlap`/`check_ceiling`）を
呼ぶ契約であるため、本ファイルの多くのテストは leaf 関数を monkeypatch でスパイ化し、公開動作
（`evaluate_stage` 経由の呼出し回数・結果）だけを観測する（`tests/test_masks.py` の
`_install_stage_spies` と同方針）。

`Candidate` は非frozen dataclass で `pos_rel`/`osize` に numpy 配列を持つため、`==` による
Candidate（や Candidate のリスト）比較は曖昧な真偽値エラーを招く。本ファイルは常に `is`
（オブジェクト identity）で Candidate の同一性を検証する。
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.container_space import build_container_space
from src.packing_core.state import PackingState
from src.packing_core.types import Candidate, EMSBox

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
CELL = 0.02
PP0 = constants.PlacementParams()


def _box_cdict():
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


def _flat_space():
    return build_container_space(_box_cdict(), index=0, cell=CELL)


def _ems(min_rel=(0.0, -0.5, 0.0), max_rel=(1.0, 0.5, 1.0)):
    return EMSBox(
        min_rel=np.array(min_rel, dtype=np.float64), max_rel=np.array(max_rel, dtype=np.float64)
    )


def _cand(pos_rel=(0.0, 0.0, 0.5), osize=(0.1, 0.1, 0.1), container_idx=0, ems_id=0, item_idx=0):
    return Candidate(
        item_idx=item_idx, container_idx=container_idx, ems_id=ems_id, orientation=0,
        pos_rel=np.array(pos_rel, dtype=np.float64), osize=np.array(osize, dtype=np.float64),
    )


def _state(n_placed_items=0):
    space = _flat_space()
    placed = []
    return PackingState(
        containers=[space], placed={0: placed}, pool=[],
        ems={0: [_ems()]}, ems_truncation={0: 0.0}, meta={},
    )


def _budget_never_over():
    from src.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)


class _FlipClock:
    """呼び出し回数が `flip_after_calls` を超えると `after` を返す決定論的fake clock。"""

    def __init__(self, flip_after_calls: int, before: float = 0.0, after: float = 1000.0):
        self._n = 0
        self._flip = flip_after_calls
        self._before = before
        self._after = after

    def __call__(self) -> float:
        self._n += 1
        return self._after if self._n > self._flip else self._before


def _install_leaf_spies(monkeypatch, *, dims=True, inclusion=True, overlap=True, ceiling=True):
    """4段階leaf関数を固定の合否でスパイ化し、呼出し回数を記録する（実ジオメトリ非依存）。"""
    from src.packing_core import masks

    calls = {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0}

    monkeypatch.setattr(masks, "prefilter_dims", lambda *a, **k: calls.__setitem__("dims", calls["dims"] + 1) or dims)
    monkeypatch.setattr(masks, "check_inclusion", lambda *a, **k: calls.__setitem__("inclusion", calls["inclusion"] + 1) or inclusion)
    monkeypatch.setattr(masks, "check_overlap", lambda *a, **k: calls.__setitem__("overlap", calls["overlap"] + 1) or overlap)
    monkeypatch.setattr(masks, "check_ceiling", lambda *a, **k: calls.__setitem__("ceiling", calls["ceiling"] + 1) or ceiling)
    return calls


# --- FILT-001: 短絡評価（param×4） ---------------------------------------------------------


@pytest.mark.parametrize(
    "case,flags,expected_calls",
    [
        pytest.param(
            "dims_stops", {"dims": False, "inclusion": True, "overlap": True, "ceiling": True},
            {"dims": 1, "inclusion": 0, "overlap": 0, "ceiling": 0}, id="T024-FILT-001-DIMS-STOPS",
        ),
        pytest.param(
            "incl_stops", {"dims": True, "inclusion": False, "overlap": True, "ceiling": True},
            {"dims": 1, "inclusion": 1, "overlap": 0, "ceiling": 0}, id="T024-FILT-001-INCL-STOPS",
        ),
        pytest.param(
            "overlap_stops", {"dims": True, "inclusion": True, "overlap": False, "ceiling": True},
            {"dims": 1, "inclusion": 1, "overlap": 1, "ceiling": 0}, id="T024-FILT-001-OVERLAP-STOPS",
        ),
        pytest.param(
            "ceiling_last", {"dims": True, "inclusion": True, "overlap": True, "ceiling": True},
            {"dims": 1, "inclusion": 1, "overlap": 1, "ceiling": 1}, id="T024-FILT-001-CEILING-LAST",
        ),
    ],
)
def test_filt_001_short_circuits_at_failing_stage(monkeypatch, case, flags, expected_calls):
    from src.packing_core.candidates import filter_candidates

    calls = _install_leaf_spies(monkeypatch, **flags)
    state = _state()
    cand = _cand()
    tp = constants.TimeParams()

    filter_candidates(state, [cand], PP0, tp, _budget_never_over())

    assert calls == expected_calls


# --- FILT-003: 最初の不合格段階がreject_reasonに設定される ---------------------------------


def test_filt_003_reject_reason_is_first_failing_stage(monkeypatch):
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch, dims=True, inclusion=False, overlap=True, ceiling=True)
    state = _state()
    cand = _cand()
    tp = constants.TimeParams()

    filter_candidates(state, [cand], PP0, tp, _budget_never_over())

    assert cand.reject_reason == "inclusion"
    assert cand.feasible is False


# --- FILT-004: reject_countsは最初の不合格段階のみ計上 -------------------------------------


def test_filt_004_reject_counts_only_first_failing_stage(monkeypatch):
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch, dims=True, inclusion=False, overlap=True, ceiling=True)
    state = _state()
    cand = _cand()
    tp = constants.TimeParams()

    pools = filter_candidates(state, [cand], PP0, tp, _budget_never_over())

    assert pools.reject_counts == {"inclusion": 1}


# --- FILT-005/006: dims_candidates / geo_candidates 集合分類（各family=1関数） --------------


def _three_candidate_dims_geo_fixture(monkeypatch):
    """dims不合格1件・geo不合格1件（dims合格）・全合格1件、の3Candidate fixture。"""
    from src.packing_core.candidates import filter_candidates
    from src.packing_core import masks

    cand_dims_fail = _cand(item_idx=0)
    cand_geo_fail = _cand(item_idx=1)
    cand_pass_all = _cand(item_idx=2)
    raw = [cand_dims_fail, cand_geo_fail, cand_pass_all]

    def _dims(cand, ems):
        return cand is not cand_dims_fail

    def _inclusion(space, cand, pp):
        return cand is not cand_geo_fail

    monkeypatch.setattr(masks, "prefilter_dims", _dims)
    monkeypatch.setattr(masks, "check_inclusion", _inclusion)
    monkeypatch.setattr(masks, "check_overlap", lambda *a, **k: True)
    monkeypatch.setattr(masks, "check_ceiling", lambda *a, **k: True)

    state = _state()
    tp = constants.TimeParams()
    pools = filter_candidates(state, raw, PP0, tp, _budget_never_over())
    return pools, cand_dims_fail, cand_geo_fail, cand_pass_all


def test_filt_005_dims_candidates_contains_only_dims_passing(monkeypatch):
    pools, cand_dims_fail, cand_geo_fail, cand_pass_all = _three_candidate_dims_geo_fixture(monkeypatch)

    assert len(pools.dims_candidates) == 2
    assert pools.dims_candidates[0] is cand_geo_fail
    assert pools.dims_candidates[1] is cand_pass_all
    # `in`演算子はCandidateの`==`（配列フィールドを含む）を経由し曖昧な真偽値エラーを招くため
    # 使わず、`is`によるidentity比較のみで非包含を確認する。
    assert all(c is not cand_dims_fail for c in pools.dims_candidates)


def test_filt_006_geo_candidates_contains_only_all_four_stages_passing(monkeypatch):
    pools, cand_dims_fail, cand_geo_fail, cand_pass_all = _three_candidate_dims_geo_fixture(monkeypatch)

    assert len(pools.geo_candidates) == 1
    assert pools.geo_candidates[0] is cand_pass_all
    assert all(c is not cand_geo_fail for c in pools.geo_candidates)
    assert all(c is not cand_dims_fail for c in pools.geo_candidates)


# --- FILT-007/008: path_candidates/l_path_cacheは空初期化（各family=1関数） ------------------


def _empty_pools_after_single_pass(monkeypatch):
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch)
    state = _state()
    tp = constants.TimeParams()
    return filter_candidates(state, [_cand()], PP0, tp, _budget_never_over())


def test_filt_007_path_candidates_initialized_empty(monkeypatch):
    pools = _empty_pools_after_single_pass(monkeypatch)
    assert pools.path_candidates == []


def test_filt_008_l_path_cache_initialized_empty(monkeypatch):
    pools = _empty_pools_after_single_pass(monkeypatch)
    assert pools.l_path_cache == {}


# --- FILT-009: L_PATH/score/riskをここで計算しない ------------------------------------------


def test_filt_009_does_not_compute_l_path_score_or_risk(monkeypatch):
    from src.packing_core import masks
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch)
    l_path_calls = []
    monkeypatch.setattr(masks, "check_l_path", lambda *a, **k: l_path_calls.append(1) or True)

    score_calls = []
    try:
        from src.packing_core import score as score_module
        monkeypatch.setattr(score_module, "heuristic_score", lambda *a, **k: score_calls.append(1), raising=False)
    except ImportError:
        pass  # score.py未実装。呼ばれ得ないため0回のまま。

    risk_calls = []
    try:
        from src.packing_core import risk as risk_module
        monkeypatch.setattr(risk_module, "provisional_p_ng", lambda *a, **k: risk_calls.append(1), raising=False)
    except ImportError:
        pass

    state = _state()
    tp = constants.TimeParams()
    filter_candidates(state, [_cand()], PP0, tp, _budget_never_over())

    assert l_path_calls == []
    assert score_calls == []
    assert risk_calls == []


# --- FILT-010: 包含関係（部分列・順序維持） --------------------------------------------------


def test_filt_010_dims_subset_of_raw_geo_subset_of_dims_order_preserved(monkeypatch):
    from src.packing_core.candidates import filter_candidates
    from src.packing_core import masks

    # c0: dims fail / c1: dims pass, inclusion fail / c2: dims+inclusion pass, overlap fail /
    # c3: 全段階pass
    c0, c1, c2, c3 = _cand(item_idx=0), _cand(item_idx=1), _cand(item_idx=2), _cand(item_idx=3)
    raw = [c0, c1, c2, c3]

    def _dims(cand, ems):
        return cand is not c0

    def _inclusion(space, cand, pp):
        return cand is not c1

    def _overlap(state, cand, tol):
        return cand is not c2

    monkeypatch.setattr(masks, "prefilter_dims", _dims)
    monkeypatch.setattr(masks, "check_inclusion", _inclusion)
    monkeypatch.setattr(masks, "check_overlap", _overlap)
    monkeypatch.setattr(masks, "check_ceiling", lambda *a, **k: True)

    state = _state()
    tp = constants.TimeParams()
    pools = filter_candidates(state, raw, PP0, tp, _budget_never_over())

    # dims_candidatesはraw中でDIMS合格した部分列（順序維持）: [c1, c2, c3]
    assert [id(c) for c in pools.dims_candidates] == [id(c1), id(c2), id(c3)]
    # geo_candidatesはdims_candidates中で全段階合格した部分列: [c3]
    assert [id(c) for c in pools.geo_candidates] == [id(c3)]


# --- FILT-011/012/013/014: 途中終了時の集合契約（各family=1関数、fixtureは共通ヘルパ） -------


def _mid_termination_fixture(monkeypatch):
    """budget_poll_every=1・6候補・2回目のover_soft確認で超過するfixture。"""
    from src.packing_core.candidates import filter_candidates
    from src.packing_core.watchdog import StepBudget

    _install_leaf_spies(monkeypatch)  # 全段階pass想定（timeout前に処理された分は合格扱い）

    raw = [_cand(item_idx=i) for i in range(6)]
    tp = constants.TimeParams(budget_poll_every=1)
    clock = _FlipClock(flip_after_calls=2, before=0.0, after=1000.0)
    budget = StepBudget(t0=0.0, soft=1.0, hard=2.0, now_fn=clock)

    state = _state()
    pools = filter_candidates(state, raw, PP0, tp, budget)
    return raw, pools


def test_filt_011_raw_candidates_keeps_full_input_on_timeout(monkeypatch):
    raw, pools = _mid_termination_fixture(monkeypatch)

    assert len(pools.raw_candidates) == len(raw) == 6
    for i in range(6):
        assert pools.raw_candidates[i] is raw[i]


def test_filt_012_dims_geo_reject_reflect_processed_prefix_only(monkeypatch):
    raw, pools = _mid_termination_fixture(monkeypatch)

    processed_n = len(pools.dims_candidates)
    assert 0 <= processed_n < len(raw)
    # 処理済みprefixはraw先頭からの部分列と一致する（順序維持）。
    for i in range(processed_n):
        assert pools.dims_candidates[i] is raw[i]


def test_filt_013_unprocessed_candidates_remain_at_default_state(monkeypatch):
    raw, pools = _mid_termination_fixture(monkeypatch)

    processed_n = len(pools.dims_candidates)
    for cand in raw[processed_n:]:
        assert cand.feasible is False
        assert cand.reject_reason == ""


def test_filt_014_n_cand0_does_not_decrease_on_timeout(monkeypatch):
    raw, pools = _mid_termination_fixture(monkeypatch)

    n_cand0 = len(pools.raw_candidates)
    assert n_cand0 == len(raw) == 6


# --- FILT-015: 開始時点で既にover_soft超過なら空集合＋raw全体保持 ---------------------------


def test_filt_015_already_over_soft_at_start(monkeypatch):
    from src.packing_core.candidates import filter_candidates
    from src.packing_core.watchdog import StepBudget

    _install_leaf_spies(monkeypatch)
    raw = [_cand(item_idx=0), _cand(item_idx=1)]
    tp = constants.TimeParams()
    budget = StepBudget(t0=0.0, soft=0.0, hard=1.0, now_fn=lambda: 100.0)  # 常に超過

    state = _state()
    pools = filter_candidates(state, raw, PP0, tp, budget)

    assert len(pools.raw_candidates) == 2
    for i in range(2):
        assert pools.raw_candidates[i] is raw[i]
    assert pools.dims_candidates == []
    assert pools.geo_candidates == []
    assert pools.reject_counts == {}
    for cand in raw:
        assert cand.feasible is False
        assert cand.reject_reason == ""


# --- FILT-016: budget_poll_every件ごとに再確認する（単発チェックでないこと） ----------------


def test_filt_016_rechecks_periodically_not_only_at_start(monkeypatch):
    """開始時点のチェックは通過（under）させ、後続の再チェックでのみ超過を検出させる
    ことで、単発チェックではなく周期的な再確認が行われることを確認する。"""
    from src.packing_core.candidates import filter_candidates
    from src.packing_core.watchdog import StepBudget

    _install_leaf_spies(monkeypatch)
    raw = [_cand(item_idx=i) for i in range(8)]
    tp = constants.TimeParams(budget_poll_every=2)
    # 呼出し1回目（開始時チェック）はunder、2回目以降（ループ中の再チェック）でover。
    clock = _FlipClock(flip_after_calls=1, before=0.0, after=1000.0)
    budget = StepBudget(t0=0.0, soft=1.0, hard=2.0, now_fn=clock)

    state = _state()
    pools = filter_candidates(state, raw, PP0, tp, budget)

    # 開始時チェックだけでは超過しない設定のため、少なくとも1件は処理されたはず
    # （単発チェックのみなら開始時に超過していないので全件処理されてしまうはずだが、
    # 再チェックが働けば全件未満で打ち切られる）。
    assert 0 <= len(pools.dims_candidates) < 8


# --- FILT-017: budget_poll_every<=0はValueError ---------------------------------------------


def test_filt_017_non_positive_poll_every_raises_value_error(monkeypatch):
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch)
    state = _state()
    raw = [_cand()]

    for poll_every in (0, -1):
        tp = constants.TimeParams(budget_poll_every=poll_every)
        with pytest.raises(ValueError):
            filter_candidates(state, raw, PP0, tp, _budget_never_over())


# --- FILT-018: over_hard専用分岐を設けない（soft超過のみで判定する） ------------------------


def test_filt_018_no_separate_hard_only_branch(monkeypatch):
    from src.packing_core.candidates import filter_candidates
    from src.packing_core.watchdog import StepBudget

    _install_leaf_spies(monkeypatch)
    raw = [_cand(item_idx=i) for i in range(3)]
    tp = constants.TimeParams()
    # soft=10(未達) だが hard=1(既に超過) という非通常の関係を人為的に作る。
    # over_hard()専用の分岐が無ければ、over_soft()==Falseのため全件処理される。
    budget = StepBudget(t0=0.0, soft=10.0, hard=1.0, now_fn=lambda: 5.0)
    assert budget.over_hard() is True
    assert budget.over_soft() is False

    state = _state()
    pools = filter_candidates(state, raw, PP0, tp, budget)

    assert len(pools.dims_candidates) == 3  # 打ち切られず全件処理される


# --- FILT-019（訂正）: 入力全体を同順序・要素単位のidentityで保持（listコンテナ自体のidentityは要求しない） ---


def test_filt_019_raw_candidates_preserves_input_order_and_element_identity(monkeypatch):
    from src.packing_core.candidates import filter_candidates

    _install_leaf_spies(monkeypatch)
    raw = [_cand(item_idx=i) for i in range(4)]
    tp = constants.TimeParams()
    state = _state()

    pools = filter_candidates(state, list(raw), PP0, tp, _budget_never_over())  # 別listオブジェクトを渡す

    assert len(pools.raw_candidates) == len(raw)
    for i in range(len(raw)):
        assert pools.raw_candidates[i] is raw[i]  # 要素単位のidentity
    # listコンテナ自体のidentity（pools.raw_candidates is 入力list）は要求しない。


# --- FILT-020: 手計算で一意な合否fixture（param×2、evaluate_stageをoracleに使わない） --------


def _pass_fixture_state_and_candidate():
    """全4段階を明確に満たす配置（手計算: 内壁±0.4に対し十分内側、既配置なし、天井余裕大）。"""
    space = _flat_space()
    state = PackingState(
        containers=[space], placed={0: []}, pool=[],
        ems={0: [_ems(min_rel=(-1.0, -1.0, 0.0), max_rel=(1.0, 1.0, 1.0))]},
        ems_truncation={0: 0.0}, meta={},
    )
    # box: x∈[-0.05,0.05], y∈[-0.05,0.05], z∈[0.45,0.55]。内壁±0.4に対し0.35の余裕。
    cand = _cand(pos_rel=(0.0, 0.0, 0.5), osize=(0.1, 0.1, 0.1))
    return state, cand


def _reject_fixture_state_and_candidate():
    """INCLUSIONのみを明確に違反する配置（DIMSは寸法のみでEMSは十分大、OVERLAPは既配置なし
    で自動合格、CEILINGは天井余裕大——手計算でINCLUSIONだけが不合格と分かる）。"""
    space = _flat_space()
    state = PackingState(
        containers=[space], placed={0: []}, pool=[],
        ems={0: [_ems(min_rel=(-1.0, -1.0, 0.0), max_rel=(1.0, 1.0, 1.0))]},
        ems_truncation={0: 0.0}, meta={},
    )
    # box: x∈[0.40,0.50]。内壁上限0.40を0.10はみ出す（有効margin~0.01よりずっと大きい）。
    cand = _cand(pos_rel=(0.45, 0.0, 0.5), osize=(0.1, 0.1, 0.1))
    return state, cand


@pytest.mark.parametrize(
    "fixture_fn,expected_in_geo,expected_reject_reason,expected_reject_counts",
    [
        pytest.param(_pass_fixture_state_and_candidate, True, "", {}, id="T024-FILT-020-PASS"),
        pytest.param(
            _reject_fixture_state_and_candidate, False, "inclusion", {"inclusion": 1},
            id="T024-FILT-020-REJECT",
        ),
    ],
)
def test_filt_020_hand_computed_pass_reject_fixtures(
    fixture_fn, expected_in_geo, expected_reject_reason, expected_reject_counts
):
    from src.packing_core.candidates import filter_candidates

    state, cand = fixture_fn()
    tp = constants.TimeParams()
    pools = filter_candidates(state, [cand], PP0, tp, _budget_never_over())

    assert (len(pools.dims_candidates) == 1 and pools.dims_candidates[0] is cand)
    if expected_in_geo:
        assert len(pools.geo_candidates) == 1
        assert pools.geo_candidates[0] is cand
    else:
        assert pools.geo_candidates == []
    assert cand.reject_reason == expected_reject_reason
    assert pools.reject_counts == expected_reject_counts
