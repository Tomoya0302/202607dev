"""統合T-024: candidates.enumerate_candidates の契約テスト（詳細仕様書 v1.18 §4.11「基本順序」・
§4.12「候補列挙・候補生成契約」、契約計画 §3.C）。

`candidates.py` は未実装のため対象関数の import は各テスト関数内で行う（`tests/test_masks.py`
と同方針）。全15家族が分類 I。

基本順序（決定論）: pool現在順 → container_idx昇順 → orientation昇順(0..5) →
    state.ems[container_idx]（select_topn出力順）のems_id昇順(0始まり)。

budget確認位置（§4.12）: 開始時に直ちに over_soft() を確認（超過なら空リスト）。以後
    tp.budget_poll_every 件ごとに再確認し、超過検出時点までの部分結果を返す。
    budget_poll_every<=0 は ValueError。
"""
import numpy as np
import pytest

from agents.heuristic.packing_core import constants
from agents.heuristic.packing_core.container_space import build_container_space
from agents.heuristic.packing_core.state import PackingState
from agents.heuristic.packing_core.types import EMSBox, ItemSpec

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
CELL = 0.02


def _box_cdict():
    """軸整列6面（cut_x=cut_y=0、shelf=False）の合成 cdict（`test_stability.py` と同方針）。"""
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


def _item(idx, size=(0.05, 0.05, 0.05), weight=1.0):
    return ItemSpec(
        idx=idx, size=np.array(size, dtype=np.float64), weight=weight,
        kind=None, is_soft=False, is_priority=False,
    )


def _ems(min_rel, max_rel):
    return EMSBox(
        min_rel=np.array(min_rel, dtype=np.float64), max_rel=np.array(max_rel, dtype=np.float64)
    )


def _state(pool, ems_by_container, n_containers=1):
    containers = [_flat_space() for _ in range(n_containers)]
    placed = {i: [] for i in range(n_containers)}
    ems_trunc = {i: 0.0 for i in range(n_containers)}
    return PackingState(
        containers=containers, placed=placed, pool=pool,
        ems=ems_by_container, ems_truncation=ems_trunc, meta={},
    )


def _keys(cands):
    """(item_idx, container_idx, orientation, ems_id) の射影（Candidateの`==`比較は
    numpy配列フィールドを含み曖昧になるため使わない）。"""
    return [(c.item_idx, c.container_idx, c.orientation, c.ems_id) for c in cands]


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


# --- 共通fixture: 全組合せが合格する2アイテム×2コンテナ×6姿勢×2EMS ------------------------


def _full_product_fixture():
    """全48組合せがcandidate_from_emsで合格する（Noneが出ない）fixture。

    pool位置順は idx 昇順とわざと逆にする（idx=9が位置0、idx=3が位置1）ことで、
    「pool現在順（list位置）」が「idx値」とは独立の契約であることを列挙順で担保する。
    """
    pool = [_item(idx=9), _item(idx=3)]
    ems_a = _ems([-0.3, -0.3, 0.02], [-0.1, -0.1, 0.22])
    ems_b = _ems([0.1, 0.1, 0.02], [0.3, 0.3, 0.22])
    ems_by_container = {0: [ems_a, ems_b], 1: [ems_a, ems_b]}
    state = _state(pool, ems_by_container, n_containers=2)
    return state, pool


def _expected_full_order(pool):
    expected = []
    for item in pool:
        for container_idx in range(2):
            for orientation in range(6):
                for ems_id in range(2):
                    expected.append((item.idx, container_idx, orientation, ems_id))
    return expected


# --- ENUM-001: 基本順序 -------------------------------------------------------------------


def _budget_never_over():
    from agents.heuristic.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)


# --- ENUM-002: 決定論（2回列挙で完全一致） -------------------------------------------------


def test_enum_002_deterministic_across_two_calls():
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()

    result_a = enumerate_candidates(state, pp, tp, _budget_never_over())
    result_b = enumerate_candidates(state, pp, tp, _budget_never_over())
    assert _keys(result_a) == _keys(result_b)


# --- ENUM-003: orientation 0..5 昇順で完全網羅 ---------------------------------------------


# --- ENUM-004: ems_id は state.ems[container] の0始まりindex（select_topn順と同義） --------


def test_enum_004_ems_id_is_positional_index_into_state_ems():
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    left = _ems([-0.35, -0.35, 0.02], [-0.25, -0.25, 0.12])   # X<0側
    right = _ems([0.25, 0.25, 0.02], [0.35, 0.35, 0.12])       # X>0側
    item = _item(idx=0)
    pp = constants.PlacementParams()
    tp = constants.TimeParams()

    # 順序A: [left, right] → ems_id0はX<0側、ems_id1はX>0側。
    state_a = _state([item], {0: [left, right]}, n_containers=1)
    result_a = enumerate_candidates(state_a, pp, tp, _budget_never_over())
    by_id_a = {c.ems_id: c for c in result_a if c.orientation == 0}
    assert by_id_a[0].pos_rel[0] < 0.0
    assert by_id_a[1].pos_rel[0] > 0.0

    # 順序B（left/rightを入替）: ems_idの割当がlist位置に追随して反転する。
    state_b = _state([item], {0: [right, left]}, n_containers=1)
    result_b = enumerate_candidates(state_b, pp, tp, _budget_never_over())
    by_id_b = {c.ems_id: c for c in result_b if c.orientation == 0}
    assert by_id_b[0].pos_rel[0] > 0.0
    assert by_id_b[1].pos_rel[0] < 0.0


# --- ENUM-005: item×container×orientation×EMS の完全直積 ----------------------------------


# --- ENUM-006: candidate_from_ems が None を返す組合せは除外され、順序は維持される ---------


# --- ENUM-007: 同一値EMSが複数存在してもset/dict変換で消えない -----------------------------


def test_enum_007_duplicate_valued_ems_not_deduplicated():
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    dup_a = _ems([-0.05, -0.05, 0.02], [0.05, 0.05, 0.12])
    dup_b = _ems([-0.05, -0.05, 0.02], [0.05, 0.05, 0.12])  # 値は同一・別オブジェクト
    item = _item(idx=0, size=(0.02, 0.02, 0.02))

    state = _state([item], {0: [dup_a, dup_b]}, n_containers=1)
    pp = constants.PlacementParams()
    tp = constants.TimeParams()

    result = enumerate_candidates(state, pp, tp, _budget_never_over())
    ems_ids = sorted({c.ems_id for c in result if c.orientation == 0})
    assert ems_ids == [0, 1]  # 2件とも残る（1件にデデュープされない）


# --- ENUM-008/009/010: mask/score/L_PATH を呼ばない -----------------------------------------


def test_enum_008_does_not_call_mask_functions(monkeypatch):
    from agents.heuristic.packing_core import masks
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    calls = {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0}
    monkeypatch.setattr(masks, "prefilter_dims", lambda *a, **k: calls.__setitem__("dims", calls["dims"] + 1) or True)
    monkeypatch.setattr(masks, "check_inclusion", lambda *a, **k: calls.__setitem__("inclusion", calls["inclusion"] + 1) or True)
    monkeypatch.setattr(masks, "check_overlap", lambda *a, **k: calls.__setitem__("overlap", calls["overlap"] + 1) or True)
    monkeypatch.setattr(masks, "check_ceiling", lambda *a, **k: calls.__setitem__("ceiling", calls["ceiling"] + 1) or True)

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()
    enumerate_candidates(state, pp, tp, _budget_never_over())

    assert calls == {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0}


def test_enum_009_does_not_call_heuristic_score(monkeypatch):
    from agents.heuristic.packing_core import score as score_module
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    calls = []
    monkeypatch.setattr(score_module, "heuristic_score", lambda *a, **k: calls.append(1), raising=False)

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()
    enumerate_candidates(state, pp, tp, _budget_never_over())

    assert calls == []


def test_enum_010_does_not_call_check_l_path(monkeypatch):
    from agents.heuristic.packing_core import masks
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    calls = []
    monkeypatch.setattr(masks, "check_l_path", lambda *a, **k: calls.append(1) or True)

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()
    enumerate_candidates(state, pp, tp, _budget_never_over())

    assert calls == []


# --- ENUM-011: 新しいStepBudgetを生成しない ------------------------------------------------


def test_enum_011_does_not_construct_new_step_budget(monkeypatch):
    from agents.heuristic.packing_core import watchdog
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    budget = watchdog.StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)

    calls = {"n": 0}
    original_init = watchdog.StepBudget.__init__

    def _counting_init(self, *args, **kwargs):
        calls["n"] += 1
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(watchdog.StepBudget, "__init__", _counting_init)

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()
    enumerate_candidates(state, pp, tp, budget)

    assert calls["n"] == 0


# --- ENUM-012: 開始時点で既にover_soft超過なら空リスト -------------------------------------


def test_enum_012_already_over_soft_at_start_returns_empty_list():
    from agents.heuristic.packing_core.candidates import enumerate_candidates
    from agents.heuristic.packing_core.watchdog import StepBudget

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams()
    budget = StepBudget(t0=0.0, soft=0.0, hard=1.0, now_fn=lambda: 100.0)  # 常に超過

    result = enumerate_candidates(state, pp, tp, budget)
    assert result == []


# --- ENUM-013: budget_poll_every件ごとの再確認で途中終了（部分結果は全体の接頭辞） ----------


# --- ENUM-014: budget_poll_every<=0 は ValueError（param×2） --------------------------------


@pytest.mark.parametrize(
    "poll_every",
    [
        pytest.param(0, id="T024-ENUM-014-ZERO"),
        pytest.param(-1, id="T024-ENUM-014-NEG"),
    ],
)
def test_enum_014_non_positive_poll_every_raises_value_error(poll_every):
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    state, _ = _full_product_fixture()
    pp = constants.PlacementParams()
    tp = constants.TimeParams(budget_poll_every=poll_every)

    with pytest.raises(ValueError):
        enumerate_candidates(state, pp, tp, _budget_never_over())


# --- ENUM-015: 空poolは空リスト -------------------------------------------------------------


def test_enum_015_empty_pool_returns_empty_list():
    from agents.heuristic.packing_core.candidates import enumerate_candidates

    ems = _ems([-0.1, -0.1, 0.02], [0.1, 0.1, 0.22])
    state = _state([], {0: [ems]}, n_containers=1)
    pp = constants.PlacementParams()
    tp = constants.TimeParams()

    result = enumerate_candidates(state, pp, tp, _budget_never_over())
    assert result == []
