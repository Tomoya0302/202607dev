"""統合T-024: watchdog.layer1_main〜layer4_max_p・l_path_cache の契約テスト
（詳細仕様書 v1.18 §4.11「4層の定義」「各層の詳細契約」「L_PATHキャッシュ」、契約計画 §3.H）。

`watchdog.py` は T-022（`StepBudget`）のみ実装済みで4層は未実装のため、対象関数の import は
各テスト関数内で行う。`CandidatePools`/`candidate_key`（`candidates.py`、未実装）も同様に
関数内で import する。全27家族が分類 I。

シグネチャ（§4.11）:
    def layer1_main(state, budget, *, pools, pp, stage_params) -> Candidate | None
    def layer2_dblf_strict(state, budget, *, pools, pp, stage_params) -> Candidate | None
    def layer3_first_fit(state, budget, *, pools, pp, stage_params) -> Candidate | None
    def layer4_max_p(state, budget, *, pools, pp, stage_params) -> Candidate | None

`check_l_path` の呼出し元import形式を固定しないため、spyは `masks.check_l_path` と
`watchdog.check_l_path` の両方へ（後者は raising=False で）patchする。
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.state import PackingState
from src.packing_core.types import Candidate

PP0 = constants.PlacementParams()


def _cand(item_idx=0, container_idx=0, ems_id=0, orientation=0, score=0.0, p_success=1.0):
    c = Candidate(
        item_idx=item_idx, container_idx=container_idx, ems_id=ems_id, orientation=orientation,
        pos_rel=np.array([0.0, 0.0, 0.5], dtype=np.float64),
        osize=np.array([0.1, 0.1, 0.1], dtype=np.float64),
    )
    c.score = score
    c.p_success = p_success
    return c


def _key(cand):
    return (cand.item_idx, cand.container_idx, cand.orientation, cand.ems_id)


def _pools(geo=None, dims=None, path=None, l_path_cache=None, reject_counts=None, raw=None):
    from src.packing_core.candidates import CandidatePools
    return CandidatePools(
        raw_candidates=raw if raw is not None else [],
        dims_candidates=dims if dims is not None else [],
        geo_candidates=geo if geo is not None else [],
        path_candidates=path if path is not None else [],
        l_path_cache=l_path_cache if l_path_cache is not None else {},
        reject_counts=reject_counts if reject_counts is not None else {},
    )


def _minimal_state():
    return PackingState(containers=[], placed={}, pool=[], ems={}, ems_truncation={}, meta={})


def _never_over_budget():
    from src.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e6, hard=2e6, now_fn=lambda: 0.0)


def _already_over_soft_budget():
    from src.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=0.0, hard=1.0, now_fn=lambda: 100.0)


def _already_over_hard_budget():
    from src.packing_core.watchdog import StepBudget
    return StepBudget(t0=0.0, soft=0.0, hard=0.0, now_fn=lambda: 100.0)  # soft/hard共に既に超過


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


def _patch_check_l_path(monkeypatch, fn):
    """import文の形式（`masks.check_l_path`直呼びか`from...import`か）に依存せずスパイ化する。"""
    from src.packing_core import masks, watchdog
    monkeypatch.setattr(masks, "check_l_path", fn, raising=False)
    monkeypatch.setattr(watchdog, "check_l_path", fn, raising=False)


def _spy_check_l_path(monkeypatch, result_by_key=None, default=True, calls_list=None, raise_for=None):
    """key→合否のdict、または既定合否でcheck_l_pathをスパイ化する。calls_listに呼出し順を記録する。"""
    if calls_list is None:
        calls_list = []
    if raise_for is None:
        raise_for = set()

    def _fn(state, cand, pp):
        key = _key(cand)
        calls_list.append(key)
        if key in raise_for:
            raise RuntimeError("injected check_l_path failure")
        if result_by_key is not None and key in result_by_key:
            return result_by_key[key]
        return default

    _patch_check_l_path(monkeypatch, _fn)
    return calls_list


# --- LAYER-001..005: L_PATHキャッシュ ---------------------------------------------------------


def test_layer_001_cache_records_pass_as_true(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand])
    _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert pools.l_path_cache[_key(cand)] is True


def test_layer_002_cache_records_fail_as_false(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand])
    _spy_check_l_path(monkeypatch, default=False)
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert pools.l_path_cache[_key(cand)] is False


def test_layer_003_cached_key_not_reevaluated(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand], l_path_cache={_key(cand): True})  # 事前キャッシュ済み
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert calls == []  # 既にキャッシュ済みのため再評価しない


def test_layer_004_none_is_never_stored_as_cache_value(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand_pass = _cand(item_idx=0, score=1.0)
    cand_fail = _cand(item_idx=1, score=0.5)
    pools = _pools(geo=[cand_pass, cand_fail])
    _spy_check_l_path(
        monkeypatch, result_by_key={_key(cand_pass): True, _key(cand_fail): False}
    )
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert all(v is not None for v in pools.l_path_cache.values())
    assert set(pools.l_path_cache.keys()) == {_key(cand_pass), _key(cand_fail)}


def test_layer_005_passing_candidate_added_to_path_candidates_exactly_once(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand])
    _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    matching = [c for c in pools.path_candidates if c is cand]
    assert len(matching) == 1


# --- LAYER-006..014: layer1_main ---------------------------------------------------------------


def test_layer_006_stable_sort_by_score_descending(monkeypatch):
    """評価順序はscore降順・同点は基本順序（sort前のgeo_candidates順）を維持する。"""
    from src.packing_core.watchdog import layer1_main

    c_tie1 = _cand(item_idx=0, score=0.5)
    c_high = _cand(item_idx=1, score=0.9)
    c_tie2 = _cand(item_idx=2, score=0.5)
    pools = _pools(geo=[c_tie1, c_high, c_tie2])  # 基本順序: tie1, high, tie2
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    # score降順: high(0.9) → tie1(0.5,基本順序で先) → tie2(0.5)
    assert calls == [_key(c_high), _key(c_tie1), _key(c_tie2)]


def test_layer_007_evaluates_at_most_l_path_top_m_candidates(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cands = [_cand(item_idx=i, score=float(10 - i)) for i in range(5)]  # 降順scoreで既に整列
    pools = _pools(geo=cands)
    calls = _spy_check_l_path(monkeypatch, default=False)  # 全不合格でも呼出し回数だけ見る
    sp = constants.StageParams(l_path_top_m=2)

    layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert len(calls) == 2


def test_layer_008_returns_max_score_among_passing(monkeypatch):
    """最高scoreの候補がL_PATH不合格でも、合格した中で最大scoreの候補を返す。"""
    from src.packing_core.watchdog import layer1_main

    c_best_but_fails = _cand(item_idx=0, score=0.9)
    c_second_passes = _cand(item_idx=1, score=0.6)
    pools = _pools(geo=[c_best_but_fails, c_second_passes])
    _spy_check_l_path(
        monkeypatch,
        result_by_key={_key(c_best_but_fails): False, _key(c_second_passes): True},
    )
    sp = constants.StageParams(l_path_top_m=10)

    result = layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is c_second_passes


def test_layer_009_top_m_non_positive_never_calls_check_l_path_returns_none(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand])
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=0)

    result = layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is None
    assert calls == []


def test_layer_010_soft_exceeded_returns_best_so_far_without_further_evaluation(monkeypatch):
    from src.packing_core.watchdog import StepBudget, layer1_main

    cands = [_cand(item_idx=i, score=float(10 - i)) for i in range(5)]  # 降順score
    pools = _pools(geo=cands)
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)
    # 1回目の確認（最初の候補評価前後どこか）はunder、2回目でover_soft。
    clock = _FlipClock(flip_after_calls=1, before=0.0, after=1000.0)
    budget = StepBudget(t0=0.0, soft=1.0, hard=2.0, now_fn=clock)

    result = layer1_main(_minimal_state(), budget, pools=pools, pp=PP0, stage_params=sp)

    assert len(calls) < 5  # 全件は評価しない（途中終了）
    assert result is not None
    assert result is cands[0]  # 評価済みの中でscore最大（先頭のみ評価済みなら先頭）


def test_layer_011_hard_exceeded_from_start_makes_no_new_check_l_path_calls(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cands = [_cand(item_idx=i, score=float(10 - i)) for i in range(3)]
    pools = _pools(geo=cands)  # キャッシュ未登録
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    result = layer1_main(_minimal_state(), _already_over_hard_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert calls == []  # 新規評価は一切開始しない
    assert result is None  # キャッシュも無いため確定候補なし


def test_layer_012_reuses_cached_result_for_selection(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand_cached = _cand(item_idx=0, score=0.9)
    cand_fresh = _cand(item_idx=1, score=0.5)
    pools = _pools(geo=[cand_cached, cand_fresh], l_path_cache={_key(cand_cached): True})
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    result = layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert _key(cand_cached) not in calls  # キャッシュ済み分は再評価されない
    assert result is cand_cached  # scoreが高くキャッシュ済みで合格 → これが選ばれる


def test_layer_013_no_passing_candidate_returns_none(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cands = [_cand(item_idx=i, score=float(i)) for i in range(3)]
    pools = _pools(geo=cands)
    _spy_check_l_path(monkeypatch, default=False)
    sp = constants.StageParams(l_path_top_m=10)

    result = layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is None


def test_layer_014_exception_in_layer_is_caught_and_returns_none(monkeypatch):
    from src.packing_core.watchdog import layer1_main

    cand = _cand(item_idx=0, score=1.0)
    pools = _pools(geo=[cand])
    _spy_check_l_path(monkeypatch, raise_for={_key(cand)})
    sp = constants.StageParams(l_path_top_m=10)

    result = layer1_main(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is None  # 層内で例外を捕捉しNoneを返す（外へ伝播しない）


# --- LAYER-015..019: layer2_dblf_strict --------------------------------------------------------


def test_layer_015_evaluates_unevaluated_candidates_in_basic_order(monkeypatch):
    from src.packing_core.watchdog import layer2_dblf_strict

    c0, c1, c2 = _cand(item_idx=0), _cand(item_idx=1), _cand(item_idx=2)
    pools = _pools(geo=[c0, c1, c2])  # 未評価3件、基本順序どおり
    calls = _spy_check_l_path(monkeypatch, default=False)  # 全不合格（順序だけ見る）
    sp = constants.StageParams(l_path_top_m=10)

    layer2_dblf_strict(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert calls == [_key(c0), _key(c1), _key(c2)]


def test_layer_016_reuses_cached_results_without_reevaluation(monkeypatch):
    from src.packing_core.watchdog import layer2_dblf_strict

    c0, c1 = _cand(item_idx=0), _cand(item_idx=1)
    pools = _pools(geo=[c0, c1], l_path_cache={_key(c0): False})
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer2_dblf_strict(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert _key(c0) not in calls
    assert _key(c1) in calls


def test_layer_017_returns_first_passing_in_basic_order(monkeypatch):
    from src.packing_core.watchdog import layer2_dblf_strict

    c0, c1, c2 = _cand(item_idx=0), _cand(item_idx=1), _cand(item_idx=2)
    pools = _pools(geo=[c0, c1, c2])
    _spy_check_l_path(
        monkeypatch, result_by_key={_key(c0): False, _key(c1): True, _key(c2): True}
    )
    sp = constants.StageParams(l_path_top_m=10)

    result = layer2_dblf_strict(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is c1  # 基本順序で最初に合格したc1（c2は評価すらされない可能性がある）


def test_layer_018_hard_exceeded_stops_new_evaluation_returns_none_if_nothing_passed(monkeypatch):
    from src.packing_core.watchdog import layer2_dblf_strict

    c0, c1 = _cand(item_idx=0), _cand(item_idx=1)
    pools = _pools(geo=[c0, c1])
    _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    result = layer2_dblf_strict(_minimal_state(), _already_over_hard_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is None  # 新規評価を一切開始できないため合格判定不可


def test_layer_019_empty_timeout_and_exception_all_return_none(monkeypatch):
    from src.packing_core.watchdog import layer2_dblf_strict

    sp = constants.StageParams(l_path_top_m=10)

    # (a) 候補なし
    _spy_check_l_path(monkeypatch, default=True)
    pools_empty = _pools(geo=[])
    assert layer2_dblf_strict(_minimal_state(), _never_over_budget(), pools=pools_empty, pp=PP0, stage_params=sp) is None

    # (b) 開始時点で時間切れ
    c0 = _cand(item_idx=0)
    pools_timeout = _pools(geo=[c0])
    assert layer2_dblf_strict(_minimal_state(), _already_over_hard_budget(), pools=pools_timeout, pp=PP0, stage_params=sp) is None

    # (c) 層内例外
    c1 = _cand(item_idx=1)
    pools_exc = _pools(geo=[c1])
    _spy_check_l_path(monkeypatch, raise_for={_key(c1)})
    assert layer2_dblf_strict(_minimal_state(), _never_over_budget(), pools=pools_exc, pp=PP0, stage_params=sp) is None


# --- LAYER-020..022: layer3_first_fit -----------------------------------------------------------


def test_layer_020_never_calls_check_l_path(monkeypatch):
    from src.packing_core.watchdog import layer3_first_fit

    c0, c1, c2 = _cand(item_idx=0), _cand(item_idx=1), _cand(item_idx=2)
    pools = _pools(geo=[c0, c1, c2])  # 全て未評価
    calls = _spy_check_l_path(monkeypatch, default=True)
    sp = constants.StageParams(l_path_top_m=10)

    layer3_first_fit(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert calls == []


def test_layer_021_three_group_priority_true_then_unknown_then_false():
    from src.packing_core.watchdog import layer3_first_fit

    c_false = _cand(item_idx=0)
    c_unknown = _cand(item_idx=1)
    c_true = _cand(item_idx=2)
    pools = _pools(
        geo=[c_false, c_unknown, c_true],  # 基本順序: false, unknown, true
        l_path_cache={_key(c_false): False, _key(c_true): True},
    )
    sp = constants.StageParams(l_path_top_m=10)

    result = layer3_first_fit(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)

    assert result is c_true  # True群が最優先


def test_layer_022_empty_geo_returns_none_and_l_path_pass_not_required():
    from src.packing_core.watchdog import layer3_first_fit

    sp = constants.StageParams(l_path_top_m=10)

    assert layer3_first_fit(_minimal_state(), _never_over_budget(), pools=_pools(geo=[]), pp=PP0, stage_params=sp) is None

    # True群が無くてもunknown/False群から返せる（L_PATH合格必須ではない）。
    c_false = _cand(item_idx=0)
    pools = _pools(geo=[c_false], l_path_cache={_key(c_false): False})
    result = layer3_first_fit(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_false


# --- LAYER-023..027: layer4_max_p（v1.18改訂、有効性 = isfinite かつ 0<=p_success<=1） ----------


def test_layer_023_returns_max_p_success_among_valid():
    from src.packing_core.watchdog import layer4_max_p

    c_low = _cand(item_idx=0, p_success=0.3)
    c_high = _cand(item_idx=1, p_success=0.7)
    pools = _pools(dims=[c_low, c_high])
    sp = constants.StageParams(l_path_top_m=10)

    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_high


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(float("nan"), id="T024-LAYER-024-NAN"),
        pytest.param(float("inf"), id="T024-LAYER-024-PINF"),
        pytest.param(float("-inf"), id="T024-LAYER-024-NINF"),
    ],
)
def test_layer_024_excludes_non_finite_p_success(bad_value):
    from src.packing_core.watchdog import layer4_max_p

    c_valid = _cand(item_idx=0, p_success=0.4)
    c_invalid = _cand(item_idx=1, p_success=bad_value)
    pools = _pools(dims=[c_valid, c_invalid])
    sp = constants.StageParams(l_path_top_m=10)

    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_valid


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(-0.1, id="T024-LAYER-025-NEG"),
        pytest.param(1.1, id="T024-LAYER-025-GT1"),
    ],
)
def test_layer_025_excludes_out_of_range_finite_p_success(bad_value):
    from src.packing_core.watchdog import layer4_max_p

    c_valid = _cand(item_idx=0, p_success=0.4)
    c_invalid = _cand(item_idx=1, p_success=bad_value)
    pools = _pools(dims=[c_valid, c_invalid])
    sp = constants.StageParams(l_path_top_m=10)

    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_valid


def test_layer_026_all_invalid_returns_none():
    from src.packing_core.watchdog import layer4_max_p

    c_nan = _cand(item_idx=0, p_success=float("nan"))
    c_neg = _cand(item_idx=1, p_success=-0.5)
    c_over = _cand(item_idx=2, p_success=1.5)
    pools = _pools(dims=[c_nan, c_neg, c_over])
    sp = constants.StageParams(l_path_top_m=10)

    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is None


def test_layer_027_selects_from_valid_subset_only_when_some_invalid():
    """有効候補{0.3,0.7}・無効候補{1.5,NaN}混在時、有効内最大(0.7)を返す
    （比較・同点判定は有効候補内だけで行う）。"""
    from src.packing_core.watchdog import layer4_max_p

    c_valid_low = _cand(item_idx=0, p_success=0.3)
    c_valid_high = _cand(item_idx=1, p_success=0.7)
    c_invalid_over = _cand(item_idx=2, p_success=1.5)
    c_invalid_nan = _cand(item_idx=3, p_success=float("nan"))
    pools = _pools(dims=[c_valid_low, c_invalid_over, c_valid_high, c_invalid_nan])
    sp = constants.StageParams(l_path_top_m=10)

    result = layer4_max_p(_minimal_state(), _never_over_budget(), pools=pools, pp=PP0, stage_params=sp)
    assert result is c_valid_high
