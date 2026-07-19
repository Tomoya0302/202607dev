"""T-022: packing_core.watchdog の StepBudget 契約テスト（詳細仕様書 §4.11・§6 T-022）。

fake clock を注入し、実時間待ち・`sleep()`・monkeypatch を使わずに決定論的に
soft/hard 締切の境界を検証する。`safe_decide()` 等 T-026 の機能はここでは
検証しない（本チケットの責務外）。
"""

import inspect
import time

from src.packing_core import constants
from src.packing_core.watchdog import StepBudget


class FakeClock:
    """テスト用の進められる時計。

    `StepBudget` の `now_fn` に注入するための、呼び出すたびに現在値を返す
    ゼロ引数 callable。`advance()` で時刻を進める。
    """

    def __init__(self, start: float = 0.0) -> None:
        self._current = start

    def __call__(self) -> float:
        return self._current

    def advance(self, dt: float) -> None:
        self._current += dt


# --- elapsed() ---------------------------------------------------------------


def test_elapsed_reads_from_fake_clock() -> None:
    """fake clock から経過時間を取得できる。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    assert budget.elapsed() == 0.0


def test_elapsed_follows_clock_advance() -> None:
    """時計を進めると `elapsed()` が追随する。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(1.5)
    assert budget.elapsed() == 1.5

    clock.advance(0.5)
    assert budget.elapsed() == 2.0


# --- soft/hard 境界（>= を採用。elapsed==threshold は超過扱い） -----------------


def test_over_soft_false_just_before_threshold() -> None:
    """soft 閾値の直前では超過扱いにならない。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(1.9)
    assert budget.over_soft() is False


def test_over_soft_true_at_threshold() -> None:
    """soft 閾値と完全に同値なら超過扱いにする（`>=` 契約）。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(2.0)
    assert budget.over_soft() is True


def test_over_soft_true_just_after_threshold() -> None:
    """soft 閾値の直後では超過扱いになる。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(2.1)
    assert budget.over_soft() is True


def test_over_hard_false_just_before_threshold() -> None:
    """hard 閾値の直前では超過扱いにならない。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(4.9)
    assert budget.over_hard() is False


def test_over_hard_true_at_threshold() -> None:
    """hard 閾値と完全に同値なら超過扱いにする（`>=` 契約）。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(5.0)
    assert budget.over_hard() is True


def test_over_hard_true_just_after_threshold() -> None:
    """hard 閾値の直後では超過扱いになる。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(5.1)
    assert budget.over_hard() is True


def test_over_soft_and_over_hard_are_independent() -> None:
    """soft 超過・hard 未達の状態を両判定が独立して表す。"""
    clock = FakeClock(start=100.0)
    budget = StepBudget(t0=100.0, soft=2.0, hard=5.0, now_fn=clock)

    clock.advance(3.0)
    assert budget.over_soft() is True
    assert budget.over_hard() is False


# --- TimeParams との整合 -------------------------------------------------------


def test_matches_time_params_policy_soft_and_hard() -> None:
    """`TimeParams.policy_soft` / `policy_hard` と同値到達時の判定が整合する。"""
    time_params = constants.TimeParams()
    clock = FakeClock(start=0.0)
    budget = StepBudget(
        t0=0.0,
        soft=time_params.policy_soft,
        hard=time_params.policy_hard,
        now_fn=clock,
    )

    clock.advance(time_params.policy_soft)
    assert budget.over_soft() is True
    assert budget.over_hard() is False

    clock.advance(time_params.policy_hard - time_params.policy_soft)
    assert budget.over_hard() is True


# --- 既定時計 ------------------------------------------------------------------


def test_default_now_fn_is_monotonic_clock() -> None:
    """`now_fn` 省略時は `time.monotonic()` を既定時計として使う。"""
    t0 = time.monotonic()
    budget = StepBudget(t0=t0, soft=2.0, hard=5.0)

    elapsed = budget.elapsed()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


# --- 静的な実装制約の確認 -------------------------------------------------------


def test_watchdog_source_does_not_use_time_time() -> None:
    """`watchdog.py` が `time.time()` を使用していないことを静的に確認する。"""
    from src.packing_core import watchdog

    source = inspect.getsource(watchdog)
    assert "time.time(" not in source
    assert "sleep(" not in source
